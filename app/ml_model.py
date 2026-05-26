"""
ml_model.py — Model loading and inference core.

Key decisions:
  - Singleton via app.state (one load per process)
  - torch.no_grad() + model.eval() → no gradient tracking, faster + less RAM
  - Batched tokenisation → single forward pass for batch requests
  - Softmax on CPU avoids unnecessary GPU↔CPU round-trips for small payloads
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

from app.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Loads the fine-tuned FinBERT model once and exposes predict() / predict_batch().
    Thread-safe for read-only inference under a single worker process.
    """

    def __init__(self, model_dir: str, device: str = "auto", max_length: int = 128):
        # Resolve to absolute path — avoids issues with relative paths inside Docker
        model_path = Path(model_dir).resolve()

        if not model_path.exists():
            logger.info("Downloading from Hugging Face...")
            snapshot_download(
                repo_id="ghost5151/crypto-finbert",
                local_dir=str(model_path),
                local_dir_use_symlinks=False,   # ← copies files directly, no symlinks
            )
            logger.info("Model downloaded to %s", model_path)

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        logger.info("Loading model on %s", self.device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        # Read labels directly from config — never hardcode
        self.labels = [
            self.model.config.id2label[i]
            for i in range(len(self.model.config.id2label))
        ]
        logger.info("Labels loaded from config: %s", self.labels)

        self.max_length = max_length

        # Warm-up pass to compile CUDA kernels (avoids cold-start latency)
        if self.device == "cuda":
            self._warmup()

    def _warmup(self):
        logger.info("Running GPU warm-up pass...")
        self.predict(["warm up"])

    def predict(self, headlines: list[str]) -> list[dict]:
        """
        Run inference on one or more headlines.
        Returns a list of dicts: {label, confidence, scores, latency_ms}
        """
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            headlines,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits         # (B, 3)

        # Softmax → probabilities; move to CPU for json serialisation
        probs = F.softmax(logits, dim=-1).cpu().float()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_item_ms = elapsed_ms / len(headlines)

        results = []
        for i, headline in enumerate(headlines):
            p = probs[i].tolist()
            label_idx = int(probs[i].argmax())
            results.append({
                "headline":   headline,
                "label":      self.labels[label_idx],
                "confidence": round(p[label_idx], 4),
                "scores": {
                    self.labels[0]: round(p[0], 4),
                    self.labels[1]: round(p[1], 4),
                    self.labels[2]: round(p[2], 4),
                },
                "latency_ms": round(per_item_ms, 2),
            })

        return results