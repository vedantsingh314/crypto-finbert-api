# Crypto FinBERT — FastAPI Deployment Guide

## Quick start (local, no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in NEWS_API_KEY
uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## Docker

```bash
# Build and run
docker compose up --build

# Force GPU
DEVICE=cuda docker compose up
```

---

## API endpoints

### POST /api/v1/predict — single headline

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "Bitcoin surges past $100k on ETF inflows"}'
```

Response:
```json
{
  "headline": "Bitcoin surges past $100k on ETF inflows",
  "label": "bullish",
  "confidence": 0.9213,
  "scores": { "bullish": 0.9213, "bearish": 0.0321, "neutral": 0.0466 },
  "latency_ms": 1.4
}
```

---

### POST /api/v1/predict/batch — multiple headlines

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "headlines": [
      "Bitcoin surges past $100k on ETF inflows",
      "SEC sues major crypto exchange for securities violations",
      "Ethereum network upgrade goes live without issues"
    ]
  }'
```

---

### GET /api/v1/coin-sentiment — live news + aggregated sentiment

```bash
curl "http://localhost:8000/api/v1/coin-sentiment?coin=bitcoin"
curl "http://localhost:8000/api/v1/coin-sentiment?coin=eth"
```

---

## Postman setup

1. Import → Raw text → paste any cURL above
2. Or create a new POST request:
   - URL: `http://localhost:8000/api/v1/predict`
   - Body → raw → JSON → paste the body above

---

## GPU vs CPU recommendations

| Scenario                  | Recommendation                                      |
|---------------------------|-----------------------------------------------------|
| Development / testing     | CPU is fine; ~5–15ms per headline                   |
| Production, high traffic  | GPU (T4/A10G); ~1–2ms per headline, batch preferred |
| Cloud (cheap)             | AWS g4dn.xlarge (T4 GPU) — ~$0.53/hr on-demand     |
| CPU-only production       | Export to ONNX first — 3× faster than PyTorch CPU   |
| Multi-model on one GPU    | Use `torch_dtype=float16` (already set) to save VRAM|

Keep `API_WORKERS=1` when on GPU. Running multiple workers forks the process,
each allocating its own GPU memory — defeats the purpose unless you have multiple GPUs.

For CPU-only horizontal scaling: set `API_WORKERS=4` (or `2 × CPU cores`) and
run behind an nginx or Caddy reverse proxy.

---

## ONNX export (optional — 3× CPU speedup)

```python
# Run once after training
from optimum.exporters.onnx import main_export
main_export("model/", output="model_onnx/", task="text-classification")
```

Then swap `AutoModelForSequenceClassification` for `ORTModelForSequenceClassification`
in ml_model.py.

---

## Production checklist

- [ ] Set `CORS_ORIGINS` to your frontend domain (not `"*"`)
- [ ] Add an API key middleware if the endpoint is public
- [ ] Replace the in-process cache with Redis (use `fastapi-cache2`)
- [ ] Add Prometheus metrics via `prometheus-fastapi-instrumentator`
- [ ] Set up log shipping (Datadog, Loki, CloudWatch)
- [ ] Pin `MODEL_DIR` to an absolute path in `.env`
- [ ] Run `docker scout` or `trivy` on the image before deploying