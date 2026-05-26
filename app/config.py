"""
config.py — Centralised settings via pydantic-settings.
All values can be overridden with environment variables or a .env file.
"""

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_DIR: str = "model"          # Path to your trained FinBERT dir
    MAX_LENGTH: int = 128             # Tokenizer max sequence length
    BATCH_SIZE: int = 32              # Max headlines per batch request

    # ── Device: "cuda", "cpu", or "auto" ─────────────────────────────────────
    # "auto" → GPU if available, else CPU
    DEVICE: str = "auto"

    # ── API ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]
    API_WORKERS: int = 1              # Keep at 1 when using GPU (shared memory)

    # ── Caching ───────────────────────────────────────────────────────────────
    CACHE_TTL_SECONDS: int = 86400    # 1 day

    # ── News API ──────────────────────────────────────────────────────────────
    NEWS_API_KEY: str = ""

    @property
    def resolved_device(self) -> str:
        if self.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.DEVICE


settings = Settings()
