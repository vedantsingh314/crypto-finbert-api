"""
schemas.py — All Pydantic request/response models.
Strict typing means FastAPI auto-validates inputs and
generates correct OpenAPI docs without extra work.
"""

from pydantic import BaseModel, Field, field_validator


# ── Request models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-headline prediction request."""
    headline: str = Field(
        ...,
        min_length=3,
        max_length=1024,
        examples=["Bitcoin surges past $100k as institutional demand grows"],
    )

    @field_validator("headline")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class BatchPredictRequest(BaseModel):
    """Batch prediction — up to BATCH_SIZE headlines."""
    headlines: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,           # hard cap; enforce via config too
        examples=[["BTC rises 5%", "ETH faces regulatory pressure"]],
    )

    @field_validator("headlines")
    @classmethod
    def strip_all(cls, v: list[str]) -> list[str]:
        stripped = [h.strip() for h in v if h.strip()]
        if not stripped:
            raise ValueError("At least one non-empty headline is required")
        return stripped


# ── Response models ──────────────────────────────────────────────────────────

class SentimentScores(BaseModel):
    negative: float
    neutral: float
    positive: float


class PredictResponse(BaseModel):
    """Response for a single headline."""
    headline: str
    label: str                       # "bullish" | "bearish" | "neutral"
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: SentimentScores
    latency_ms: float


class BatchPredictResponse(BaseModel):
    """Response for a batch request."""
    results: list[PredictResponse]
    total: int
    total_latency_ms: float


class ErrorResponse(BaseModel):
    detail: str
