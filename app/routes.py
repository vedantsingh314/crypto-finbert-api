"""
routes.py — All API endpoints.

Endpoints:
  POST /predict            → single headline
  POST /predict/batch      → list of headlines (up to BATCH_SIZE)
  GET  /coin-sentiment     → backward-compatible: fetch news + run sentiment
"""

import time

from fastapi import APIRouter, HTTPException, Request  # type: ignore[import]

from app.config import settings
from app.logger import get_logger
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    SentimentScores,
)

# Import your existing news fetcher unchanged
from app.news import get_news_for_coin

logger = get_logger(__name__)

router = APIRouter()

# ── Simple in-process cache (swap for Redis in production) ───────────────────
_cache: dict[str, dict] = {}

COIN_MAP = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL",   "sol": "SOL",
    "dogecoin": "DOGE","doge": "DOGE",
}

COIN_KEYWORDS = {
    "BTC":  ["bitcoin", "btc"],
    "ETH":  ["ethereum", "eth"],
    "SOL":  ["solana", "sol"],
    "DOGE": ["dogecoin", "doge"],
}


def _model(request: Request):
    """Helper to grab the model from app.state."""
    return request.app.state.model


# ── Helper: build PredictResponse from raw dict ──────────────────────────────

def _to_response(raw: dict) -> PredictResponse:
    return PredictResponse(
        headline=raw["headline"],
        label=raw["label"],
        confidence=raw["confidence"],
        scores=SentimentScores(**raw["scores"]),
        latency_ms=raw["latency_ms"],
    )


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single crypto news headline",
    tags=["sentiment"],
)
async def predict(body: PredictRequest, request: Request):
    try:
        results = _model(request).predict([body.headline])
        return _to_response(results[0])
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /predict/batch ───────────────────────────────────────────────────────

@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Classify a batch of headlines in one forward pass",
    tags=["sentiment"],
)
async def predict_batch(body: BatchPredictRequest, request: Request):
    if len(body.headlines) > settings.BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Max {settings.BATCH_SIZE} headlines per batch request",
        )
    try:
        t0 = time.perf_counter()
        raw_results = _model(request).predict(body.headlines)
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return BatchPredictResponse(
            results=[_to_response(r) for r in raw_results],
            total=len(raw_results),
            total_latency_ms=total_ms,
        )
    except Exception as exc:
        logger.exception("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /coin-sentiment ───────────────────────────────────────────────────────
# Backward-compatible endpoint: fetches news then runs the new FinBERT model.

@router.get(
    "/coin-sentiment",
    summary="Fetch live news and return aggregated sentiment for a coin",
    tags=["sentiment"],
)
async def coin_sentiment(coin: str, request: Request):
    try:
        coin = COIN_MAP.get(coin.lower(), coin.upper())

        # Cache check
        cached = _cache.get(coin)
        if cached and time.time() - cached["ts"] < settings.CACHE_TTL_SECONDS:
            logger.info("Cache hit for %s", coin)
            return cached["data"]

        # Fetch news
        headlines = get_news_for_coin(coin)
        if not headlines:
            return {coin: {"sentiment": "no data", "confidence": 0, "scores": None}}

        # Filter by coin keywords then limit
        keywords = COIN_KEYWORDS.get(coin, [])
        filtered = [h for h in headlines if any(k in h.lower() for k in keywords)] or headlines
        filtered = filtered[:10]

        # Run batch inference with the fine-tuned model (one forward pass)
        raw_results = _model(request).predict(filtered)

        # Aggregate: average softmax scores across all headlines
        keys = ["bullish", "bearish", "neutral"]
        avg = {k: round(sum(r["scores"][k] for r in raw_results) / len(raw_results), 4) for k in keys}
        dominant = max(avg, key=avg.get)

        response = {
            coin: {
                "sentiment": dominant,
                "confidence": round(avg[dominant], 4),
                "scores": avg,
                "headlines_analysed": len(filtered),
            }
        }

        _cache[coin] = {"data": response, "ts": time.time()}
        return response

    except Exception as exc:
        logger.exception("coin-sentiment error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
