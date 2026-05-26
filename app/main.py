"""
main.py — FastAPI entry point
Model is loaded ONCE at startup via lifespan context, stored in app.state.
All routes are registered from routes.py.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logger import get_logger
from app.ml_model import ModelManager
from app.routes import router

logger = get_logger(__name__)


# ── Lifespan: runs once at startup and once at shutdown ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FinBERT model from %s", settings.MODEL_DIR)
    app.state.model = ModelManager(
        model_dir=settings.MODEL_DIR,
        device=settings.DEVICE,
        max_length=settings.MAX_LENGTH,
    )
    logger.info("Model loaded on device: %s", app.state.model.device)
    yield
    # Cleanup (optional — release GPU memory)
    del app.state.model
    logger.info("Model unloaded")


# ── App factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crypto Sentiment API",
    description="Fine-tuned FinBERT for crypto news sentiment: bullish / bearish / neutral",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "message": "Crypto Sentiment API 🚀"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
