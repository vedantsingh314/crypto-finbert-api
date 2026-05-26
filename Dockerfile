# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Install wheels into a local directory so the runtime stage can copy them
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source
COPY app/  ./app/
COPY news.py utils.py ./     
COPY model/ ./model/                        

# Security: run as non-root
RUN useradd -m -u 1001 appuser
USER appuser

EXPOSE 8000

# 1 worker when using GPU (shared CUDA context), more for CPU-only
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]