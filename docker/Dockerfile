# Winnex Mini — Lightweight Inference Pipeline
# ==============================================
# Multi-stage Docker build optimized for CPU inference.
# Base: python:3.11-slim (~120MB) → ~800MB with SBERT + FAISS
#
# Package layout:
#   /app/
#   ├── server.py             # FastAPI entry point
#   ├── winnex_pipeline/      # pip install -e . compatible
#   │   ├── api.py, config.py, rag.py, ...
#   │   ├── core/, pipeline/, validation/, config/
#   │   ├── configs/
#   │   └── examples/
#   ├── data/
#   └── results/
#
# Python imports:
#   from winnex_pipeline.api import WinnexPipeline  ✅
#   python server.py                                 ✅
#
# Usage:
#   docker build -f docker/Dockerfile -t winnex-mini:latest .
#   docker run -p 8080:8080 winnex-mini:latest
#
# Reference: SGLang-Winnex-AI (docker/Dockerfile.services)
# License: BSL 1.1 | pay@winnex.ai

FROM python:3.11-slim AS builder

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy scikit-learn \
        fastapi uvicorn pydantic \
        sentence-transformers faiss-cpu \
        httpx

# ── Runtime stage ──────────────────────────────────────────────

FROM python:3.11-slim AS runtime

LABEL maintainer="Winnex AI <pay@winnex.ai>"
LABEL description="Winnex Madhava Pipeline — Deterministic vector search with guaranteed bounds"
LABEL version="12.3.0"

WORKDIR /app

RUN apt-get update && apt-get install -y curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy server entry point at /app/ level
COPY server.py .

# Copy package code into winnex_pipeline/ subdirectory
# This makes "from winnex_pipeline.xxx import" work natively
RUN mkdir -p winnex_pipeline
COPY config.py winnex_pipeline/
COPY configs winnex_pipeline/configs/
COPY core winnex_pipeline/core/
COPY pipeline winnex_pipeline/pipeline/
COPY validation winnex_pipeline/validation/
COPY rag.py winnex_pipeline/
COPY api.py winnex_pipeline/
COPY config/ winnex_pipeline/config/
COPY .gitignore BENCHMARK_RESULTS.json CONFIG.md LICENSE README.md requirements.txt winnex_pipeline/

# Ensure data/results dirs
RUN mkdir -p data results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["python3", "server.py"]
