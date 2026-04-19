# Mycelium — multi-stage Dockerfile
#
# Stages:
#   base    — shared Python environment
#   api     — REST API server  (docker compose up api)
#   worker  — background companion loop (docker compose up worker)
#
# Quick build:
#   docker build --target api    -t mycelium-api .
#   docker build --target worker -t mycelium-worker .
#
# Or use docker compose (recommended):
#   docker compose up

# ---------------------------------------------------------------------------
# base — shared dependencies, no code
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core + server extras via pyproject.toml
# Copy only the metadata first for better layer caching
COPY pyproject.toml README.md ./
COPY physml/__init__.py physml/_log.py physml/py.typed ./physml/

# Install the package with server + companion extras
# (physml[companion] pulls anthropic, sentence-transformers, etc.)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[server,companion]" || \
    pip install --no-cache-dir -e ".[server]"

# Now copy the rest of the source (invalidates cache only when code changes)
COPY physml/ physml/

# ---------------------------------------------------------------------------
# api — REST API server
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV MYCO_DATA_DIR=/data/.mycelium
ENV MYCO_HOST=0.0.0.0
ENV MYCO_PORT=8000

CMD ["uvicorn", "physml.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]

# ---------------------------------------------------------------------------
# worker — background companion: goal loop + scheduled goals + file watcher
# ---------------------------------------------------------------------------
FROM base AS worker

COPY scripts/run_worker.py ./scripts/

ENV MYCO_DATA_DIR=/data/.mycelium

CMD ["python", "scripts/run_worker.py"]
