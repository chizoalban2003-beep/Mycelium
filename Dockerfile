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
        build-essential curl openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core + server extras via pyproject.toml
# Copy only the metadata first for better layer caching
COPY pyproject.toml README.md ./
COPY physml/__init__.py physml/_log.py physml/py.typed ./physml/

# Install the package with server + companion extras
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[server,companion,dotenv]" || \
    pip install --no-cache-dir -e ".[server,dotenv]"

# Now copy the rest of the source (invalidates cache only when code changes)
COPY physml/ physml/

# ---------------------------------------------------------------------------
# api — REST API server (production-hardened)
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000
EXPOSE 8443

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fk ${MYCO_HEALTH_URL:-http://localhost:8000/health} || exit 1

# Defaults — override via docker-compose environment or .env
ENV MYCO_DATA_DIR=/data/.mycelium
ENV MYCO_HOST=0.0.0.0
ENV MYCO_PORT=8000
ENV MYCO_REQUIRE_AUTH=1

# Entrypoint: use `mycelium serve` so .env and TLS are picked up automatically
ENTRYPOINT ["python", "-m", "physml.cli"]
CMD ["serve"]

# ---------------------------------------------------------------------------
# worker — background companion: goal loop + scheduled goals + file watcher
# ---------------------------------------------------------------------------
FROM base AS worker

COPY scripts/run_worker.py ./scripts/

ENV MYCO_DATA_DIR=/data/.mycelium

CMD ["python", "scripts/run_worker.py"]
