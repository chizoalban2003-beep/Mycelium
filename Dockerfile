# Mycelium Parent Hub (SaaS-ready)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements/ requirements/
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements/prod.txt

# Copy app code + templates/static
COPY mycelium_app/ mycelium_app/
COPY templates/ templates/
COPY static/ static/

# Create storage dir (sqlite fallback, logs)
RUN mkdir -p storage

# Runtime defaults (override in platform env)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    HIVE_ENABLED=true

EXPOSE 8000

# NOTE: For SQLite, keep workers=1. For Postgres, you can raise workers.
# Railway (and many PaaS) inject a dynamic PORT env var. The JSON-array CMD does
# not expand env vars, so we use a shell command here.
CMD ["sh", "-c", "python -m uvicorn mycelium_app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
