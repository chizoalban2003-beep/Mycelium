# SaaS Deployment (Cloud Parent Hub + Edge Children)

This repo already runs as a local “Parent Hub” + “Child” model.
This doc makes it deployable as a SaaS platform.

## What changes in SaaS mode

- Parent Hub runs in the cloud (FastAPI + Postgres).
- Children run on user devices (scripts/agent later) and whisper privacy-safe events back.
- Web UI can be served from the same FastAPI app (templates) or from a separate frontend domain.

## 1) Docker (recommended)

### Build + run locally (SaaS-like)

```bash
docker compose up --build
```

Open:
- `http://localhost:8000/health`
- `http://localhost:8000/hive/health`

### Environment variables

- `DATABASE_URL` should be Postgres in production:
  - `postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME`
- Set a strong `SECRET_KEY`.
- Set `COOKIE_SECURE=true` behind HTTPS.

If your frontend is on another domain, set:
- `CORS_ALLOW_ORIGINS_CSV=https://app.example.com`

## 2) Deploy to a platform

Any Docker-friendly platform works (Fly.io, Render, Railway, AWS App Runner).

High-level checklist:
- Create managed Postgres.
- Set `DATABASE_URL` to the managed Postgres URL.
- Set `SECRET_KEY`.
- Set `HIVE_ENABLED=true`.
- Set `HIVE_INGEST_TOKEN` (shared secret for headless child ingest). For multi-tenant SaaS you’ll likely replace this with per-user/device tokens.

## 3) Child node (edge)

Minimum connectivity test:

```bash
export PARENT_HUB_URL="https://your-api.example.com"
export HIVE_INGEST_TOKEN="..."
export NEXUS_DEVICE_ID="edge-1"
./scripts/run_child.sh
```

This runs a headless whisper import (concept) via `X-Hive-Token`.

For richer observation, use:
- `scripts/passive_telemetry_daemon.py` (posts to `/api/nexus/telemetry/ingest`)

## Notes

- This repo uses `SQLModel.metadata.create_all()` at startup (no Alembic migrations).
- For SaaS: you’ll eventually want real migrations + per-tenant scoping.
