# SaaS Deployment (Cloud Parent Hub + Edge Children)

This repo already runs as a local “Parent Hub” + “Child” model.
This doc makes it deployable as a SaaS platform.

## What changes in SaaS mode

- Parent Hub runs in the cloud (FastAPI + Postgres).
- Children run on user devices (scripts/agent later) and whisper privacy-safe events back.
- Web UI can be served from the same FastAPI app (templates) or from a separate frontend domain.

## 1) Docker (recommended)

### Prerequisites

You need a machine/environment with **Docker Engine** and the **Docker Compose plugin** available.

Quick check:

```bash
docker --version
docker compose version
```

If those commands fail (e.g. `docker: command not found`), install Docker first on your deployment host.

### Build + run locally (SaaS-like)

```bash
docker compose up --build
```

Verify it’s alive:

```bash
curl -sS http://localhost:8000/health
```

Note: `GET /api/hive/health` requires an authenticated user (cookie or bearer token).

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

Notes:
- This repo already includes a production-oriented container entrypoint in `Dockerfile`.
- For platforms like Render/Fly/Railway/App Runner, you typically just point the service at this repo and set the env vars above.

### Railway (recommended for MVP)

1) Create a new Railway Project → **Deploy from GitHub repo**.

2) Add **PostgreSQL** to the project.

3) Set service environment variables:
- `DATABASE_URL` (from Railway Postgres)
  - Railway often provides `postgres://...` URLs; the app will normalize this automatically.
- `SECRET_KEY` (long random string)
- `HIVE_ENABLED=true`
- `HIVE_INGEST_TOKEN` (shared secret for Child → Parent Hive imports)
- `COOKIE_SECURE=true` (Railway provides HTTPS)
- Optional: `NEXUS_DEVICE_ID=parent-hub`

4) Deploy and verify:
```bash
curl -sS https://<your-railway-domain>/health
```

If you are serving the UI using the built-in FastAPI templates on the same domain,
you typically do **not** need CORS.

## 3) Child node (edge)

Minimum connectivity test:

```bash
export PARENT_HUB_URL="https://your-api.example.com"
export HIVE_INGEST_TOKEN="..."
export NEXUS_DEVICE_ID="edge-1"
./scripts/run_child.sh
```

This runs a headless whisper import (concept) via `X-Hive-Token`.

### First Whisper (manual curl)

This repo’s `POST /api/hive/whisper/import` validates that `whisper.meta.kind == "wisdom_whisper"`.

```bash
export PARENT_HUB_URL="http://<your-host-ip>:8000"
export HIVE_INGEST_TOKEN="your-shared-secret-123"

curl -sS -X POST "$PARENT_HUB_URL/api/hive/whisper/import" \
  -H "X-Hive-Token: $HIVE_INGEST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "edge-child",
    "version": "whisper_v1",
    "whisper": {
      "meta": {
        "kind": "wisdom_whisper",
        "device_id": "edge-1",
        "created_at": "2026-04-06T00:00:00Z"
      },
      "wisdom": {"recommended_kwargs": {"cycle_learning_rate": 0.03}},
      "note": "hello parent"
    }
  }'
```

If you want this to show up as a connected node in Hive Health, include `meta.device_id`.

If this is the first time the Parent Hub has ever seen that `meta.device_id`, it will also create a small in-app **Nudge** (title: “New child connected”).
The nudge appears in the web UI banner after you log in.

Operator targeting:
- If `HIVE_HEALTH_ALLOWLIST_EMAILS_CSV` is set, the nudge is created only for those accounts.
- Otherwise, the nudge is created for all active users.

### Fetch Latest Wisdom (headless child)

Children can fetch the aggregated global baseline via `GET /api/hive/wisdom/latest` using the same `X-Hive-Token`.

Note: when authenticated via `X-Hive-Token`, the endpoint is restricted to **global wisdom only** (no project-scoped pulls).

```bash
export HIVE_URL="https://<your-railway-domain>"
export HIVE_INGEST_TOKEN="..."

curl -sS "$HIVE_URL/api/hive/wisdom/latest" \
  -H "X-Hive-Token: $HIVE_INGEST_TOKEN" \
  | python -c 'import sys,json; j=json.load(sys.stdin); print(j.get("recommended_kwargs", {}))'
```

### Hive Health (authenticated curl)

```bash
export BASE_URL="http://localhost:8000"
export EMAIL="parent_$(date +%s)@example.com"
export PASS="pass1234"

curl -sS -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\",\"full_name\":\"Parent\"}" \
  "$BASE_URL/api/auth/register" > /dev/null

TOKEN=$(curl -sS -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=$EMAIL" \
  --data-urlencode "password=$PASS" \
  "$BASE_URL/api/auth/login" | python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -sS -H "Authorization: Bearer $TOKEN" \
  "$BASE_URL/api/hive/health?include_regression=false&include_smoothing=false" \
  | python -c 'import sys,json; j=json.load(sys.stdin); print(j["totals"])'
```

For richer observation, use:
- `scripts/passive_telemetry_daemon.py` (posts to `/api/nexus/telemetry/ingest`)

Important: telemetry ingest currently uses user auth (bearer token) rather than `X-Hive-Token`.
The headless `X-Hive-Token` path is intended for Hive import endpoints (wisdom/concepts).

## Notes

- This repo uses `SQLModel.metadata.create_all()` at startup (no Alembic migrations).
- For SaaS: you’ll eventually want real migrations + per-tenant scoping.
