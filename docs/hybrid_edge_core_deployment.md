# Hybrid Edge–Core Deployment (Parent Hub + Child Nodes)

This repo can run as a **Sovereign Root (Parent Hub)** and one or more **Child Nodes**.

- Parent Hub: hosts FastAPI + SQLite and ingests whispers/concepts into `HiveGlobalUpdate`.
- Child Node: runs locally for a user/device and optionally exports privacy-safe payloads.

## Stage 0 — Quick reality check (no migrations)

This project uses SQLModel `create_all()` (see `mycelium_app/db.py`).

- New tables are created automatically.
- Existing tables are **not migrated** (no Alembic in this repo).
- If you change schemas materially, you’ll need to migrate manually or rebuild the DB.

## Stage 1 — Parent Hub setup

### 1) Create venv + install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt
```

### 2) Configure `.env`

```bash
cp .env.example .env
```

Set at minimum:
- `HIVE_ENABLED=true`
- `HIVE_INGEST_TOKEN=...` (shared secret for child → parent ingest)

### 3) Run the Parent Hub

```bash
source .venv/bin/activate
uvicorn mycelium_app.main:app --host 0.0.0.0 --port 8000
```

Then open:
- `http://localhost:8000/hive/health`

## Stage 2 — Secure connectivity (recommended)

Use a private mesh like **Tailscale**.

Conceptually:
- Parent Hub is reachable at something like `http://parent-hub:8000` over Tailnet.
- Children send whispers to `http://parent-hub:8000/api/hive/.../import` with `X-Hive-Token`.

## Stage 3 — Child nodes (headless ingest)

### 1) Child → Parent ingest endpoints

The Parent Hub accepts a shared secret header when `HIVE_INGEST_TOKEN` is set:

- Header: `X-Hive-Token: <token>`

Endpoints:
- `POST /api/hive/whisper/import`
- `POST /api/hive/curiosity/import`
- `POST /api/hive/curiosity/concept/import`
- `POST /api/hive/updates/import`

If the token is not set, these endpoints require normal user auth.

### 2) Smoke test

You can use `curl` from a child to verify connectivity (example: concept import):

```bash
curl -sS -X POST \
  "http://parent-hub:8000/api/hive/curiosity/concept/import" \
  -H "Content-Type: application/json" \
  -H "X-Hive-Token: $HIVE_INGEST_TOKEN" \
  -d '{
    "source": "child_smoketest",
    "version": "concept_v1",
    "concept": {
      "meta": {"kind": "curiosity_concept", "device_id": "edge-1", "created_at": "2026-04-06T00:00:00Z"},
      "tag": "smoketest",
      "verdict": "confirm",
      "note": "Child connected to Parent Hub."
    }
  }'
```

## Ops notes

- Backups: the “identity / soul” is in `storage/mycelium.db` by default.
- Consider a daily encrypted backup of `storage/`.
