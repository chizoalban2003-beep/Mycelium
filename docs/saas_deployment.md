# SaaS Deployment (Cloud Parent Hub + Edge Children)

This repo already runs as a local “Parent Hub” + “Child” model.
This doc makes it deployable as a SaaS platform.

## Fast path: your devices are the “main brain” (self-hosted Parent Hub)

If you want *your* machine/home-server to be the primary brain, run the **Parent Hub** on one device and point your other devices at it as **Children**.

Recommended networking:
- Same Wi‑Fi/LAN for quick testing, or
- **Tailscale** for secure “works anywhere” connectivity without exposing port 8000 to the public internet.

### A) Brain device: run the Parent Hub with Docker (recommended)

From the repo root on the brain device:

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:
- `SECRET_KEY` (long random)
- `HIVE_INGEST_TOKEN` (shared secret children will use)
- Optional: `COOKIE_SECURE=true` if you’re behind HTTPS

Then run:

```bash
docker compose up --build -d
```

Verify:

```bash
curl -sS http://127.0.0.1:8000/health
```

### B) Brain device: run the Parent Hub without Docker (SQLite dev mode)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt
cp .env.example .env
uvicorn mycelium_app.main:app --host 0.0.0.0 --port 8000
```

This stores state in `storage/mycelium.db`.

### C) Other devices: connect as Children (smoketest)

On each child device (can be your laptop/phone-on-Termux/another server), in a clone of the repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt

export PARENT_HUB_URL="http://<BRAIN_IP_OR_TAILSCALE_NAME>:8000"
export HIVE_INGEST_TOKEN="<same-as-brain-.env>"
export NEXUS_DEVICE_ID="edge-1"  # unique per device

./scripts/run_child.sh
```

If connectivity is correct, the child will POST a concept into the brain via:
- `POST /api/hive/curiosity/concept/import` with `X-Hive-Token`

To see child nodes in the UI, log into the brain and open:
- `/hive/health`

### D) Phone experience: make it feel like a personal assistant

This repo already has a **Nudges** system (the assistant “voice”) and a **Telemetry** ledger for digital signals.

What you get out of the box:
- A nudge banner inside the web UI.
- Optional telemetry-derived assistant nudges (server-side background loop).
- Optional device notifications:
  - **In-app notifications** when the web app is open.
  - **Android notifications** via Termux poller (works even when the web UI is closed).

#### Install on Android (PWA)

1) Open the Parent Hub URL in Chrome on your phone (LAN/Tailscale):
- `http://<brain-ip>:8000` or `http://<tailscale-name>:8000`

2) Chrome menu → **Add to Home screen** (or **Install app**).

3) Launch it from the home screen. It opens as a standalone app.

#### True downloadable Android app (APK/AAB)

If you want a real installable Android package (not just PWA install), use Trusted Web Activity packaging:

- See [docs/android_twa_packaging.md](docs/android_twa_packaging.md)
- This produces APK/AAB while keeping your existing Railway-hosted web app

#### Enable telemetry assistant nudges (server-side)

On the brain device, set in `.env`:

```bash
NEXUS_TELEMETRY_ASSISTANT_ENABLED=true
NEXUS_TELEMETRY_ASSISTANT_TICK_SECONDS=60
NEXUS_TELEMETRY_ASSISTANT_CONFIDENCE_THRESHOLD=0.85
```

Then restart the Parent Hub.

Per-user permission is controlled by parental policy. To allow *action proposals* (still confirmation-based), set:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/nexus/policy \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"policy":{"actions":{"enabled":true,"notify_only":true,"require_confirm":true}}}'
```

#### Android notifications via Termux (optional)

If you want real device notifications even when the UI is closed:

1) Install Termux + Termux:API.

2) In Termux:

```bash
pkg update
pkg install python
pip install -r requirements/base.txt
```

3) Run the poller:

```bash
python3 scripts/poll_nudges_and_notify.py \
  --base-url http://<brain-ip-or-tailscale>:8000 \
  --email you@example.com \
  --password '...' \
  --kinds telemetry_assistant,wisdom_update,child_connected \
  --min-confidence 0.85 \
  --auto-ack
```

This will call `termux-notification` if available.

#### Keep only the newest assistant nudge (optional)

If your banner is crowded while testing, keep only the newest unseen nudge:

```bash
python3 scripts/ack_old_nudges.py --email you@example.com --keep 1
```

#### Export your digital signals to CSV

On the brain device:

```bash
python3 scripts/telemetry_export_csv.py --since-hours 168 --out storage/telemetry_export.csv
```

To export only your account’s signals:

```bash
python3 scripts/telemetry_export_csv.py --email you@example.com --since-hours 168 --out storage/telemetry_you.csv
```

#### Phone child ↔ laptop brain sharing loop

For your setup (laptop = main brain, phone = child):

1) Laptop brain runs Parent Hub (Tailscale/LAN reachable).
2) Phone installs PWA and logs in for nudges + approve/reject actions.
3) Optional on phone (Termux): run notification poller to receive nudges even when browser is closed.
4) Optional on phone/other devices: run child ingest (`run_child.sh`) with same `HIVE_INGEST_TOKEN` and unique `NEXUS_DEVICE_ID`.

This gives two-way collaboration:
- child devices whisper updates to the laptop brain
- laptop brain broadcasts insights/nudges back to users

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

#### Option A: keep Railway as staging/demo (brain runs on your device)

If you are switching the “main brain” to a self-hosted Parent Hub (home server / laptop),
keep Railway running but prevent your children from accidentally ingesting into Railway:

- Simplest: set `HIVE_ENABLED=false` on Railway.
- Or: keep `HIVE_ENABLED=true` but set a **different** `HIVE_INGEST_TOKEN` on Railway than the one used by your self-hosted brain.
- Optional: set `NEXUS_DEVICE_ID=railway-staging` so it’s visually distinct in logs/telemetry.

Then, on child devices, ensure `PARENT_HUB_URL` points to your self-hosted brain (LAN IP or Tailscale name), not the Railway URL.

#### Option B: Railway is the brain (cloud Parent Hub)

Use this if you want “works anywhere” without running a home server.

- Keep `HIVE_ENABLED=true` and set `HIVE_INGEST_TOKEN` on Railway.
- Children use `PARENT_HUB_URL=https://<your-railway-domain>`.
- For phone-as-assistant, install the PWA from that Railway URL.

#### Option C: Hybrid (device brain + cloud mirror)

Use this if you want local-first privacy (device brain), but also want a cloud copy for resilience/demo.

- Keep two Parent Hubs (local + Railway).
- Use **different** `HIVE_INGEST_TOKEN`s so children don’t cross-stream accidentally.
- Optionally run exports from local to cloud using your own sync rules (this repo has Hive outbox primitives, but a full two-way mirror is intentionally not automatic).

## 2.5) Bring other users into the Hive

### Quick invite (API)

Use the helper script:

```bash
export BASE_URL="http://<brain-ip-or-tailscale>:8000"
export OWNER_TOKEN="<owner-access-token>"
export INVITE_EMAIL="friend@example.com"
export INVITE_PASS="temporary-password"
export INVITE_NAME="Friend"
export PROJECT_ID="1"   # optional
export ROLE="viewer"    # owner|editor|viewer

bash scripts/hive_invite_user.sh
```

What this does:
- creates the invited account (if it doesn’t already exist)
- optionally adds them to a project (`POST /api/projects/{id}/members`)

### Local-only invite (SQLite brain)

If you prefer local shell creation:

```bash
python3 scripts/create_user.py --email friend@example.com --password 'temporary-password' --full-name 'Friend'
```

Then they can sign in to your laptop brain URL over Tailscale.

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
