# SaaS Deployment (Railway Public Hub + Android Distribution)

## Public Alpha fast path (recommended)

This is the current default strategy for market velocity:

- Backend: Railway (public HTTPS domain)
- App distribution: Android TWA/PWA wrapper (Google Play)
- Security membrane: strict CORS + secure cookies + ingest throttling

### Required Railway variables

- `DATABASE_URL` (Railway Postgres)
- `SECRET_KEY` (long random)
- `COOKIE_SECURE=true`
- `HIVE_ENABLED=true`
- `HIVE_INGEST_TOKEN=<strong-random>`
- `CORS_ALLOW_ORIGINS_CSV=https://<your-domain>`
- `HIVE_WISDOM_MIN_WHISPERS=2`
- `HIVE_WISDOM_MIN_DEVICES=3`

### Closed-loop directive quick check (manual)

1) Bootstrap directive:

```bash
curl -X POST "https://<domain>/api/nexus/tasks/bootstrap/work-session" \
  -H "Authorization: Bearer <TOKEN>"
```

2) Approve replica:

```bash
curl -X POST "https://<domain>/api/nexus/tasks/replicas/<REPLICA_ID>/decision" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"decision":"approve"}'
```

3) Verify outcome:

```bash
python3 scripts/report_task_replica_focus.py \
  --base-url "https://<domain>" \
  --token "<TOKEN>" \
  --replica-id <REPLICA_ID> \
  --planned-minutes 45 \
  --focused-minutes 45 \
  --completed
```

### Android/TWA verification

Host Digital Asset Links on your production domain:

- `/.well-known/assetlinks.json`

This repository now serves that path directly from env vars:

- `ANDROID_APP_PACKAGE_NAME`
- `ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV`

Set those in Railway before publishing your Play build.

### Public-ingest protection

`/api/hive/whisper/import` now has a basic rate limiter (windowed source/device
caps) for public internet exposure. Tune with:

- `HIVE_WHISPER_IMPORT_RATE_LIMIT_ENABLED`
- `HIVE_WHISPER_IMPORT_RATE_LIMIT_WINDOW_SECONDS`
- `HIVE_WHISPER_IMPORT_RATE_LIMIT_MAX_PER_SOURCE`
- `HIVE_WHISPER_IMPORT_RATE_LIMIT_MAX_PER_DEVICE`

### Optional: Telegram "Synapse Bridge"

If you want nudges to arrive outside the app, enable Telegram dispatch:

- `NOTIFICATIONS_BRIDGE_ENABLED=true`
- `NOTIFICATIONS_TELEGRAM_BOT_TOKEN=<bot-token>`
- `NOTIFICATIONS_TELEGRAM_WEBHOOK_SECRET=<secret-token>`
- `APP_PUBLIC_BASE_URL=https://<your-domain>`

Per-user opt-in lives in `POST /api/nexus/policy` under `notifications`:

- `notifications.enabled`
- `notifications.telegram_enabled`
- `notifications.telegram_chat_id`
- `notifications.telegram_nudge_kinds`

Inbound Telegram chat can be wired to:

- `POST /api/nexus/chat/telegram/webhook`
- header: `X-Telegram-Bot-Api-Secret-Token: <secret-token>`

When `telegram_chat_id` matches a user policy, inbound messages are persisted to chat history and answered.
Status prompts (e.g., "how are you") now include live viscosity-grounded response context.

Emergency brake commands (Telegram, implemented):

- `/freeze` → enables `actions.kill_switch=true` and clears pending device actions
- `/unfreeze` → disables `actions.kill_switch`
- `/freeze status` (or `/killswitch`) → returns current governance safety state

### Hybrid predictor API (physics governor + timing model)

Use this endpoint to ask whether a focus work session should be recommended now:

- `POST /api/nexus/hybrid/work-session/next`

Adaptive directive endpoint (duration adapts to viscosity):

- `POST /api/nexus/hybrid/directive/work-session/adaptive`

Multi-node coordination endpoint (device handoff):

- `POST /api/nexus/hybrid/directive/work-session/multinode`
- Ranks candidate devices by live viscosity and recommends when to move from phone to laptop/desktop.

Auto-handoff launch endpoint (one-step):

- `POST /api/nexus/hybrid/directive/work-session/auto-handoff-launch`
- Runs multi-node analysis and proposes a focus-session replica on the best node.
- Safety: if all nodes are gated, returns recovery mode with `suggested_duration_minutes=0`.

One-tap confirmation endpoint:

- `POST /api/nexus/hybrid/directive/work-session/auto-handoff-confirm`
- Queues execution with idempotent behavior (duplicate confirmation reuses pending device action).

Deterministic handoff session endpoints:

- `POST /api/nexus/hybrid/handoff/session/start`
- `POST /api/nexus/hybrid/handoff/session/{session_id}/tick`
- Enables retry/timeout orchestration with explicit state transitions.

Chat UX:

- "launch now" in chat/Telegram triggers analyze → propose → confirm flow.

Autonomy policy modes (`POST /api/nexus/policy`, field `actions.autonomy_mode`):

- `strict`: always manual confirm
- `balanced`: auto-confirm only for low-viscosity non-handoff launches
- `auto`: auto-confirm when governor + viscosity gates pass

`actions.require_confirm=true` always forces manual confirmation regardless of mode.

### Release validation smoke flow (recommended before deploy)

Run the E2E autonomy path in one command:

- `python3 scripts/smoke_autonomy_handoff_flow.py --base-url <url> --token <token> --completed --planned-minutes 45 --focused-minutes 45`

This validates:

1. `auto-handoff-launch`
2. `auto-handoff-confirm` (if not already auto-approved)
3. `replicas/{id}/ack`
4. `replicas/{id}/verify`

`/verify` now supports optional structured feedback labels to improve policy learning quality:

- `helpful`
- `timely`
- `annoying`
- `wrong_device`
- `too_early`
- `too_late`
- `too_long`
- `too_short`
- `interruptive`

Feedback analytics endpoint:

- `GET /api/nexus/tasks/replicas/feedback/summary?window_hours=168`

Config flags:

- `HYBRID_PREDICTOR_ENABLED`
- `HYBRID_PREDICTOR_WINDOW_MINUTES`
- `HYBRID_PREDICTOR_MIN_SIGNAL_EVENTS`
- `HYBRID_PREDICTOR_GOVERNOR_MIN_CONFIDENCE`

### v7 ActionSchema (behavioral mirroring foundation)

New API surface for "observe → propose → approve → execute":

- `POST /api/nexus/tasks/bootstrap/work-session` (one-click starter directive)
- `POST /api/nexus/tasks/trajectory/record`
- `POST /api/nexus/tasks/replicas/propose`
- `GET /api/nexus/tasks/replicas/recent`
- `POST /api/nexus/tasks/replicas/{replica_id}/decision`
- `POST /api/nexus/tasks/replicas/{replica_id}/ack`
- `POST /api/nexus/tasks/replicas/{replica_id}/verify` (reports real focus outcome)

Approved replicas are queued into the existing device action outbox for local
companion execution.

Verification updates the directive confidence and stores evidence in GrowthLedger
(`domain=task_replica_focus`, `metric=adherence`).

### Assistant naming ceremony (identity sovereignty)

You can personalize assistant identity (name/gender/voice) globally per user.

API:

- `GET /api/nexus/identity/assistant/profile`
- `POST /api/nexus/identity/assistant/profile`
- `GET /api/nexus/assistant/configure` (compat route)
- `POST /api/nexus/assistant/configure` (compat route)

Web UI:

- `/assistant/profile`

Fields:

- `given_name` (e.g. "Jarvis")
- `gender_identity` (`neutral|female|male|nonbinary|custom`)
- `vocal_preset` (provider-specific voice key)
- `assistant_avatar_url` (optional `http(s)` image URL)

Identity presentation (`/api/nexus/identity/presentation`) now uses this profile
for personalized assistant display naming.

Project-scoped identity updates are owner-gated by ProjectMembrane.

### Real-time Live Neural Map

Main-brain observability endpoint:

- `GET /api/nexus/live/state?window_minutes=30`

Web UI:

- `/hive/health` now includes an animated flow map of signal → growth → assistant → nudge movement.
- `/hive/health` also includes a live viscosity gauge (`flow|observe|gated`).

### Omnichannel chat (app + messaging)

Unified chat API:

- `POST /api/nexus/chat/send`
- `GET /api/nexus/chat/history`

Channels (current):

- `app` (in-product chat)
- `telegram` (if bridge + policy are enabled)

Web UI:

- `/chat`

Memory consolidation trigger:

- Send "daily summary" through app chat or Telegram.
- Backed by: `GET /api/nexus/reflection/daily-summary?window_hours=24`

### Child trajectory capture: manual first, auto optional

Recommended rollout:

1) Manual-quality phase: record trajectories intentionally for a few sessions.
2) Optional auto phase: let child passively infer trajectories from app-open patterns.

`run_child.sh` now supports opt-in auto capture:

- `CHILD_AUTO_CAPTURE_TRAJECTORIES=true`
- `CHILD_BEARER_TOKEN=<access-token>` or `CHILD_EMAIL` + `CHILD_PASSWORD`
- Optional tuning:
  - `CHILD_TRAJECTORY_WINDOW_SIZE=3`
  - `CHILD_TRAJECTORY_COOLDOWN_SECONDS=600`
  - `CHILD_TRAJECTORY_MUST_INCLUDE_CSV=mycelium`

### Launch readiness checklist (Railway + Play)

Use this before opening public alpha:

- [ ] Rotate `SECRET_KEY` in Railway and redeploy.
- [ ] Confirm `COOKIE_SECURE=true` and strict `CORS_ALLOW_ORIGINS_CSV`.
- [ ] Set Android identity vars:
  - [ ] `ANDROID_APP_PACKAGE_NAME`
  - [ ] `ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV`
- [ ] Verify `https://<domain>/.well-known/assetlinks.json` returns expected values.
- [ ] Verify public ingest throttling is enabled:
  - [ ] `HIVE_WHISPER_IMPORT_RATE_LIMIT_ENABLED=true`
  - [ ] Window/source/device limits are set.
- [ ] Run one closed-loop directive test:
  1. `POST /api/nexus/tasks/bootstrap/work-session`
  2. approve a proposed replica (`/decision`)
  3. execute locally via child companion
  4. report outcomes via `/verify`
- [ ] Confirm GrowthLedger records adherence:
  - `domain=task_replica_focus`
  - `metric=adherence`
- [ ] Keep device capabilities allowlisted (minimum viable set) in user policy.

### Live demonstration runbook (phone → laptop handoff)

Use this to demonstrate multi-node intelligence before opening Alpha:

1. Policy set for demo user:
  - `actions.enabled=true`
  - `actions.device_control_enabled=true`
  - `actions.notify_only=false`
  - `actions.require_confirm=false`
  - `actions.autonomy_mode=auto` (or `balanced` for semi-auto demo)
  - `actions.min_confidence=0.0` for smoke/demo accounts (use higher values in production)
2. Ensure both devices are visible in `/hive/health` with distinct `device_id`s.
3. On phone (or Telegram), send: **"launch now"**.
4. Confirm launch response includes:
  - `launch_mode=approved` (or `pending_confirm` in balanced flow)
  - `recommended_device_id=laptop` (or desktop candidate)
  - `replica_id > 0`
5. If pending, call `POST /api/nexus/hybrid/directive/work-session/auto-handoff-confirm`.
6. Execute on laptop child companion, then call:
  - `POST /api/nexus/tasks/replicas/{id}/ack`
  - `POST /api/nexus/tasks/replicas/{id}/verify`
7. Validate ledger evidence:
  - GrowthLedger `domain=task_replica_focus`
  - `metric=adherence`

One-command rehearsal:

```bash
python3 scripts/smoke_autonomy_handoff_flow.py \
  --base-url "https://<domain>" \
  --token "<TOKEN>" \
  --current-device-id phone \
  --candidate-device-ids phone,laptop,desktop \
  --completed \
  --planned-minutes 45 \
  --focused-minutes 45
```

### 10-user Alpha rollout sequence (recommended)

Stage 0 — Gate check (same day):
- Run the smoke flow successfully.
- Verify `/.well-known/assetlinks.json` and TWA package fingerprints.
- Confirm strict CORS + secure cookies + ingest rate limiter.

Stage 1 — Cohort A (users 1-3, 24 hours):
- Policy default: `autonomy_mode=balanced`.
- Require feedback on every proposed/auto-confirmed directive.
- Monitor: launch success rate, recovery-mode frequency, verify adherence.

Stage 2 — Cohort B (users 4-7, next 24-48 hours):
- Keep `balanced` default.
- Allow `auto` only for opt-in testers with low-risk profiles.
- Track Telegram latency, nudge delivery, and false-positive handoffs.

Stage 3 — Cohort C (users 8-10):
- Expand to mixed device topologies (phone+laptop, phone+desktop).
- Keep project-scoped permissions strict (`viewer` cannot execute actions).

Promotion rule to public Alpha:
- 0 critical auth/cross-project incidents.
- Smoke test pass rate 100% for last 5 runs.
- Median adherence non-decreasing across cohorts.
- Auto-handoff recovery rate within acceptable threshold for your product target.

### Jarvis-grade roadmap (Grow with Data)

To evolve from operational assistant to durable cognitive system, execute in phases:

Phase 1 (now): learning signal quality
- Capture structured post-action feedback labels via `/verify`.
- Monitor `feedback/summary` trends (`annoying`, `wrong_device`, `too_early`, etc.).

Phase 2: adaptive memory
- Separate episodic, semantic, and procedural memory lanes.
- Add reinforcement/decay so stable routines strengthen while stale behaviors fade.

Phase 2 API (implemented):

- `POST /api/nexus/memory/upsert`
- `GET /api/nexus/memory/list`
- `POST /api/nexus/memory/{memory_id}/reinforce`
- `POST /api/nexus/memory/decay/run`

Automatic sync (implemented):

- Successful/failed `POST /api/nexus/tasks/replicas/{replica_id}/verify` now auto-upserts and reinforces memory lanes:
  - `episodic` (`replica:{id}`)
  - `semantic` (`focus_pattern:{device}:planned_{minutes}`)
  - `procedural` (`procedure:{capability}:work_session`)

Lane values:

- `episodic`
- `semantic`
- `procedural`

Phase 3: trust and governance
- Per-action permission tiers (`suggest`, `queue`, `execute`).
- Explainability surface for every autonomous action (“why now, why this device”).
- Operator kill-switch and audit replay as standard controls.

Phase 3 API (implemented):

- `GET /api/nexus/tasks/replicas/{replica_id}/explain`
- `POST /api/nexus/tasks/actions/kill-switch`
- `POST /api/nexus/tasks/actions/{message_id}/replay`

Phase 3 policy keys (implemented under `actions`):

- `default_permission_tier`: `suggest|queue|execute`
- `permission_tiers`: per-capability tier map (for example `{"start_focus_session":"queue"}`)
- `kill_switch`: hard-stop for queuing/replaying device actions

Phase 4: reliability and evaluation
- Deterministic handoff state machine with retries and timeout recovery.
- Scenario-based autonomy eval suite (`strict|balanced|auto`) and regression gates.

Phase 4 API (implemented):

- `POST /api/nexus/hybrid/handoff/session/start`
- `POST /api/nexus/hybrid/handoff/session/{session_id}/tick`
- `GET /api/nexus/tasks/actions/audit/timeline`

Phase 4 eval script (implemented):

- `python3 scripts/eval_autonomy_modes.py --base-url <url> --token <token>`

Global audit report (implemented):

- `python3 scripts/global_audit_report.py --base-url <url> --token <token>`
- Runs governance timeline analysis + mode probes (`strict|balanced|auto`) and restores original policy after evaluation.
- Optional Telegram executive summary:
  - `--telegram-bot-token <token> --telegram-chat-id <chat_id>`
  - or env: `NOTIFICATIONS_TELEGRAM_BOT_TOKEN` + `AUDIT_TELEGRAM_CHAT_ID`

Deterministic handoff states:

- `proposed`
- `queued`
- `waiting_retry`
- `completed`
- `failed`
- `timed_out`
- `recovery`

Phase 5: identity depth
- Persona mode profiles (coach, calm, briefing) with context-aware tone routing.
- Keep identity presentation consistent across web + Telegram + device actions.

---

This repo already runs as a local “Parent Hub” + “Child” model.
This doc makes it deployable as a SaaS platform.

## Advanced path: your devices are the “main brain” (self-hosted Parent Hub)

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

#### One-shot assistant bootstrap (register/login/tick/nudges)

```bash
export BASE_URL="http://<brain-ip-or-tailscale>:8000"
export EMAIL="you@example.com"
export PASS="your-password"
./scripts/run_assistant_bootstrap.sh
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

## 2.6) ProjectMembrane (project isolation)

The current repo now enforces a project membrane for key collaboration paths:

- Project-scoped Hive imports (`meta.project_id`) require authenticated user membership in that project.
- Headless `X-Hive-Token` imports are restricted to global scope (no project-scoped ingest).
- `GET /api/hive/wisdom/latest?project_id=...` requires project membership when authenticated.
- Telemetry assistant action execution in project scope requires `owner` or `editor` role (`viewer` cannot execute actions).

Practical effect:
- Viewers can observe metrics/nudges.
- Owners/editors can approve project-scoped telemetry actions.
- Cross-project whispers are rejected when caller is not a member.

### Global Wisdom filter (consensus gate)

Before broadcasting `recommended_kwargs`, the Hive now applies evidence gating:

- minimum whispers: `HIVE_WISDOM_MIN_WHISPERS`
- minimum distinct devices: `HIVE_WISDOM_MIN_DEVICES`
- key-level consensus threshold: `HIVE_WISDOM_CONSENSUS_FRACTION`

Example production defaults:

```bash
HIVE_WISDOM_MIN_WHISPERS=3
HIVE_WISDOM_MIN_DEVICES=2
HIVE_WISDOM_CONSENSUS_FRACTION=0.6
```

If evidence is below threshold, `recommended_kwargs` is intentionally empty and `evidence.wisdom_filter.gated=true`.

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
