# Proofgrid

Proofgrid is a multi-user, project-tree workspace for data/ML work (ETL → EDA → stats tests → feature engineering → modeling → dashboards → deployable predictions).

This repo currently ships an MVP platform:

- User auth (register/login)
- Projects (multi-user ready via roles)
- A tree of nodes inside each project (foundation for ETL/EDA/ML steps)

## Quickstart (Linux)

1) Activate the project virtualenv (created earlier):

```bash
cd /home/chizoalban2003/Mycelium
source .venv/bin/activate
```

2) Install dependencies:

```bash
python -m pip install -r requirements/base.txt
```

Notes:
- `requirements/base.txt` includes `scipy` + `feature-engine` to support optional outlier capping backends (`cleaning_outlier_strategy=feature_engine`).
- Optional dependencies live in `requirements/`:
	- `requirements/benchmarks.txt` is only needed for the `scripts/benchmark_*.py` scripts.
	- `requirements/optional-analytics.txt` duplicates the pinned `scipy` + `feature-engine` installs (handy if you want an explicit “analytics extras” install).

3) Run the app:

```bash
uvicorn mycelium_app.main:app --reload --port 8000
```

Open:

- Web UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

## Create your first user

Option A (script):

```bash
python scripts/create_user.py --email you@example.com --password "change-me" --full-name "Your Name"
```

Option B (API): `POST /api/auth/register`

## Dev notes

- SQLite DB is stored in `storage/mycelium.db` (created automatically).
- `storage/` is ignored by git.

## Passive Telemetry (Silent-24)

This repo includes a best-effort Linux PassiveTelemetry daemon that posts `app_open` signals into the Nexus telemetry ledger.

Requirements:
- X11 session (needs `DISPLAY`)
- `xprop` installed (Debian/Ubuntu: `sudo apt-get install x11-utils`)

Run (logs in to get a token, then starts posting app changes):

```bash
python scripts/passive_telemetry_daemon.py \
	--base-url http://127.0.0.1:8000 \
	--email you@example.com \
	--password "change-me" \
	--device-id local \
	--poll-seconds 2
```

Dry-run (prints events, does not POST):

```bash
python scripts/passive_telemetry_daemon.py --dry-run
```

Note: Wayland support is not implemented yet.

Run the Deep Freeze sweep + print growth status:

```bash
python3 scripts/silent24_deep_freeze.py \
	--base-url http://127.0.0.1:8000 \
	--email you@example.com \
	--password "change-me" \
	--device-id local
```

## Consented device actions (confidence-gated)

Mycelium can queue local device operations only after explicit user permission.

1) Enable actions in policy (and keep confirmations on):

- `POST /api/nexus/policy` with:
	- `actions.enabled=true`
	- `actions.device_control_enabled=true`
	- `actions.require_confirm=true`
	- `actions.notify_only=false`
	- `actions.min_confidence=0.90`
	- `actions.allowed_capabilities=["start_focus_session"]`

2) Generate a telemetry proposal:

- `POST /api/nexus/telemetry/assistant/tick`

3) User approves a proposed action:

- `POST /api/nexus/telemetry/assistant/action`

4) Companion agent polls pending actions:

- `GET /api/nexus/telemetry/device-actions/pending?device_id=local`

5) Companion agent reports result:

- `POST /api/nexus/telemetry/device-actions/{message_id}/ack`

This keeps operations auditable and reversible while improving learning from accept/reject/execute outcomes.

## Telegram nudge bridge (optional)

For external messaging (without opening the app), enable:

- `NOTIFICATIONS_BRIDGE_ENABLED=true`
- `NOTIFICATIONS_TELEGRAM_BOT_TOKEN=<bot-token>`
- `APP_PUBLIC_BASE_URL=https://<your-domain>`

Then opt in per user via `POST /api/nexus/policy`:

- `notifications.enabled=true`
- `notifications.telegram_enabled=true`
- `notifications.telegram_chat_id=<telegram-chat-id>`
- Optional `notifications.telegram_nudge_kinds=["telemetry_assistant","wisdom_update"]`

## ActionSchema (task reproduction foundation)

Behavioral mirroring now has a first API skeleton:

- Record observed trajectories: `POST /api/nexus/tasks/trajectory/record`
- Propose executable replicas: `POST /api/nexus/tasks/replicas/propose`
- Approve/reject: `POST /api/nexus/tasks/replicas/{id}/decision`
- Ack local execution result: `POST /api/nexus/tasks/replicas/{id}/ack`

Approvals queue into the existing device action outbox for companion-agent execution.

SelfReflection (analyze best sweeps):

- `GET /api/nexus/reflection?window_days=30&top_limit=5`

Homeostasis (stability + pruning + identity backup):

- Enable background ticking with env var: `NEXUS_HOMEOSTASIS_ENABLED=true` (maps to settings `nexus_homeostasis_enabled`).
- Manual tick: `POST /api/nexus/homeostasis/tick`
- Status: `GET /api/nexus/homeostasis/status`

When homeostasis is enabled, `/api/predict/electrophoresis` will consult the latest homeostasis mood and may apply a small, allowlisted learning-rate tightening when mood is `agitated` (reported back in the response under `homeostasis`).

Rebuild local virtualenv:

```bash
bash scripts/rebuild_env.sh
```

Benchmarks / scratch data:
- `tmp_eval/` is ignored by git (local outputs + datasets). This repo no longer ships the benchmark dataset.
- Use your own CSV when running benchmark scripts, e.g. `python scripts/benchmark_salary_models.py --csv /path/to/data.csv --target salary`.
- Or generate a synthetic sample dataset: `python scripts/sample_salary_dataset.py --out tmp_eval/sample_salary_dataset.csv`, then pass it into scripts.
