# Mycelium

Mycelium is a multi-user, project-tree workspace for data/ML work (ETL → EDA → stats tests → feature engineering → modeling → dashboards → deployable predictions).

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

## Consent-based device telemetry

This repo includes an opt-in Linux telemetry daemon that posts `app_open` signals into the Nexus telemetry ledger.
It is designed for transparent, user-controlled learning from the devices you own and use.

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
- `NOTIFICATIONS_TELEGRAM_WEBHOOK_SECRET=<secret-token>`
- `APP_PUBLIC_BASE_URL=https://<your-domain>`

Then opt in per user via `POST /api/nexus/policy`:

- `notifications.enabled=true`
- `notifications.telegram_enabled=true`
- `notifications.telegram_chat_id=<telegram-chat-id>`
- Optional `notifications.telegram_nudge_kinds=["telemetry_assistant","wisdom_update"]`

Telegram inbound webhook endpoint:

- `POST /api/nexus/chat/telegram/webhook`
- Add request header `X-Telegram-Bot-Api-Secret-Token: <secret-token>` when webhook secret is configured.
- Telegram status prompts like "how are you" now return a viscosity-grounded state summary.

## Email recovery channel (optional)

Mycelium can also send password-reset links by email when SMTP is configured.

Set these environment variables:

- `MAIL_ENABLED=true`
- `MAIL_FROM_ADDRESS=noreply@your-domain.com`
- `MAIL_SMTP_HOST=smtp.your-provider.com`
- `MAIL_SMTP_PORT=587`
- `MAIL_SMTP_USERNAME=<username>`
- `MAIL_SMTP_PASSWORD=<password>`
- `MAIL_SMTP_USE_TLS=true`
- `MAIL_SMTP_USE_SSL=false`

Account recovery then uses the login page’s recovery form and sends a one-time reset link to the user’s email address, with Telegram as an optional extra channel if enabled.

## ActionSchema (task reproduction foundation)

Behavioral mirroring now has a first API skeleton:

- Record observed trajectories: `POST /api/nexus/tasks/trajectory/record`
- Propose executable replicas: `POST /api/nexus/tasks/replicas/propose`
- Approve/reject: `POST /api/nexus/tasks/replicas/{id}/decision`
- Ack local execution result: `POST /api/nexus/tasks/replicas/{id}/ack`

Approvals queue into the existing device action outbox for companion-agent execution.

Quick start (recommended first directive):

- `POST /api/nexus/tasks/bootstrap/work-session`

This seeds a trajectory + a proposed replica for:
"open dashboard + open focus app + enable DND + start 45m focus timer".

Optional: auto-capture trajectories from child app-open transitions:

- `CHILD_AUTO_CAPTURE_TRAJECTORIES=true`
- `CHILD_BEARER_TOKEN=<access-token>` (or `CHILD_EMAIL` + `CHILD_PASSWORD`)
- Run `./scripts/run_child.sh`

Success verification loop (recommended):

- `POST /api/nexus/tasks/replicas/{id}/verify`

Helper script:

- `python3 scripts/report_task_replica_focus.py --base-url <url> --token <token> --replica-id <id> --planned-minutes 45 --focused-minutes 45 --completed`

End-to-end autonomy smoke script (launch → confirm → ack → verify):

- `python3 scripts/smoke_autonomy_handoff_flow.py --base-url <url> --token <token> --completed --planned-minutes 45 --focused-minutes 45`

This updates directive confidence and records a GrowthLedger outcome for learning.

## Assistant profile (name/gender/voice)

Personalize assistant identity:

- Web UI: `/assistant/profile`
- API:
	- `GET /api/nexus/identity/assistant/profile`
	- `POST /api/nexus/identity/assistant/profile`
	- `POST /api/nexus/assistant/configure` (compat alias)

This updates identity presentation so the assistant is addressed by the configured name.
Telemetry assistant nudges also include the configured assistant name/voice traits.

## Learning contract

Mycelium learns from consented digital signals, including app launches, open/close activity, approved actions, chat messages, and explicit feedback.
It is designed to grow from infancy to adulthood as a transparent assistant that stays user-controlled, visible, and reversible.

## Human↔AI relationship

Mycelium is meant to feel like a sci-fi co-pilot that earns trust over time.
It should feel alive in tone, but never hidden in behavior.

Suggestions that keep the relationship healthy:

- Make every signal visible and explainable.
- Keep permissions explicit, granular, and easy to revoke.
- Increase autonomy only after repeated successful interactions.
- Let the assistant speak in a consistent voice, but keep the user in charge.
- Show a short reason trail for each recommendation: what it learned, why it matters, and which permission allowed it.
- Keep recent signal history and learning summaries easy to inspect from the app.
- Put the revoke path next to each opt-in surface, especially recovery, telemetry, and device actions.
You can also set `assistant_avatar_url` (optional `http(s)` image URL).

## Device communication surfaces

Because Mycelium can live on the user’s own device, it should use any communication surface the device/browser exposes and the user has approved.

Current supported surfaces:

- In-app chat in the web UI
- Telegram bridge for off-app conversations
- Browser notifications for live nudges
- Speech synthesis for spoken replies in the browser
- Clipboard copy for quick handoff to other apps
- Native share sheet when the browser supports it
- Termux notifications on Android for background alerts

## Universal stimulus pipeline

Mycelium now has a shared path for digital stimulus: app events, chat payloads, UI actions, telemetry, file-derived metadata, and other structured signals can all be normalized into the same ledger flow.

Use the universal ingest endpoint when the source is not a plain text note:

- `POST /api/nexus/stimulus/ingest`

It stores:

- a safe surface copy of the stimulus
- a flattened tabular feature map
- structural counts like depth, list length, and scalar mix
- a stable digest for dedupe and replay
- a learning profile that recommends the right metrics and model head

Visible signal sources now include:

- device boot and daemon startup
- app opens and navigation/focus changes
- chat messages, recovery requests, and assistant replies
- project creation and workspace setup
- member invites, role changes, and node creation
- task bootstrap, approval, ack, and verification loops
- curiosity answers, dismissals, and export feedback
- nudge acknowledgments and seen-state updates
- predictor uploads, corrections, and measured outcomes

Suggested improvement loop:

1. Emit every meaningful UI interaction as a stimulus event.
2. Normalize it into a compact tabular row.
3. Feed those rows into `SignalLedgerEvent`, `GrowthLedgerEntry`, and the predictor.
4. Promote only stable, consented patterns into memory or Hive export.

Self-evaluation rule:

- Use classification metrics like `accuracy`, `f1_macro`, and `balanced_accuracy` for discrete outcomes.
- Use regression metrics like `mae`, `rmse`, and `r2` for numeric outcomes.
- Keep the physics model as the feature/pattern engine, and let the evaluation layer choose the scoreboard.

That keeps the app aligned with the motto: Grow with Data.

Design rule:

- Use every available channel, but only after explicit consent and policy enablement.
- Prefer visible, reversible communication over silent background behavior.
- Treat the local device as the main brain when the user chooses that mode; the Hive should amplify it, not replace it.

## Live neural map + chat

- Live state API: `GET /api/nexus/live/state?window_minutes=30`
- Visual map UI: `/hive/health`

Live state now includes a `viscosity` barometer payload (`score`, `band`, `prediction_state`, and component factors) used by the Hive Health Viscosity Gauge.

Chat APIs:

- `POST /api/nexus/chat/send`
- `GET /api/nexus/chat/history`

Chat UI:

- `/chat`

`channel=telegram` is supported when the Telegram bridge and user notification policy are enabled.
The chat page also exposes device-side actions such as speech playback, share, and copy for the latest assistant reply.

## Hybrid predictor (physics governor + neural timing)

- API: `POST /api/nexus/hybrid/work-session/next`
- Request body: `{"project_id": null, "window_minutes": 120}`
- Output includes `timing_score`, `governor_confidence`, `governor_ok`, and recommendation.

Runtime flags:

- `HYBRID_PREDICTOR_ENABLED`
- `HYBRID_PREDICTOR_WINDOW_MINUTES`
- `HYBRID_PREDICTOR_MIN_SIGNAL_EVENTS`
- `HYBRID_PREDICTOR_GOVERNOR_MIN_CONFIDENCE`

Adaptive directive API (session duration adjusts to viscosity):

- `POST /api/nexus/hybrid/directive/work-session/adaptive`
- Request body: `{"project_id": null, "window_minutes": 120, "base_duration_minutes": 45}`
- Response includes `suggested_duration_minutes`, `strategy`, and merged `hybrid` + `viscosity` snapshots.

Multi-node coordination API (device handoff suggestion):

- `POST /api/nexus/hybrid/directive/work-session/multinode`
- Request body: `{"project_id": null, "window_minutes": 120, "base_duration_minutes": 45, "current_device_id": "phone", "candidate_device_ids": ["phone","laptop"]}`
- Response ranks devices by viscosity and returns `recommended_device_id` + `handoff_recommended`.

Auto-handoff launch API (analyze + propose in one call):

- `POST /api/nexus/hybrid/directive/work-session/auto-handoff-launch`
- If all candidate devices are `gated`, returns `launch_mode=recovery` and `suggested_duration_minutes=0`.
- Otherwise creates a trajectory + proposed task replica on the lowest-viscosity node.

One-tap confirm API:

- `POST /api/nexus/hybrid/directive/work-session/auto-handoff-confirm`
- Approves a proposed replica and queues execution idempotently (reuses existing pending action if already queued).

Chat trigger:

- Send "launch now" to run analyze → propose → confirm in one step.

Autonomy modes (set via `POST /api/nexus/policy` under `actions`):

- `autonomy_mode="strict"` → always manual proposal
- `autonomy_mode="balanced"` → auto-confirm only when low resistance and no handoff
- `autonomy_mode="auto"` → auto-confirm when governor + viscosity gates pass

Notes:

- `require_confirm=true` overrides autonomy modes and keeps manual confirm.
- Auto-launch responses may return `launch_mode=approved` with `queued_device_action_id`.

SelfReflection (analyze best sweeps):

- `GET /api/nexus/reflection?window_days=30&top_limit=5`
- `GET /api/nexus/reflection/daily-summary?window_hours=24`

Chat trigger for memory consolidation:

- Send "daily summary" (app or Telegram) to receive end-of-day accepted outcomes and focus adherence recap.

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
