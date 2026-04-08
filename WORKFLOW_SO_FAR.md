# Mycelium workflow so far (Apr 2, 2026)

## 1) What Mycelium is right now

- A FastAPI app with login + project UI, plus an electrophoresis-inspired predictor.
- You can upload a CSV, pick any column as the target (numeric/categorical/datetime), and Mycelium:
  - infers feature/target kinds,
  - builds association weights ("physics-like" pulls/drag),
  - runs an electrophoresis simulation with v4 cascade (Zone-1 fractionation, inhibition, optional thermal noise, scavenger recycling),
  - returns metrics + explanations (weights, migration map, bonding map, zones, iteration gains).

## 2) Run the web app

From repo root:

- Start server:
  - `uvicorn mycelium_app.main:app --reload --host 0.0.0.0 --port 8000`

- Open:
  - `http://localhost:8000/login`

- Create a user (if needed):
  - `./.venv/bin/python scripts/create_user.py`

## 3) Use the predictor UI

- Open:
  - `http://localhost:8000/predict`

- Fill in:
  - CSV file
  - Target column name
  - Plane (`solid`, `liquid`, `gas`)
  - Train ratio + optional `No split`
  - Random seed

- Advanced (optional):
  - `Cycles`, `Learning rate`
  - `Cascade enabled`, `Competitive inhibition`, `Thermal noise`
  - `Stage-2 cycles`, `Scavenger cycles`, `Inhibition strength`

Outputs shown:
- Arrival velocity ranking
- Migration map (entropy/variance/density/viscosity)
- Preview rows (actual vs predicted)

## 4) Use the JSON API endpoint

- Endpoint:
  - `POST /api/predict/electrophoresis`

- Form fields (common):
  - `file` (CSV)
  - `target_col`
  - `plane`
  - `top_k`
  - `train_ratio`
  - `random_seed`
  - `no_split`
  - `cascade_enabled`
  - `competitive_inhibition`
  - `thermal_noise`
  - `max_rows`

Returns JSON:
- `weights`, `migration_map`, `bonding_map`, `equilibrium_zones`
- `iteration_gains` (classification: `test_accuracy`; regression: `test_mae/test_rmse`)
- `metrics` (includes `baseline_*` and best-cycle info)

## 5) Benchmarking on the salary dataset

Dataset:
- `tmp_eval/job_salary_prediction_dataset.csv`
- Columns include: job_title, experience_years, education_level, skills_count, industry, company_size, location, remote_work, certifications, salary

### Regression (predict `salary`)

Compare Mycelium electrophoresis vs Decision Tree vs Random Forest vs Gradient Boosting vs Neural Net (MLP):

- `./.venv/bin/python scripts/benchmark_salary_models.py --nrows 50000 --target salary --train-fraction 0.8 --seed 42`

Optional:
- Disable baselines if you only want some models:
  - `--no-tree`
  - `--no-mlp`

Latest observed (after regression boosting + tuned defaults):

| Model | MAE | RMSE | Time (s) |
|---|---:|---:|---:|
| Mycelium v4 | 5534.376 | 7240.357 | 5.19 |
| RandomForest | 7024.342 | 8982.273 | 26.64 |
| HistGB | 4374.905 | 5499.121 | 4.65 |

Notes:
- These numbers are for `--nrows 50000`, `--seed 42`, and `--train-fraction 0.8`.
- Runtime will vary by machine; metrics should be close if the split/seed are the same.

### Classification (predict `remote_work`)

Compare Mycelium electrophoresis vs Decision Tree vs Random Forest vs Gradient Boosting vs Neural Net (MLP), and print a full `classification_report` on the test split:

- `./.venv/bin/python scripts/benchmark_salary_classifiers.py --nrows 50000 --target remote_work --train-fraction 0.8 --seed 42 --report`

Add confusion matrices (raw + normalized):

- `./.venv/bin/python scripts/benchmark_salary_classifiers.py --nrows 50000 --target remote_work --train-fraction 0.8 --seed 42 --report --confusion`

## 6) Key algorithm milestones implemented

- MVP physics-weight predictor for any target type.
- Train/test split (default 80/20), `no_split`, and `random_seed` for reproducibility.
- v4 cascade mechanics:
  - Stage-1 migration → equilibrium zones
  - Zone-1 secondary fractionation (Stage 2)
  - competitive inhibition
  - optional thermal noise early cycles
  - cluster shattering into sub-zones (IDs 100+)
  - scavenger recycling
- Regression (numeric) upgraded to multi-cycle residual refinement (boosting-style) + cascade.

## 7) Nexus Homeostasis ("Body")

Homeostasis is a small stability layer that periodically computes:
- mood + identity hash (from recent telemetry/experience)
- resource health (disk free/total)
- optional pruning under low disk

It persists a snapshot into `HomeostasisState` so other subsystems (like the predictor) can consult it.

### Homeostasis failure behavior (policy)

If Homeostasis throws an exception in the background daemon:
- The app stays up ("degrade gracefully")
- A throttled Hive outbox message is queued: `kind=homeostasis_failure` with `action=ask_parent`

This creates a clear audit trail without crashing the server.

## 8) HiveEmpathy (collective learning) — “wisdom whispers”

HiveEmpathy is a lightweight, privacy-conservative protocol to share **best known stable predictor knobs** across devices.

### What a whisper contains

Queued as a Hive outbox message with `kind=wisdom_whisper`.

Payload shape (high-level):
- `meta`: created_at, device_id, project_id, kind, version
- `homeostasis`: mood, identity_hash
- `wisdom.recommended_kwargs`: **only allowlisted knobs** (same safety filtering as Physics Ledger recall)
- `wisdom.evidence`: counts + coarse score stats (no raw feature names or dataset content)

### Queue a whisper (server-side builder)

- `POST /api/hive/whisper/queue`
  - body: `{ "project_id": null, "limit": 200 }`

This summarizes your top Physics Ledger entries and queues one message into the outbox.

### List recent outbox messages

- `GET /api/hive/outbox/messages/recent?limit=50`
- Optional filter by kind:
  - `GET /api/hive/outbox/messages/recent?kind=wisdom_whisper`
  - `GET /api/hive/outbox/messages/recent?kind=homeostasis_failure`

Notes:
- All Hive export is gated by:
  - `settings.hive_enabled = true`
  - parental policy `privacy.export_enabled = true`

### Parent-side ingest (store as Global Update)

To ingest a `wisdom_whisper` from another device and make it available via global updates:

- `POST /api/hive/whisper/import`
  - body: `{ "whisper": { ...full whisper payload... } }`

The server will upsert it into `HiveGlobalUpdate` idempotently using a deterministic UUID (SHA256 of the whisper JSON) unless you provide `update_uuid`.

### WisdomBroadcast (Genetic Push)

Children can fetch the latest aggregated baseline physics knobs (derived from imported whispers):

- `GET /api/hive/wisdom/latest`

Optional parameters:
- Scope to a specific project: `?project_id=123`
- Include project-scoped whispers in global mode: `?include_project_scoped=true`

## 9) IdentityPresentation + Nudges ("Voice")

The child can now describe itself and proactively surface short notifications.

### IdentityPresentation

- `GET /api/nexus/identity/presentation`

Returns a deterministic `display_name` + `tagline` computed from `identity_hash` and current mood.

### Nudges

- List unseen nudges: `GET /api/nexus/nudges/recent?unseen_only=true&limit=5`
- Acknowledge (dismiss): `POST /api/nexus/nudges/ack` with body `{ "nudge_id": 123 }`

Autonomous nudge: when Hive wisdom changes meaningfully, the server queues a `wisdom_update` nudge for active users.

## 10) Validation Shadow (MetricSnapshot) — “Empirical Nudges”

If you want the nudge to honestly say “I’m ~12% better now”, enable the Validation Shadow.

When enabled, on wisdom change the system will:
- run a mini-benchmark on a local CSV evaluation set using the *old* kwargs and the *new* broadcast kwargs
- store both results as `MetricSnapshot` rows (baseline + trial)
- only nudge the user if the measured delta is positive and above the threshold

### Enable (via env vars)

Set in your `.env` (or environment):
- `NEXUS_VALIDATION_SHADOW_ENABLED=true`
- `NEXUS_VALIDATION_SHADOW_DATASET_PATH=/absolute/path/to/eval.csv`
- `NEXUS_VALIDATION_SHADOW_TARGET_COL=salary` (or your target)

Optional tuning:
- `NEXUS_VALIDATION_SHADOW_MAX_ROWS=5000`
- `NEXUS_VALIDATION_SHADOW_N_CYCLES=12`
- `NEXUS_VALIDATION_SHADOW_MIN_IMPROVEMENT_FRAC=0.02` (2%)

## 11) CausalTrace (MetricCausalTrace) — “Why did it improve?”

When Validation Shadow runs, the predictor also produces an explanation surface (`weights`).

If both baseline + trial runs succeed, the system will:
- compare baseline vs trial feature weights to find the biggest “pull” shifts
- persist that explanation as a `MetricCausalTrace` row (linked to the baseline/trial `MetricSnapshot` IDs)
- append a short explanatory clause to the `wisdom_update` nudge when an empirical improvement is reported

## 12) Active Curiosity — “Ask for Human Ground Truth”

When enabled, after a prediction run the system will capture a few high-error ("agitated") test samples as `CuriosityCase` rows and (optionally) nudge the user to answer them.

User flow:
- Visit `/curiosity` to see the next pending question and answer/dismiss it.
- Answers are stored as `CuriosityAnswer`.

Hive flow (privacy-safe):
- The system exports only coarse tags + meta as an outbox message kind `curiosity_feedback`.
- It does **not** export freeform answer text or raw row excerpts.

Parent ingest (optional):
- `POST /api/hive/curiosity/import` stores a `curiosity_feedback` payload as a `HiveGlobalUpdate`.
- `GET /api/hive/wisdom/latest` includes aggregated curiosity `top_tags` under `evidence.curiosity`.

### Enable (via env vars)

- `NEXUS_ACTIVE_CURIOSITY_ENABLED=true`
- `NEXUS_ACTIVE_CURIOSITY_SAFE_COLUMNS_CSV=job_title,city,seniority` (optional; only these columns are shown on `/curiosity`)

Optional tuning:
- `NEXUS_ACTIVE_CURIOSITY_MAX_CASES_PER_RUN=3`
- `NEXUS_ACTIVE_CURIOSITY_MIN_ERROR_QUANTILE=0.97`
- `NEXUS_ACTIVE_CURIOSITY_MIN_ABS_ERROR=0.0`
- `NEXUS_ACTIVE_CURIOSITY_NUDGE_ENABLED=true`
- `NEXUS_ACTIVE_CURIOSITY_NUDGE_THROTTLE_MINUTES=120`

## 5) Device Shell UI Polish (Apr 7, 2026)

The device shell (`/device`) has been refined to feel more like a native app dashboard:

- **Compact Header**: Reduced padding and title sizes for tighter desktop layout.
- **Interactive Today Tiles**: Activity tiles (Signals, Growth, Nudges) are now clickable links to relevant pages (/knowledge, /projects, /hive/health) with hover effects.
- **Collapsible Status Sidebar**: Added toggle button to hide/show the status panel for better space management.
- **Enhanced Learning Trail**: Added icons (🔍 Observed, 🔧 Normalized, ✅ Explained) for better visual hierarchy.
- **Quick Settings**: Added a settings button in the header linking to assistant profile.
- **Mobile Optimizations**: Improved bottom navigation bar with consistent touch targets.

Architecture confirmed: HIVE collaboration, child-safe controls, physics-guided prediction, and data-driven growth remain core. The UI now better supports transparent, device-native AI learning loops.
