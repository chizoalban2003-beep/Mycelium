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

Run the benchmark (50k rows):
- `./.venv/bin/python scripts/benchmark_salary_models.py --nrows 50000 --target salary --train-fraction 0.8 --seed 42`

Latest observed (after regression boosting + tuned defaults):

| Model | MAE | RMSE | Time (s) |
|---|---:|---:|---:|
| Mycelium v4 | 5534.376 | 7240.357 | 5.19 |
| RandomForest | 7024.342 | 8982.273 | 26.64 |
| HistGB | 4374.905 | 5499.121 | 4.65 |

Notes:
- These numbers are for `--nrows 50000`, `--seed 42`, and `--train-fraction 0.8`.
- Runtime will vary by machine; metrics should be close if the split/seed are the same.

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
