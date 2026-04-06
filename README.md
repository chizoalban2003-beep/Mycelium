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

Benchmarks / scratch data:
- `tmp_eval/` is ignored by git (local outputs + datasets). This repo no longer ships the benchmark dataset.
- Use your own CSV when running benchmark scripts, e.g. `python scripts/benchmark_salary_models.py --csv /path/to/data.csv --target salary`.
- Or generate a synthetic sample dataset: `python scripts/sample_salary_dataset.py --out tmp_eval/sample_salary_dataset.csv`, then pass it into scripts.
