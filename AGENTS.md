# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Mycelium is a FastAPI-based Python web application — a local-first, physics-inspired prediction engine and SaaS platform. It uses SQLite by default for development (zero config), Jinja2 templates, and includes background daemons for homeostasis, wisdom nudges, telemetry, and messaging.

### Running the dev server

```bash
pip install -r requirements/base.txt
cp .env.example .env          # only needed on first setup
uvicorn mycelium_app.main:app --reload --host 0.0.0.0 --port 8000
```

The server auto-creates the SQLite DB at `storage/mycelium.db` on startup when `DB_AUTO_CREATE_TABLES=true` (the default).

### Creating a test user

```bash
python3 scripts/create_user.py --email dev@example.com --password devpass123 --full-name "Dev User"
```

Or via the API:

```bash
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"dev@example.com","password":"devpass123","full_name":"Dev User"}'
```

### Getting an auth token

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'username=dev@example.com' \
  --data-urlencode 'password=devpass123'
```

Returns `{"access_token": "...", "token_type": "bearer"}`.

### CI smoke tests

The CI workflow (`.github/workflows/release-gate.yml`) runs these scripts in order:

1. `python3 scripts/db_migration_preflight.py` — validates DB config
2. Start the server, then:
   - `scripts/smoke_autonomy_handoff_flow.py --base-url ... --token ...`
   - `scripts/eval_autonomy_modes.py --base-url ... --token ...`
   - `scripts/global_audit_report.py --base-url ... --token ...`

### Gotchas

- `pip install` puts binaries in `~/.local/bin`, which may not be on `PATH`. Run `export PATH="$HOME/.local/bin:$PATH"` if `uvicorn` is not found.
- There is no `pyproject.toml`, `setup.py`, or formal test suite (no pytest/unittest). The project relies on CI smoke scripts and the prediction engine benchmarks for validation.
- The prediction endpoint is `POST /api/predict/electrophoresis` and requires a CSV file upload (`multipart/form-data`), not a JSON body.
- The `smoke_app.sh` script's "Live state" call (`/api/nexus/live/state`) requires authentication — a 401 is expected when called without a token.
- PostgreSQL is only needed for production-like testing; SQLite is sufficient for development.
