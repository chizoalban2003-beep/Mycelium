# Deploy to Railway Today

Use this page if you want the fastest safe path from this repo to a live Railway deployment.

## Target

- **Platform:** Railway
- **Runtime:** Dockerfile on Railway + Railway Postgres
- **Deployment type:** controlled pilot first, then public launch after validation

## Time estimate

- **Controlled pilot:** 1–2 hours if Railway, Postgres, and domain/DNS are already ready
- **Public-ready launch:** 1–2 days once you finish smoke tests, backups, and Android packaging

## Cost estimate

- **Railway:** usually low tens of dollars per month for a small pilot, depending on usage and plan
- **Google Play:** $25 one-time if you ship the Android TWA

## Do this now

### 1) Set Railway variables

Copy the required values from [docs/railway_launch_pack.md](docs/railway_launch_pack.md) or [docs/railway_production_env.md](docs/railway_production_env.md).

### 2) Attach Railway Postgres

- Create a Railway Postgres database
- Copy the database URL into `DATABASE_URL`
- Keep `COOKIE_SECURE=true`
- Keep `DB_MIGRATION_MODE=migrate`
- Keep `DB_AUTO_CREATE_TABLES=false`

### 3) Deploy the service

- Use the root of this repo
- Use the provided `Dockerfile`
- Deploy the container

### 4) Run the launch gate

From the repo root:

```bash
STRICT_PRODUCTION=true ./.venv/bin/python scripts/db_migration_preflight.py
BASE_URL=https://<your-app> ./.venv/bin/python scripts/health_smoketest.py
BASE_URL=https://<your-app> TOKEN=<api-token> ./.venv/bin/python scripts/deploy_readiness_check.sh
```

### 5) Verify manually

Open these pages in the browser:

- `/docs`
- `/login`
- `/predict`

Then:

- create a test user
- create a project
- verify a basic Nexus flow works

### 6) Confirm recovery

- confirm backups are enabled
- restore one backup into a fresh environment
- verify login and data access after restore

### 7) Go/no-go

Go live only when:

- strict preflight passes
- health smoke test passes
- deploy readiness check passes
- backup restore works
- Android packaging checks pass if you are shipping mobile

## If something fails

- Fix the env var or secret first
- Re-run the strict preflight
- Re-run the health smoke test
- Re-run the deploy readiness check

