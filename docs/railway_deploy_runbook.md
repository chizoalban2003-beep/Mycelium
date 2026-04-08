# Railway Deploy Runbook

This is the copy/paste path for deploying Mycelium to Railway.

## 1) Prepare secrets

Set these Railway variables first:

- `APP_NAME=Mycelium`
- `SECRET_KEY=<long-random-secret>`
- `DATABASE_URL=<railway-postgres-url>`
- `COOKIE_SECURE=true`
- `DB_MIGRATION_MODE=migrate`
- `DB_AUTO_CREATE_TABLES=false`
- `CORS_ALLOW_ORIGINS_CSV=https://<your-public-app-domain>`
- `STRICT_PRODUCTION=true`
- `NEXUS_DEVICE_ID=parent-hub`
- `HIVE_ENABLED=true`
- `HIVE_INGEST_TOKEN=<long-random-token>`

Recommended:

- `HIVE_EXPORT_ENABLED_DEFAULT=false`
- `HIVE_HEALTH_ALLOWLIST_EMAILS_CSV=`
- `NEXUS_HOMEOSTASIS_ENABLED=true`
- `APP_PUBLIC_BASE_URL=https://<your-public-app-domain>`
- `CORS_ALLOW_CREDENTIALS=true`

## 2) Add optional features later

Leave these off until you actually use them:

- `NOTIFICATIONS_BRIDGE_ENABLED=false`
- `MAIL_ENABLED=false`
- `NEXUS_TELEMETRY_ASSISTANT_ENABLED=false`
- `ANDROID_APP_PACKAGE_NAME=`
- `ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV=`

## 3) Deploy the app

1. Connect the repo to Railway.
2. Point the service at the root of this repo.
3. Use the provided `Dockerfile`.
4. Add Railway Postgres and wire `DATABASE_URL`.
5. Deploy the container.

## 4) Run launch checks

Run these after deploy:

```bash
STRICT_PRODUCTION=true ./.venv/bin/python scripts/db_migration_preflight.py
BASE_URL=https://<your-app> ./.venv/bin/python scripts/health_smoketest.py
BASE_URL=https://<your-app> TOKEN=<api-token> ./.venv/bin/python scripts/deploy_readiness_check.sh
```

## 5) Verify the app

Check these manually in the browser:

- `/docs`
- `/login`
- `/predict`
- create a test user
- create a project
- run a simple Nexus action flow

## 6) Backup and recovery

Before public use:

- confirm Postgres backups are enabled in Railway
- restore one backup into a fresh environment
- verify the restored app can log in and read data

## 7) Android TWA

If you are shipping Android:

- generate launcher icons
- verify asset links
- confirm `versionName` and `versionCode`
- build the TWA with Bubblewrap
- test install on one device

## 8) Go/no-go

Go live only when:

- the strict preflight passes
- the health smoke test passes
- the deploy readiness check passes
- backups are confirmed
- the Android packaging checks pass, if applicable
