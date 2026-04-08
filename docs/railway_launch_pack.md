# Railway Launch Pack

This is the single copy/paste packet for launching Mycelium on Railway.

## 1) Set environment variables

Use [docs/railway_production_env.md](docs/railway_production_env.md) for the full set.

Copy this into Railway and replace the placeholders:

```bash
APP_NAME=Mycelium
SECRET_KEY=<long-random-secret>
DATABASE_URL=<railway-postgres-url>
COOKIE_SECURE=true
DB_MIGRATION_MODE=migrate
DB_AUTO_CREATE_TABLES=false
CORS_ALLOW_ORIGINS_CSV=https://<your-public-app-domain>
STRICT_PRODUCTION=true
NEXUS_DEVICE_ID=parent-hub
HIVE_ENABLED=true
HIVE_INGEST_TOKEN=<long-random-token>
HIVE_EXPORT_ENABLED_DEFAULT=false
HIVE_HEALTH_ALLOWLIST_EMAILS_CSV=
NEXUS_HOMEOSTASIS_ENABLED=true
APP_PUBLIC_BASE_URL=https://<your-public-app-domain>
CORS_ALLOW_CREDENTIALS=true
```

### Required

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

### Recommended

- `HIVE_EXPORT_ENABLED_DEFAULT=false`
- `HIVE_HEALTH_ALLOWLIST_EMAILS_CSV=`
- `NEXUS_HOMEOSTASIS_ENABLED=true`
- `APP_PUBLIC_BASE_URL=https://<your-public-app-domain>`
- `CORS_ALLOW_CREDENTIALS=true`

### Optional later

- `NOTIFICATIONS_BRIDGE_ENABLED=false`
- `MAIL_ENABLED=false`
- `NEXUS_TELEMETRY_ASSISTANT_ENABLED=false`
- `ANDROID_APP_PACKAGE_NAME=`
- `ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV=`

## 2) Deploy

1. Connect the repo to Railway.
2. Use the root of this repository.
3. Keep the provided `Dockerfile`.
4. Attach Railway Postgres.
5. Set `DATABASE_URL` from Railway Postgres.
6. Deploy.

## 3) Run the launch checks

Run these from the repo root after deploy:

```bash
STRICT_PRODUCTION=true ./.venv/bin/python scripts/db_migration_preflight.py
BASE_URL=https://<your-app> ./.venv/bin/python scripts/health_smoketest.py
BASE_URL=https://<your-app> TOKEN=<api-token> ./.venv/bin/python scripts/deploy_readiness_check.sh
```

## 4) Verify manually

Check these in the browser:

- `/docs`
- `/login`
- `/predict`
- create a test user
- create a project
- confirm a basic Nexus flow works

## 5) Recovery checks

- confirm Postgres backups are enabled
- restore one backup into a fresh environment
- verify the restored app can log in and read data

## 6) Android TWA checks

If you are shipping Android:

- generate launcher icons
- verify asset links
- confirm `versionName` and `versionCode`
- build the TWA with Bubblewrap
- test install on one device

## 7) Go/no-go

Go live only when all of these pass:

- strict preflight passes
- health smoke test passes
- deploy readiness check passes
- backups are confirmed
- Android packaging checks pass, if applicable
