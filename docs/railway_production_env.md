# Railway Production Environment Variables

Use this as the concrete Railway configuration for a production deployment.

## Required

Set these before launch:

- `APP_NAME=Myco`
- `SECRET_KEY=<long-random-secret>`
- `DATABASE_URL=<railway-postgres-url>`
- `COOKIE_SECURE=true`
- `DB_MIGRATION_MODE=migrate`
- `DB_AUTO_CREATE_TABLES=false`
- `CORS_ALLOW_ORIGINS_CSV=https://<your-public-app-domain>`
- `NEXUS_DEVICE_ID=parent-hub`
- `HIVE_ENABLED=true`
- `HIVE_INGEST_TOKEN=<long-random-token>`

## Recommended

These are strongly recommended for a production launch:

- `STRICT_PRODUCTION=true`
- `HIVE_EXPORT_ENABLED_DEFAULT=false`
- `HIVE_HEALTH_ALLOWLIST_EMAILS_CSV=`
- `NEXUS_HOMEOSTASIS_ENABLED=true`
- `APP_PUBLIC_BASE_URL=https://<your-public-app-domain>`
- `CORS_ALLOW_CREDENTIALS=true`

## Optional but common

Enable only if you use these features:

- `NOTIFICATIONS_BRIDGE_ENABLED=false`
- `NOTIFICATIONS_TELEGRAM_BOT_TOKEN=`
- `NOTIFICATIONS_TELEGRAM_WEBHOOK_SECRET=`
- `MAIL_ENABLED=false`
- `MAIL_FROM_ADDRESS=noreply@your-domain.com`
- `MAIL_SMTP_HOST=`
- `MAIL_SMTP_PORT=587`
- `MAIL_SMTP_USERNAME=`
- `MAIL_SMTP_PASSWORD=`
- `MAIL_SMTP_USE_TLS=true`
- `MAIL_SMTP_USE_SSL=false`
- `ANDROID_APP_PACKAGE_NAME=`
- `ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV=`
- `NEXUS_TELEMETRY_ASSISTANT_ENABLED=false`
- `NEXUS_TELEMETRY_ASSISTANT_TICK_SECONDS=60`
- `NEXUS_TELEMETRY_ASSISTANT_WINDOW_HOURS=6`
- `NEXUS_TELEMETRY_ASSISTANT_CONFIDENCE_THRESHOLD=0.85`
- `NEXUS_TELEMETRY_ASSISTANT_THROTTLE_MINUTES=120`

## Rollout order

1. Set the required variables.
2. Run `STRICT_PRODUCTION=true python scripts/db_migration_preflight.py`.
3. Deploy the container.
4. Run `BASE_URL=https://<your-app> ./.venv/bin/python scripts/health_smoketest.py`.
5. Run `BASE_URL=https://<your-app> TOKEN=<api-token> ./.venv/bin/python scripts/deploy_readiness_check.sh`.

## Notes

- Use Railway Postgres for `DATABASE_URL`.
- Keep `DB_AUTO_CREATE_TABLES=false` in production.
- Keep `COOKIE_SECURE=true` behind HTTPS.
- Treat `SECRET_KEY` and `HIVE_INGEST_TOKEN` as secrets, not defaults.
