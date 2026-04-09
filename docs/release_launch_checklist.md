# Myco Go-Live Checklist

This is the practical checklist for turning the repo into a production deployment.

## Ready now

- [x] Consent-first web app surface exists
- [x] Hosted backend path is documented for Railway/Postgres
- [x] Container restart policies are set for API and Postgres
- [x] `/health` endpoint exists
- [x] Container healthcheck is configured in `docker-compose.yml`
- [x] Strict production preflight exists: `scripts/db_migration_preflight.py`
- [x] Uptime smoke test exists: `scripts/health_smoketest.py`
- [x] Backup helper exists: `scripts/backup_db.sh`
- [x] Restore helper exists: `scripts/restore_db.sh`
- [x] Local child-agent install path exists
- [x] Android TWA packaging path exists

## Go-live checks

### Production config

- [ ] Copy `.env.production.example` into Railway environment variables
- [ ] Follow [docs/railway_production_env.md](docs/railway_production_env.md) for the exact Railway variable set
- [ ] Follow [docs/railway_deploy_runbook.md](docs/railway_deploy_runbook.md) during the Railway setup
- [ ] Use [docs/railway_launch_pack.md](docs/railway_launch_pack.md) as the primary Railway launch packet
- [ ] If you want the fastest path, use [docs/railway_today_deploy.md](docs/railway_today_deploy.md)
- [ ] Set `SECRET_KEY` to a long random production value
- [ ] Set `DATABASE_URL` to Railway Postgres
- [ ] Set `COOKIE_SECURE=true`
- [ ] Set `DB_MIGRATION_MODE=migrate`
- [ ] Keep `DB_AUTO_CREATE_TABLES=false`
- [ ] Set `CORS_ALLOW_ORIGINS_CSV` to the public app origin

### Preflight

- [ ] Run `STRICT_PRODUCTION=true python scripts/db_migration_preflight.py`
- [ ] Confirm the preflight exits cleanly
- [ ] Verify no placeholder secrets remain in deployment settings
- [ ] Run `BASE_URL=https://<your-app> TOKEN=<api-token> ./.venv/bin/python scripts/deploy_readiness_check.sh`

### Deployment smoke test

- [ ] Deploy the container
- [ ] Confirm `/health` responds
- [ ] Confirm `/docs` is reachable
- [ ] Run `BASE_URL=https://<your-app> ./.venv/bin/python scripts/health_smoketest.py`
- [ ] Create a test user
- [ ] Verify login works
- [ ] Verify project creation works
- [ ] Verify Nexus audit endpoints respond

### Backup and recovery

- [ ] Back up Postgres
- [ ] Restore into a fresh environment
- [ ] Confirm the restore matches expected data

### Android TWA

- [ ] Generate launcher icons and maskable variants
- [ ] Confirm asset links for Android TWA
- [ ] Set `versionName` and `versionCode`
- [ ] Generate or confirm the release keystore
- [ ] Create the Bubblewrap project directory
- [ ] Build APK/AAB with Bubblewrap
- [ ] Test install on one Android device

### Release

- [ ] Confirm branding, privacy policy, and screenshots
- [ ] Submit to Play Store when the build is verified

## Remaining hardening

- [ ] Add observability/alerting beyond basic smoke tests
- [ ] Run a short soak test on the API and child-agent loop
- [ ] Finalize secret rotation procedures
- [ ] Review HTTPS/session-cookie behavior in the hosted environment

## Current status

This repo is now in a **pilot-ready** state for deployment, with the main production gates and recovery paths in place. The remaining work is mostly launch operations, observability, and packaging.
