# Production Readiness Summary

Myco is already **deployable pilot-ready**, but not yet fully hardened for an unattended public launch.

## What looks production-oriented

- Clear consent-first product surface
- Hosted web deployment path via Railway/Postgres
- Explicit production settings in `requirements/prod.txt`, `Dockerfile`, and `docker-compose.yml`
- Background child-agent service path for local companion telemetry
- Android TWA packaging path and launch checklist
- Visible learning, validation, and policy controls

## What still needs hardening

1. Database migration discipline
   - Keep `DB_AUTO_CREATE_TABLES=false`
   - Use `DB_MIGRATION_MODE=migrate`
   - Run schema changes through a repeatable migration path

2. Secrets and configuration hygiene
   - Replace placeholder secrets such as `SECRET_KEY`
   - Store tokens, SMTP secrets, and webhook secrets in the platform secret store
   - Document rotation and revocation for external credentials
   - Run `STRICT_PRODUCTION=true python scripts/db_migration_preflight.py` before release
   - Start from [`.env.production.example`](../.env.production.example)
   - Use [docs/railway_production_env.md](docs/railway_production_env.md) for the exact Railway variable set
   - Use [docs/railway_deploy_runbook.md](docs/railway_deploy_runbook.md) for the step-by-step Railway rollout
   - Use [docs/railway_launch_pack.md](docs/railway_launch_pack.md) as the single launch packet
   - Use [docs/railway_today_deploy.md](docs/railway_today_deploy.md) for the fastest launch path

3. Observability
   - Add uptime checks for `/health` and main write paths
   - Track app errors, login failures, and queue/daemon failures
   - Confirm log retention and incident triage steps
   - Use `scripts/health_smoketest.py` after deploy to confirm `/health` and `/docs`
   - Compose now restarts both API and Postgres automatically
   - Use `scripts/deploy_readiness_check.sh` for a single deploy gate

4. Recovery and backup
   - Back up Postgres regularly
   - Use `scripts/backup_db.sh` and `scripts/restore_db.sh` for a basic recovery path
   - Test restore into a fresh environment
   - Avoid relying on SQLite fallback data in production

5. Load and soak validation
   - Run short soak tests on the API and the child-agent loop
   - Validate background daemons do not starve the web app under load

6. Release packaging
   - Finalize Android TWA branding, asset links, and keystore handling
   - Confirm the public manifest and Play submission path end to end

7. Security review
   - Recheck auth/session cookie settings behind HTTPS
   - Confirm allowlists and consent gates for telemetry/device actions
   - Verify no debug defaults ship in production manifests

## Practical verdict

The repo reads like a **production-minded pilot**: good architecture, meaningful deployment scaffolding, and a clear consent model. The remaining work is mostly operational hardening rather than feature reconstruction.
