# Production Readiness Summary

Mycelium is already **deployable pilot-ready**, but not yet fully hardened for an unattended public launch.

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

3. Observability
   - Add uptime checks for `/health` and main write paths
   - Track app errors, login failures, and queue/daemon failures
   - Confirm log retention and incident triage steps
   - Use `scripts/health_smoketest.py` after deploy to confirm `/health` and `/docs`
   - Compose now restarts both API and Postgres automatically

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
