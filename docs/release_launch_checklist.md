# Mycelium Release Launch Checklist

This checklist is the practical next step for turning the repo into a production-ready release.

## What the app is today

Mycelium is a consent-first AI web platform with:

- user login and registration
- project and tree/workspace management
- Nexus ingestion for text, feedback, and structured signals
- a visible learning trail and knowledge audit UI
- a reasoning card that surfaces `improvement_frac` from synthetic validation traces
- policy controls, including a kill-switch for device actions and diagnostics
- Telegram and email recovery channels when configured

## Production path

### Railway

Use Railway for the hosted backend and web UI.

Required production settings:

- `DATABASE_URL` pointing to Railway Postgres
- `SECRET_KEY` set to a long random value
- `COOKIE_SECURE=true`
- `DB_MIGRATION_MODE=migrate`
- `DB_AUTO_CREATE_TABLES=false`
- `CORS_ALLOW_ORIGINS_CSV` set to your public app origin

Suggested validation:

1. Deploy the container.
2. Confirm `/health` responds.
3. Confirm `/docs` is reachable.
4. Register a test user.
5. Verify login, project creation, and Nexus audit endpoints.

### Downloadable Android app

If you want a downloadable app, use the hosted web app as a Trusted Web Activity (TWA).

Use [docs/android_twa_packaging.md](docs/android_twa_packaging.md) for the build path.

That gives you:

- a real APK for sideloading or testing
- an AAB for Google Play submission
- full-screen Android launch while keeping the web app as the source of truth

## Launch checklist

- [ ] Set production env vars
- [ ] Use Postgres in production
- [ ] Deploy to Railway and confirm health/docs
- [ ] Create a test user and verify auth
- [ ] Verify `improvement_frac` appears in the reasoning card flow
- [ ] Verify kill-switch blocks the synthetic stress-test route
- [ ] Confirm asset links for Android TWA
- [ ] Build APK/AAB with Bubblewrap
- [ ] Test install on one Android device
- [ ] Submit to Play Store when branding, privacy policy, and screenshots are ready

## Current status

This repo is already in a good alpha shape for:

- hosted web deployment
- consent-gated app actions
- synthetic validation and reasoning summaries
- a TWA-based Android packaging path

The remaining work for a public release is mostly ops and packaging, not core app reconstruction.
