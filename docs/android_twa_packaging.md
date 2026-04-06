# Android APK Packaging (Trusted Web Activity)

Use this when you want a **real downloadable APK** (Play Store style) that wraps your hosted Mycelium web app.

This uses a Trusted Web Activity (TWA):
- Web app stays hosted (Railway or your domain)
- Android app launches the web app full-screen
- APK can be installed directly or published to Play Store

## Prerequisites

- Public HTTPS URL for your app (example: `https://your-app.up.railway.app`)
- PWA already enabled (manifest + service worker present in this repo)
- Node.js + Java 17+
- Android Studio (or Android SDK command-line tools)

## 1) Verify PWA readiness

Your app should already expose:
- `/static/manifest.webmanifest`
- `/static/sw.js`

Quick check:

```bash
curl -I https://your-app.up.railway.app/static/manifest.webmanifest
curl -I https://your-app.up.railway.app/static/sw.js
```

## 2) Install Bubblewrap

```bash
npm install -g @bubblewrap/cli
```

If global install is restricted, use:

```bash
npx @bubblewrap/cli --help
```

## 3) Initialize the TWA project

```bash
bubblewrap init --manifest https://your-app.up.railway.app/static/manifest.webmanifest
```

This creates Android project files under a local folder (default `twa/`).

## 4) Build APK / AAB

Inside the generated TWA folder:

```bash
bubblewrap build
```

Outputs:
- APK for direct install/testing
- AAB for Play Store upload

## 5) Test on phone

```bash
adb install app-release-signed.apk
```

Or copy APK to phone and install manually.

## 6) Digital Asset Links (important)

For full trusted behavior, host `assetlinks.json` on your web origin:

- URL: `https://your-app.up.railway.app/.well-known/assetlinks.json`

Bubblewrap prints the exact JSON you need after initialization/signing.

## Notes for this repo

- This is the cleanest path to a “downloadable app” without rewriting frontend into native code.
- Backend remains unchanged.
- Your telemetry assistant, nudges, and approve/reject flow continue to work.
- Keep `COOKIE_SECURE=true` in production.
