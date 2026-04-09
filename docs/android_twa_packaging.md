# Android APK Packaging (Trusted Web Activity)

Use this when you want a **real downloadable APK** (Play Store style) that wraps your hosted Myco web app.

Suggested package naming:

- `com.mycelium.nexus.alpha` for the alpha channel
- keep the reverse-DNS form stable once published to Play Store

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

Play Store / TWA quality notes:

- Add PNG launcher icons at 192x192 and 512x512.
- Include a maskable icon variant for safe launcher cropping.
- Keep `display: standalone` and a dark `theme_color` for the Control Room look.
- The current repo uses `/static/icon.svg` in the manifest, which is fine for development, but PNGs are the safer production path.

You can generate the PNGs from the bundled SVG motif with:

```bash
python scripts/generate_twa_icons.py --out-dir static/twa-icons
```

The release checklist expects:

- `static/twa-icons/mycelium-192.png`
- `static/twa-icons/mycelium-192-maskable.png`
- `static/twa-icons/mycelium-512.png`
- `static/twa-icons/mycelium-512-maskable.png`

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

For repeatable releases, the repo also includes a helper script:

```bash
bash scripts/release_twa_build.sh
```

CI mode is also available:

```bash
CONFIRMED_RAILWAY_SYNC=true bash scripts/release_twa_build.sh --ci
```

The helper expects the Bubblewrap project directory to already exist.
Before it builds, it extracts the keystore SHA-256 fingerprint and stops if a local `.env` file is present but does not contain a matching `HIVE_ANDROID_SHA256` entry.
It also prompts for an explicit yes/no confirmation so you do not burn time on a build when Railway has not been updated yet.
Successful builds also write a small `version_bump.txt` tracker in the repo root so you can count release iterations.

GitHub Actions secrets for the tag-driven workflow:

- `MYCELIUM_RELEASE_KEYSTORE_B64`: Base64-encoded release keystore file.
- `MYCELIUM_KEYSTORE_PASSWORD`: Keystore password used by `keytool` and Bubblewrap.
- `MYCELIUM_KEY_PASSWORD`: Key password used by Bubblewrap signing.
- `MYCELIUM_RAILWAY_DOMAIN`: Live Railway domain used to build the manifest URL.
- Optional: `MYCELIUM_MANIFEST_URL` and `MYCELIUM_TWA_PACKAGE_ID`.

The workflow uses the Bubblewrap default signing alias `android`, so the keystore you upload should contain that alias unless you intentionally override it.

Secret setup guide: [github_actions_twa_release_secrets.md](github_actions_twa_release_secrets.md)

Launch-readiness checklist: [android_twa_launch_readiness_checklist.md](android_twa_launch_readiness_checklist.md)

Before packaging, confirm the versioning in your scaffold is set for release:

- `versionName`: `1.0.0-alpha` (or your next release tag)
- `versionCode`: increment this for every Play Store upload

## 4) Build APK / AAB

Inside the generated TWA folder:

```bash
bubblewrap build
```

For the one-shot path, the helper script will run this for you after validating the keystore and manifest inputs.

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

If you do not already have a release keystore, create one before signing the APK/AAB:

```bash
keytool -genkeypair -v -keystore mycelium-release.jks -alias mycelium -keyalg RSA -keysize 2048 -validity 10000
```

After generating the fingerprint, set `HIVE_ANDROID_SHA256` in Railway to the latest SHA-256 value from the release keystore.
The helper script also prints the legacy `ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV` name for compatibility.

## 7) Web push support (future-ready)

Bubblewrap/TWA can eventually surface push notifications from the web app into the Android tray. This repo is not wired for full web-push delivery yet, but the recommended future path is:

- keep the web app as the source of truth
- add a push-capable service worker flow
- send high-priority status alerts through the same notification bridge used by the app today

For now, Telegram and in-app notifications remain the active real-time channels.

## Notes for this repo

- This is the cleanest path to a “downloadable app” without rewriting frontend into native code.
- Backend remains unchanged.
- Your telemetry assistant, nudges, and approve/reject flow continue to work.
- Keep `COOKIE_SECURE=true` in production.
