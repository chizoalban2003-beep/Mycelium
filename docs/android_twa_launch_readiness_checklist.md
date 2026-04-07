# Android TWA Launch Readiness Checklist

Use this checklist right before you build and upload the first Mycelium Android package.

## Identity

- Confirm `packageId` in `twa-manifest.json` matches the Play Console package name.
- Confirm the signed keystore fingerprint matches the `HIVE_ANDROID_SHA256` value in Railway.
- Confirm `assetlinks.json` is served from the live Railway domain.

## Build

- Run `scripts/release_twa_build.sh` once interactively to validate the sync prompt.
- Run `scripts/release_twa_build.sh --ci` in automation only after setting `CONFIRMED_RAILWAY_SYNC=true`.
- Verify the script writes `version_bump.txt` after a successful build.

## Visuals

- Confirm the generated maskable icons exist in `static/twa-icons/`.
- Confirm the manifest points at the intended icon assets and theme colors.

## Security

- Confirm `NOTIFICATIONS_TELEGRAM_BOT_TOKEN` is set in Railway if Telegram inbound is enabled.
- Confirm `NOTIFICATIONS_TELEGRAM_WEBHOOK_SECRET` matches the webhook secret configured in Telegram.
- Confirm `COOKIE_SECURE=true` and production database settings are still active.

## Verification

- Curl `https://<your-domain>/.well-known/assetlinks.json` and compare the certificate fingerprint.
- Install the APK or AAB on a device and verify the app opens as a TWA rather than a browser tab.
- Re-run the Telegram webhook smoke tests after any command-path change.
