# GitHub Actions TWA Release Secrets

Use these secrets when running the tag-driven TWA release workflow in [.github/workflows/twa-release.yml](../.github/workflows/twa-release.yml).

## Required secrets

- `MYCELIUM_RELEASE_KEYSTORE_B64`
  - Base64-encoded `.jks` file for the Android release keystore.
- `MYCELIUM_KEYSTORE_PASSWORD`
  - Keystore password used by `keytool` and Bubblewrap.
- `MYCELIUM_KEY_PASSWORD`
  - Key password used by Bubblewrap when signing the APK/AAB.
- `MYCELIUM_RAILWAY_DOMAIN`
  - Live Railway domain, for example `mycelium.up.railway.app`.

## Optional secrets

- `MYCELIUM_MANIFEST_URL`
  - Override the manifest URL if you do not want the default `/static/manifest.webmanifest`.
- `MYCELIUM_TWA_PACKAGE_ID`
  - Override the Android package id if you are not using the default `com.mycelium.nexus.alpha`.

## Encode the keystore

From Linux:

```bash
base64 -w 0 mycelium-release.jks > mycelium-release.jks.b64
```

Then copy the output into the secret value or use the GitHub CLI:

```bash
gh secret set MYCELIUM_RELEASE_KEYSTORE_B64 < mycelium-release.jks.b64
gh secret set MYCELIUM_KEYSTORE_PASSWORD
gh secret set MYCELIUM_KEY_PASSWORD
gh secret set MYCELIUM_RAILWAY_DOMAIN
```

## Keystore alias

The workflow and helper default to the Bubblewrap signing alias `android`.
If your keystore uses a different alias, set `MYCELIUM_KEYSTORE_ALIAS` in the workflow or adjust the keystore before upload.

## What the workflow does

1. Decodes the keystore secret into `twa/android.keystore`.
2. Runs the Bubblewrap TWA forge in CI mode.
3. Writes `version_bump.txt` and `twa-keystore-fingerprint.txt`.
4. Uploads the `.apk` and `.aab` artifacts to the GitHub Release.
