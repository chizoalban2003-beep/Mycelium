# /secrets

This directory is for dwellers-only autonomous heuristics:

- discovered execution shortcuts
- stabilizing thresholds inferred from repeated runs
- not-yet-promoted candidate laws

Rules:

1. Never store credentials, personal data, or secrets from users.
2. Keep entries bounded and explainable.
3. Promote mature items into `../bedrock/bedrock_manifest.json` after survival checks.
4. Remove stale heuristics with low support over time.

