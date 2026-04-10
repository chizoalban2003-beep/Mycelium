# Liquid Flow

This layer holds active logic under experimentation.

## Purpose

- Host refactors that are currently under "survival pressure."
- Keep interfaces stable while internals evolve.
- Serve as the source for potential future bedrock hardening.

## Rules

- Keep modules composable and interface-driven.
- Prefer small adapter boundaries over global state.
- If complexity grows and utility does not, dissolve and recycle.

## Suggested process

1. Add or modify active modules in this layer.
2. Run survival checks from `prompts/survival_check.md`.
3. Log findings into `../entropy_logs/spike_journal.md`.
4. Harden recurrent patterns into `../bedrock/bedrock_manifest.json`.
