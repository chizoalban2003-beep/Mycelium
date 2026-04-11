## Agent Metabolism Layer

Dynamic, mutable runtime logic for thermodynamic adaptation.

Artifacts in this layer are expected to change frequently and are eligible for:

- perturbation (execution/modification),
- selection (promotion or recycling),
- mutation (candidate generation from high-entropy noise).

## Automation helpers

- Continuous cloud cycle runner:
  `python3 scripts/run_thermal_cycle_forever.py --interval-seconds 86400`
- Worktree mutation controller (zero-downtime sedimentation):
  `python3 scripts/worktree_mutation_controller.py --dry-run`

## Operational recommendations

- Use Cloud Handoff (`&`) for long-run thermal cycles so state persists when local sessions close.
- Run coordinated parallel agents for friction:
  - `Lava` lane for mutation generation in this folder.
  - `Frost` lane for verification and hardening checks before sedimentation.
- Persist condensed learnings in `resonance_memory.json` (do not store raw chat transcripts).
