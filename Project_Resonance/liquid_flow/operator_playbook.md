# Project Resonance Operator Playbook

Use this playbook to run the environment as a co-guardian, not as a one-off coder.

## 1) Observer pass (no mutations)

Prompt:

`@Codebase Create a State of the Fluid report. Where is the most dissonance right now? Where is the bedrock becoming too rigid?`

Write output to:

`Project_Resonance/liquid_flow/state_of_fluid_report_YYYYMMDD_HHMM.md`

Also update:

- `Project_Resonance/entropy_logs/entropy_log_YYYYMMDD.md`

## 2) Perturbation pass (energy spike)

Use `Project_Resonance/liquid_flow/agent_cycle_prompts.md` Step A.

Rules:

- Start in gaseous scratch form.
- Do not harden instantly.
- Record intent and expected impact before mutation.

## 3) Survival pass (friction metabolism)

Use Step B from `agent_cycle_prompts.md`.

Rules:

- Rank modules by friction and utility.
- Propose deletions/recycles with rationale.
- Prefer reducing complexity over adding knobs.

## 4) Sedimentation pass (hardening)

Use Step C from `agent_cycle_prompts.md`.

Rules:

- Harden only patterns with repeat survival evidence.
- Record provenance in `bedrock_manifest.json`.
- Include cycle count and evidence pointers.

## 5) Secrets hygiene

Maintain `Project_Resonance/secrets/subconscious_heuristics.json`:

- Add low-risk heuristics discovered autonomously.
- Track confidence, provenance, and rollback steps.
- Never store credentials, private tokens, or personal content.

## 6) Commit style

Use sedimentary commit messages:

- `Crystallized connection logic after Energy Spike #42`
- `Dissolved high-friction parser from Liquid Layer`
- `Hardened low-entropy routing constants into Bedrock`

