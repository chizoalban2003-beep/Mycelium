# Myco AI Platform Blueprint

Myco is a consent-first AI platform built to feel like a living assistant stack: it starts in a gentle guided mode, learns only from approved signals, explains what it learned, and gradually earns the right to act more autonomously.

The product is intentionally different from a generic chatbot. It is designed as a layered system with a visible learning loop, a trust membrane, and a shared intelligence layer called Hive that only activates through consent and policy.

## Startup identity

Myco is best understood as a personal AI operating platform for founders, developers, operators, and power users.

Core promise:

- the user stays in control
- the assistant grows from approved data
- every important action is explainable
- autonomy is earned, not assumed
- shared intelligence is coarse, gated, and reversible

## Product format

The product follows the same format you described for the assistant itself:

1. Start small and visible.
2. Learn from real user signals.
3. Summarize what changed.
4. Request trust before taking action.
5. Expand into a personal copilot.
6. Share only allowlisted wisdom through Hive.

That makes Myco a platform that behaves like a growing system instead of a static app.

## Learning format

The learning loop is the heart of the startup:

1. capture consented signals
2. normalize them into structured events
3. score and predict with physics-inspired timing and viscosity
4. write memory and summaries
5. surface the result in the UI
6. let the user correct, revoke, or approve the outcome

This makes the assistant transparent and trainable without hiding the reasoning process.

## Platform layers

The repository already contains the major layers a real AI platform needs:

- experience surfaces for daily use
- learning pipelines for signal capture and transformation
- trust and governance controls for consent and revocation
- timing and prediction engines for adaptive assistance
- identity and messaging surfaces for the assistant persona
- data and API foundations for persistence and runtime behavior
- operations and rollout scripts for shipping and evaluation

These layers are not theoretical; they already map to the files in the workspace.

For the full architectural structure and Hive visualization model, see [docs/platform_architecture.md](docs/platform_architecture.md).

## Device-native learning architecture

Myco is meant to live on the user’s devices as a companion assistant that watches approved digital signals and turns them into structured learning rows.

The intended path is:

1. child device observes app, chat, action, and telemetry signals
2. signal data is normalized into safe, flattened tabular records
3. [mycelium_app/physics_predictor.py](../mycelium_app/physics_predictor.py) and the viscosity layer score what should happen next
4. memory, summaries, and recommendations are produced from those rows
5. the user sees exactly what the assistant learned and why
6. over time, the assistant becomes a personal copilot instead of a passive logger

This is the bridge from “device companion” to “personal assistant” to “platform product.”

## Hive observability

Hive should feel like a living network map of approved connected users and devices, not a hidden sync layer.

The visual design goal is:

- show nodes, trust state, and signal flow in real time
- make movement across devices easy to understand
- highlight viscosity, recommendations, and handoff readiness
- keep the network readable at a glance for the brain/operator

The codebase already points to this direction through live viscosity snapshots, handoff state, and the Hive Health surface.

The product vision is a visually appealing network dashboard where you can observe the child devices, their current flow state, and the shared Hive connections as they evolve.

## Workspace map

The current repository already contains the startup stack:

### Experience layer

- [mycelium_app/web.py](../mycelium_app/web.py)
- [templates/device_shell.html](../templates/device_shell.html)
- [templates/login.html](../templates/login.html)
- [templates/demo.html](../templates/demo.html)
- [templates/projects.html](../templates/projects.html)
- [templates/hive_health.html](../templates/hive_health.html)
- [templates/knowledge.html](../templates/knowledge.html)

These files define the daily product surfaces and the first-contact story.

### Learning layer

- [mycelium_app/stimulus.py](../mycelium_app/stimulus.py)
- [mycelium_app/routes/stimulus.py](../mycelium_app/routes/stimulus.py)
- [mycelium_app/routes/telemetry.py](../mycelium_app/routes/telemetry.py)
- [mycelium_app/telemetry_assistant.py](../mycelium_app/telemetry_assistant.py)
- [mycelium_app/feedback_ionizer.py](../mycelium_app/feedback_ionizer.py)
- [mycelium_app/metric_snapshot.py](../mycelium_app/metric_snapshot.py)
- [mycelium_app/causal_trace.py](../mycelium_app/causal_trace.py)
- [mycelium_app/self_reflection.py](../mycelium_app/self_reflection.py)

These modules turn user activity into visible learning and daily summaries.

### Trust and governance layer

- [mycelium_app/parental_policy.py](../mycelium_app/parental_policy.py)
- [mycelium_app/privacy_membrane.py](../mycelium_app/privacy_membrane.py)
- [mycelium_app/security.py](../mycelium_app/security.py)
- [mycelium_app/routes/tasks.py](../mycelium_app/routes/tasks.py)
- [mycelium_app/routes/chat.py](../mycelium_app/routes/chat.py)
- [mycelium_app/routes/memory.py](../mycelium_app/routes/memory.py)

These files make the assistant consent-first, auditable, and reversible.

### Intelligence and timing layer

- [mycelium_app/hybrid_predictor.py](../mycelium_app/hybrid_predictor.py)
- [mycelium_app/viscosity.py](../mycelium_app/viscosity.py)
- [mycelium_app/physics_predictor.py](../mycelium_app/physics_predictor.py)
- [mycelium_app/predictor_homeostasis.py](../mycelium_app/predictor_homeostasis.py)
- [mycelium_app/homeostasis.py](../mycelium_app/homeostasis.py)

These files decide when to recommend, wait, back off, or hand off.

### Identity and messaging layer

- [mycelium_app/assistant_profile.py](../mycelium_app/assistant_profile.py)
- [mycelium_app/identity_presentation.py](../mycelium_app/identity_presentation.py)
- [mycelium_app/messaging_bridge.py](../mycelium_app/messaging_bridge.py)

These modules shape persona, delivery, and external messaging.

### Data and API layer

- [mycelium_app/models.py](../mycelium_app/models.py)
- [mycelium_app/schemas.py](../mycelium_app/schemas.py)
- [mycelium_app/db.py](../mycelium_app/db.py)
- [mycelium_app/deps.py](../mycelium_app/deps.py)
- [mycelium_app/main.py](../mycelium_app/main.py)

These files form the runtime backbone of the startup.

### Productization and operations layer

- [docs/saas_deployment.md](../docs/saas_deployment.md)
- [docs/device_product_spec.md](../docs/device_product_spec.md)
- [scripts/smoke_autonomy_handoff_flow.py](../scripts/smoke_autonomy_handoff_flow.py)
- [scripts/public_alpha_checklist.py](../scripts/public_alpha_checklist.py)
- [scripts/check_handoff_slo.py](../scripts/check_handoff_slo.py)
- [scripts/verify_deploy_version.py](../scripts/verify_deploy_version.py)
- [scripts/global_audit_report.py](../scripts/global_audit_report.py)

These files turn the idea into something shippable and measurable.

## Startup narrative

Myco begins as a guided assistant that helps a user work, reflect, and stay organized.

As the user allows more signals, the assistant becomes better at summarizing context, proposing next steps, and protecting focus. Once trust is earned, it can coordinate actions across devices. Once Hive is enabled, it can share only filtered, consented wisdom with other approved users or nodes.

That creates a startup narrative with a clear progression:

- child-first onboarding
- visible learning trail
- personal assistant behavior
- trust-gated autonomy
- shared intelligence through Hive

## What the platform is becoming

Myco is becoming a visible AI platform for people who want:

- a personal assistant that learns from them
- a system they can inspect and revoke
- a device-native product instead of a hidden cloud brain
- a path from solo use to trusted shared intelligence
- a startup that can be explained in one sentence and built in layers

## One-sentence description

Myco is a consent-first AI platform that learns visibly from approved user signals, grows into a personal copilot, and expands into a trust-gated Hive of shared intelligence.
