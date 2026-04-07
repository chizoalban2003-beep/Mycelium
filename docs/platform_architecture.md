# Mycelium Platform Architecture

This document defines the architectural structure of Mycelium as a standalone AI platform, not just an app.

The platform is designed around five cooperating layers:

1. experience surfaces
2. learning and transformation
3. trust and governance
4. timing and prediction
5. Hive network observability

The goal is to let the platform live on a user’s devices, learn from approved signals, explain its behavior, and grow into a personal assistant while keeping the user in control.

## Architectural structure

### 1) Experience layer

User-facing surfaces:

- [mycelium_app/web.py](../mycelium_app/web.py)
- [templates/device_shell.html](../templates/device_shell.html)
- [templates/login.html](../templates/login.html)
- [templates/demo.html](../templates/demo.html)
- [templates/projects.html](../templates/projects.html)
- [templates/hive_health.html](../templates/hive_health.html)
- [templates/knowledge.html](../templates/knowledge.html)

Purpose:

- onboard the user
- show the current state of the assistant
- expose learning summaries and memory controls
- provide the main daily-use shell

### 2) Learning and transformation layer

Signal and memory processing:

- [mycelium_app/stimulus.py](../mycelium_app/stimulus.py)
- [mycelium_app/routes/stimulus.py](../mycelium_app/routes/stimulus.py)
- [mycelium_app/routes/telemetry.py](../mycelium_app/routes/telemetry.py)
- [mycelium_app/telemetry_assistant.py](../mycelium_app/telemetry_assistant.py)
- [mycelium_app/feedback_ionizer.py](../mycelium_app/feedback_ionizer.py)
- [mycelium_app/metric_snapshot.py](../mycelium_app/metric_snapshot.py)
- [mycelium_app/causal_trace.py](../mycelium_app/causal_trace.py)
- [mycelium_app/self_reflection.py](../mycelium_app/self_reflection.py)
- [mycelium_app/routes/memory.py](../mycelium_app/routes/memory.py)

Purpose:

- capture consented digital signals
- normalize them into safe structured rows
- turn events into summaries, memory, and explanations
- preserve a visible learning trail

### 3) Trust and governance layer

Policy and control surfaces:

- [mycelium_app/parental_policy.py](../mycelium_app/parental_policy.py)
- [mycelium_app/privacy_membrane.py](../mycelium_app/privacy_membrane.py)
- [mycelium_app/security.py](../mycelium_app/security.py)
- [mycelium_app/routes/tasks.py](../mycelium_app/routes/tasks.py)
- [mycelium_app/routes/chat.py](../mycelium_app/routes/chat.py)

Purpose:

- gate actions behind explicit permission
- keep raw personal data local by default
- allow revocation, replay, and audit
- make the system reversible and inspectable

### 4) Timing and prediction layer

Adaptive intelligence:

- [mycelium_app/hybrid_predictor.py](../mycelium_app/hybrid_predictor.py)
- [mycelium_app/viscosity.py](../mycelium_app/viscosity.py)
- [mycelium_app/physics_predictor.py](../mycelium_app/physics_predictor.py)
- [mycelium_app/predictor_homeostasis.py](../mycelium_app/predictor_homeostasis.py)
- [mycelium_app/homeostasis.py](../mycelium_app/homeostasis.py)

Purpose:

- decide when the assistant should recommend, wait, or hand off
- score flow state from live device conditions
- adapt to context instead of pushing blindly

### 5) Platform runtime layer

Core runtime and data contracts:

- [mycelium_app/models.py](../mycelium_app/models.py)
- [mycelium_app/schemas.py](../mycelium_app/schemas.py)
- [mycelium_app/db.py](../mycelium_app/db.py)
- [mycelium_app/deps.py](../mycelium_app/deps.py)
- [mycelium_app/main.py](../mycelium_app/main.py)

Purpose:

- provide persistence and API contracts
- wire the app together
- expose the platform as a coherent runtime

## Data flow structure

The platform should behave like this:

1. a user or device generates a signal
2. the signal is captured through the learning layer
3. the signal is normalized into a structured record
4. the timing layer evaluates current state and flow
5. the governance layer checks permission and policy
6. the experience layer surfaces the result
7. the user approves, corrects, revokes, or continues
8. the platform learns from the outcome

That is the central operating loop of the platform.

## Trust boundaries

Mycelium should keep the following boundaries clear:

- local device data stays local unless consented
- Hive receives allowlisted, coarse, or filtered wisdom
- project scopes remain isolated
- actions require explicit policy control
- identity and persona remain user-configurable

These boundaries are what make the platform trustworthy enough to grow.

## Hive network observability structure

Hive is the platform’s real-time network view.

### What the operator should see

- connected devices and user nodes
- current trust status per node
- current viscosity / flow state
- recent signal movement
- handoff recommendations
- whether a node is gated, observing, or in flow

### What the visualization should do

- present the network as a clean graph or flow map
- make active devices visually distinct
- show signal direction and strength
- expose handoff candidates and confidence
- stay readable at a glance on desktop and mobile

### Suggested visual structure

- left panel: connected devices / nodes
- center panel: live assistant state, learning trail, and signal flow
- right panel: Hive relationships, trust state, and handoff suggestions
- bottom rail: recent events, memory updates, and governance actions

The result should feel like a calm control room for the user’s AI network.

## Product surface structure

The platform should expose the following surfaces:

- Home: what the assistant knows now
- Learning Trail: what it saw and how it transformed it
- Memory: what is retained and how it can be revoked
- Builder Copilot: focus, momentum, and next steps
- Hive: shared intelligence and node observability
- Child-Safe: stricter autonomy and parent-approved policy

## Platform summary

Mycelium is a device-native AI platform with a visible learning loop, consent-first governance, and a real-time Hive network view.

It is designed to live across user devices, learn from structured signals, and grow from a child-like learner into a personal assistant with trust-gated shared intelligence.
