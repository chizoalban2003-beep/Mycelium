# Changelog

All notable changes to PhysML / Mycelium are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.12.0] — 2026-04-14

### Added — Tier 3: Ecosystem (Stages 18–20)

* **Stage 18 — REST API microservice** (`physml/server.py`):
  `POST /train`, `POST /query`, `POST /feedback`, `GET /report` via FastAPI.
  Wrapped in a `Dockerfile` for one-command deployment.
* **Stage 19 — Federated learning** (`physml/federated.py`):
  `FederatedMyceliumAgent` coordinates multiple local `myco` nodes using
  FedAvg (average weight deltas); raw data never leaves each node.
* **Stage 20 — Packaging**:
  `pyproject.toml` with proper metadata, `[project.scripts]` (`physml` CLI),
  optional `server` extras, `CHANGELOG.md`, updated `README.md`.

---

## [0.11.0] — 2026-04-14

### Added — Tier 2: Smarter Learning (Stages 15–17)

* **Stage 15 — Contextual bandit** (`physml/bandit.py`):
  `ContextualBandit` replaces the adaptive-threshold heuristic with a
  logistic-regression policy trained online with REINFORCE-style rewards.
  Enabled via `myco(policy="bandit")`.
* **Stage 16 — Coreset batch active learning**:
  `PhysicsAgent.select_batch(X_pool, k)` and `MyceliumAgent.select_batch`
  use greedy coreset selection to pick `k` maximally diverse candidates,
  reducing oracle calls vs independent top-k entropy selection.
* **Stage 17 — Concept drift detection** (`physml/drift.py`):
  `DriftDetector` with Page-Hinkley and ADWIN algorithms.  Integrated into
  `PhysicsAgent` via `drift_detection=True`; drift triggers homeostasis
  reset and a burst of lower ask-threshold to re-explore the new distribution.

---

## [0.10.0] — 2026-04-14

### Added — Tier 1: Production-Readiness (Stages 12–14)

* **`myco` alias** — `from physml import myco` is identical to `MyceliumAgent`.
* **Stage 12 — CLI** (`physml/cli.py`):
  `physml fit`, `physml query`, `physml report`, `physml export` entry-points.
* **Stage 13 — Confidence calibration** (`physml/calibration.py`):
  Temperature scaling (Guo et al., 2017) fitted on a 20 % held-out split
  inside `MyceliumAgent.fit()`.  Adds `temperature_` attribute; exposed in
  `report()`.
* **Stage 14 — Evaluation harness** (`physml/evaluation.py`):
  `benchmark_agent(agent, X, y, oracle_budget)` simulates an oracle loop
  and returns `BenchmarkResult` with accuracy curve, ask-rate curve, and
  per-step history.

---

## [0.9.0] — 2026-04-14

### Added — Stages 8–11: Active Learning & MyceliumAgent

* **Stage 8** — `PhysicsAgent(query_strategy="entropy")` + `select_informative`.
* **Stage 9** — `MultiTaskPhysicsEngine`: shared trunk, per-task heads.
* **Stage 10** — `PhysicsAgent(policy="adaptive")`: adaptive threshold via
  rolling error window.
* **Stage 11** — `MyceliumAgent` flagship class combining all above.

---

## [0.7.0] — prior

* Stages 1–7: physics predictor, neural engine, continual learning, agent
  loop, streaming, session API.
