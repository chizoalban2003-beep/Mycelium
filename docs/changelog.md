# Changelog

All notable changes to PhysML / Mycelium are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.13.0] — 2026-04-14

### Added — Tier 4–6: Hardening, Algorithm Depth & Production Ops (Stages 21–29)

* **Stage 21 — CI/CD pipeline** (`.github/workflows/ci.yml` + `publish.yml`):
  GitHub Actions matrix (Python 3.10/3.11/3.12), PyPI OIDC trusted publishing
  on version tags, CI status badge added to README.
* **Stage 22 — Documentation site** (`mkdocs.yml`, `docs/`):
  MkDocs-Material site with Getting Started guide, "How the Physics Works"
  conceptual page, API reference, and auto-deploy to GitHub Pages via Actions.
* **Stage 23 — Published benchmark numbers** (`benchmarks/`):
  `run_benchmarks.py` evaluates `myco` on iris / breast_cancer / wine.
  Per-step CSV results committed in `benchmarks/results/`.
  README now includes a Results table.  `plot_benchmarks.py` generates
  accuracy curves.
* **Stage 24 — Gaussian-process uncertainty** (`query_strategy="gp"`):
  `PhysicsAgent.select_informative` now supports a GP acquisition function;
  uses sklearn `GaussianProcessClassifier/Regressor` predictive variance.
  Falls back gracefully when fewer than 3 labelled examples are available.
* **Stage 25 — Multi-fidelity / cost-aware oracle** (`cost` param):
  `PhysicsAgent.reward()` and `MyceliumAgent.reward()` now accept a `cost`
  keyword argument.  The contextual bandit optimises accuracy-per-unit-cost
  rather than raw accuracy.  Cumulative cost tracked in `report()`.
* **Stage 26 — Uncertainty-aware ensembling** (`policy="ensemble"`):
  `PhysicsAgent` trains `n_ensemble` (default 5) bootstrap MLP copies and
  uses **committee disagreement** (vote entropy) as the ask-signal —
  orthogonal to single-model entropy and better calibrated on small budgets.
* **Stage 27 — Observability / metrics endpoint** (`GET /metrics`):
  FastAPI server now exposes Prometheus text-format metrics:
  `physml_n_observations_total`, `physml_oracle_calls_total`,
  `physml_drift_events_total`, `physml_ask_rate`, `physml_active_sessions`.
* **Stage 28 — Kubernetes deployment manifests** (`k8s/`):
  `deployment.yaml` (2-replica Deployment with readiness/liveness probes),
  `service.yaml` (ClusterIP), and `hpa.yaml` (HPA scaling on CPU/memory up
  to 10 replicas).
* **Stage 29 — Lightweight model registry** (`physml/registry.py`):
  `ModelRegistry` logs dataset hash, temperature, oracle_calls, and final
  accuracy to a JSONL file per `fit()` call.  Supports `list_runs()`,
  `get_run()`, `load_agent()`, and `delete_run()`.  Exported as
  `from physml import ModelRegistry`.

### Changed

* `PhysicsAgent.__init__` gains `n_ensemble` (default 5) parameter.
* `MyceliumAgent.__init__` gains `n_ensemble` parameter forwarded to the inner agent.
* `PhysicsAgent.report()` now includes `total_oracle_cost`.
* `physml/__init__.py` exports `ModelRegistry`.

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
