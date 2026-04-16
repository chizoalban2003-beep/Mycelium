# PhysML / Mycelium

> **Physics-Inspired Autonomous Machine Learning**

[![CI](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml/badge.svg)](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/physml.svg)](https://pypi.org/project/physml/)
[![Python](https://img.shields.io/pypi/pyversions/physml.svg)](https://pypi.org/project/physml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PhysML is a Python library for **autonomous active-learning** on tabular data.
Its flagship class — `myco` — trains itself, asks for labels only when
uncertain, adapts in real-time, and supports streaming, drift detection,
federated learning, and more.

## Quick Start

```python
from physml import myco
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5))
y = (X[:, 0] > 0).astype(int)

agent = myco()
agent.fit(X[:50], y[:50])

for x_new, y_true in zip(X[50:], y[50:]):
    action = agent.observe(x_new)
    if action.action == "ask":
        agent.reward(x_new, y_true)

print(agent.report())
```

## Feature Overview

| Feature | Class / API |
|---|---|
| Active learning (entropy / GP / committee) | `myco(query_strategy=...)` |
| Adaptive ask-threshold | `myco(policy="adaptive")` |
| Contextual bandit policy | `myco(policy="bandit")` |
| Ensemble / query-by-committee | `myco(policy="ensemble")` |
| Temperature calibration | `myco(calibrate=True)` |
| Concept-drift detection | `myco(drift_detection=True)` |
| Coreset batch selection | `agent.select_batch(X_pool, k=10)` |
| Federated learning | `FederatedMyceliumAgent` |
| REST microservice | `physml.server` (FastAPI) |
| Prometheus metrics | `GET /metrics` |
| Model registry | `physml.registry` |
| CLI | `physml fit / query / report` |

See [Getting Started](getting_started.md) for a step-by-step walkthrough.
