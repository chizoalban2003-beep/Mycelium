# Mycelium / PhysML — Physics-Inspired Autonomous Machine Learning

> **The headline class is `myco`** — an autonomous active-learning agent that
> trains itself, asks for labels only when uncertain, and adapts in real-time.

```python
from physml import myco
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5))
y = (X[:, 0] > 0).astype(int)

agent = myco()
agent.fit(X[:50], y[:50])          # seed with 50 labelled samples

for x_new, y_true in zip(X[50:], y[50:]):
    action = agent.observe(x_new)
    if action.action == "ask":     # only asks when uncertain
        agent.reward(x_new, y_true)

print(agent.report())
```

PhysML frames supervised learning as a **gel electrophoresis simulation**.
Features are treated as charged particles migrating through a viscous medium;
their "charge" is their statistical association with the target, and
"viscosity" is modulated by collinearity, distribution shape, and an iterative
PCR-style amplification step.

## How It Works

```
Raw tabular data
      │
      ▼
  Cleaning & imputation  (rolling median, MAD, winsorize, …)
      │
      ▼
  Feature scoring        (Pearson/Spearman/Cramér-V / KL-divergence)
      │
      ▼
  Electrophoresis        (n_cycles × learning_rate updates, viscosity field)
      │
      ▼
  Bonding & complexes    (multicollinearity suppression)
      │
      ▼
  PCR amplification      (boost statistically significant features)
      │
      ▼
  PredictionResult       (test accuracy/R², feature weights, diagnostics)
```

## Install

```bash
pip install numpy pandas scipy scikit-learn
# optional: richer outlier cleaning
pip install feature-engine
# optional: CLI, REST API server
pip install fastapi uvicorn
```

## `myco` — Autonomous Agent Quick-Start

| Feature | How to use |
|---|---|
| Active learning (entropy) | `myco(query_strategy="entropy")` |
| Adaptive threshold | `myco(policy="adaptive")` |
| Confidence calibration | `myco(calibrate=True)` (default) |
| Contextual bandit policy | `myco(policy="bandit")` |
| Coreset batch queries | `agent.select_batch(X_pool, k=10)` |
| Drift detection | `myco(drift_detection=True)` |
| Multi-task | `myco(task_id="task_A")` |
| Save / load | `agent.save("agent.pkl")` / `myco.load("agent.pkl")` |
| Evaluation harness | `from physml.evaluation import benchmark_agent` |
| Federated learning | `from physml.federated import FederatedMyceliumAgent` |
| REST API | `uvicorn physml.server:app` |
| CLI | `physml fit train.csv --target y --out agent.pkl` |


```

## Quick Start

### Convenience classes (recommended)

```python
from physml import PhysicsRegressor, PhysicsClassifier

# Regression — quantile_transform + ridge residual model enabled by default
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
reg = PhysicsRegressor()
reg.fit(X, y)
print(reg.score(X, y))   # R²

# Classification — quantile_transform + logistic residual model enabled by default
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
clf = PhysicsClassifier()
clf.fit(X, y)
print(clf.score(X, y))   # accuracy
```

### scikit-learn API (base class)

```python
from physml import PhysicsPredictor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine(as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.25, random_state=42, stratify=wine.target
)

clf = PhysicsPredictor(plane="liquid", n_cycles=20)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
```

### Low-level functional API

```python
import pandas as pd
from physml import run_physics_prediction, PhysicsPlane

df = pd.read_csv("my_data.csv")
result = run_physics_prediction(
    df,
    target_col="price",          # or any classification column
    plane=PhysicsPlane.solid,
    n_cycles=30,
    return_predictions=True,
)
print(f"R²={result.metrics.rmse:.4f}  features={len(result.weights)}")
```

### Explicit train/test control

```python
import numpy as np

n = len(df)
mask = np.zeros(n, dtype=bool)
mask[:int(0.8*n)] = True           # first 80 % → train

result = run_physics_prediction(
    df,
    target_col="label",
    explicit_train_mask=mask,      # bypass random split
    n_cycles=25,
    return_predictions=True,
)
```

## Benchmark / Evaluation

Run the comprehensive benchmark to compare PhysML against RF, GBT, MLP, KNN, SVM, and more:

```bash
python evaluate.py                    # all tasks (classification + regression + agent)
python evaluate.py --tasks classification
python evaluate.py --tasks regression
python evaluate.py --tasks agent      # agent streaming / online-learning benchmark
python evaluate.py --quick            # faster run with fewer cycles
python evaluate.py --output results.json
```

Baselines included:

- **Random Forest** (RF)
- **Extra Trees** (ET)
- **Gradient Boosting** (GB)
- **Histogram Gradient Boosting** (HGB)
- **Neural Network** (MLP)
- **K-Nearest Neighbours** (KNN)
- **SVM / SVR**
- **Logistic Regression / Ridge**
- **AdaBoost**

Datasets: iris, breast_cancer, wine (classification); diabetes, california_housing (regression).

## Autonomous Agent Loop

PhysML includes a full autonomous-agent stack layered on top of the physics predictor:

```python
from physml import PhysicsPredictor
from physml.agent import PhysicsAgent

predictor = PhysicsPredictor(backend="neural", n_cycles=20)
predictor.fit(X_seed, y_seed)

agent = PhysicsAgent(predictor, uncertainty_threshold=0.35)

for X_new in stream_of_samples:
    action = agent.observe(X_new)         # "predict" | "ask" | "abstain"
    if action.action == "ask":
        y_true = oracle(X_new)            # request human label
        agent.reward(X_new, y_true)       # partial_fit with EWC + replay
    else:
        use_prediction(action.prediction)
```

### Session API (production deployment)

```python
from physml.agent_api import PhysicsAgentSession

session = PhysicsAgentSession(user_id="alice")
session.train(X_seed, y_seed)            # initial fit
result = session.query(X_new)            # {"prediction": ..., "action": "predict"|"ask", ...}
session.feedback(X_new, y_true)          # trigger partial_fit
session.save()                           # persist to ~/.physml_agents/alice.pkl

session2 = PhysicsAgentSession.load("alice")  # restore
```

### Autonomy Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | MLP backbone (256 → 128, sklearn) |
| 2 | ✅ | Feature-attention block (electrophoresis metaphor) |
| 3 | ✅ | Continual learning: `partial_fit` + EWC + replay buffer |
| 4 | ✅ | Agent loop: `PhysicsAgent.observe()` / `.reward()` |
| 5 | ✅ | `DataStream` mini-batch streaming |
| 6 | ✅ | Save / load + curriculum pretraining (`pretrain`) |
| 7 | ✅ | `PhysicsAgentSession` stateful per-user deployment |
| 8 | 🔲 | Active learning: entropy-based query selection |
| 9 | 🔲 | Multi-task head: shared representation across datasets |
| 10 | 🔲 | Reward shaping: RL policy replacing the fixed threshold |

## Naming Conventions

PhysML follows the **scikit-learn naming convention**:

| Class | Role |
|-------|------|
| `PhysicsPredictor` | Base estimator (physics backend, all options) |
| `PhysicsRegressor` | `PhysicsPredictor` with regression defaults (`quantile_transform=True`, `residual_model="ridge"`) |
| `PhysicsClassifier` | `PhysicsPredictor` with classification defaults (`quantile_transform=True`, `residual_model="logistic"`) |
| `NeuralPhysicsEngine` | Internal neural backend (MLP + attention) |
| `PhysicsAgent` | Autonomous observe/reward/adapt loop |
| `PhysicsAgentSession` | Stateful per-user production wrapper |
| `DataStream` | Mini-batch streaming helper |
| `AgentAction` | Return value from `PhysicsAgent.observe()` |

Rule of thumb: `Physics<Algorithm>` for the core engine classes, `<Algorithm>Action` / `<Algorithm>Session` / `<Algorithm>Stream` for agent utilities.

## Package Structure

```
physml/
  __init__.py        Public API exports
  predictor.py       Core physics engine
  estimator.py       PhysicsPredictor / PhysicsRegressor / PhysicsClassifier
  neural_engine.py   NeuralPhysicsEngine (Stage 1–3)
  agent.py           PhysicsAgent + DataStream (Stage 4–5)
  agent_api.py       PhysicsAgentSession (Stage 7)
evaluate.py          Stand-alone benchmark script
tests/
  test_predictor.py  Physics engine unit tests
  test_estimator.py  Estimator / sklearn compatibility tests
  test_neural_engine.py  Neural engine tests
  test_agent.py      Agent loop, streaming, session tests
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `plane` | `"liquid"` | Medium preset: `solid` / `liquid` / `gas` |
| `n_cycles` | 30 | Number of electrophoresis iterations |
| `cycle_learning_rate` | 0.18 | Per-cycle charge update rate |
| `cascade_enabled` | `True` | Multicollinearity complex suppression |
| `pcr_enabled` | `False` | PCR amplification of strong features |
| `enable_isotopes` | `True` | Auto-generate interaction features |
| `explicit_train_mask` | `None` | Override random split with boolean array |
| `quantile_transform` | `False` (`True` in subclasses) | Rank-normalise numeric features |
| `residual_model` | `None` (`"ridge"` / `"logistic"` in subclasses) | Second-stage residual corrector |
| `backend` | `"physics"` | `"physics"` or `"neural"` (MLP + attention) |

## Running Tests

```bash
python -m pytest tests/ -q
```

## License

MIT
