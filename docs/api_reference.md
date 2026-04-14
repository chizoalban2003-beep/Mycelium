# API Reference

## `myco` — flagship agent alias

```python
from physml import myco
agent = myco(
    uncertainty_threshold=0.3,
    query_strategy="entropy",   # "threshold" | "entropy" | "gp"
    policy="adaptive",          # "fixed" | "adaptive" | "bandit" | "ensemble"
    calibrate=True,
    drift_detection=False,
    drift_algorithm="page_hinkley",  # or "adwin"
)
```

### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Seed fit on labelled data |
| `observe(x)` → `AgentAction` | Predict; returns ask/tell/abstain action |
| `reward(x, y, cost=1.0)` | Provide true label (online update) |
| `partial_fit(X, y)` | Mini-batch update |
| `select_informative(X_pool)` → `int` | Index of most informative sample |
| `select_batch(X_pool, k)` → `list[int]` | Coreset greedy batch selection |
| `save(path)` | Pickle agent to disk |
| `load(path)` | Class method — restore from pickle |
| `report()` → `dict` | Activity summary |

---

## `PhysicsPredictor`

Low-level predictor with physics-inspired feature selection.

```python
from physml import PhysicsPredictor

p = PhysicsPredictor(
    backend="neural",          # "neural" | "tree"
    n_cycles=30,
    learning_rate=0.12,
    quantile_transform=True,
    poly_degree=2,
    n_estimators=5,
    residual_model=True,
)
p.fit(X_train, y_train)
result = p.predict(X_test)
print(result.test_accuracy)
```

---

## `benchmark_agent`

```python
from physml import benchmark_agent, BenchmarkResult

result: BenchmarkResult = benchmark_agent(
    agent,
    X, y,
    oracle_budget=80,
    seed_size=20,
    n_trials=5,
    seed=42,
)
print(result.summary())
result.save_csv("results.csv")
```

---

## `FederatedMyceliumAgent`

```python
from physml import FederatedMyceliumAgent

fed = FederatedMyceliumAgent()
fed.add_node("hospital_A", X_A, y_A)
fed.add_node("hospital_B", X_B, y_B)
fed.aggregate()          # FedAvg over MLP weight deltas
```

---

## `DriftDetector`

```python
from physml import DriftDetector

dd = DriftDetector(algorithm="adwin")
for residual in residuals:
    if dd.update(residual):
        print("Drift detected!")
```

---

## `ModelRegistry`

```python
from physml.registry import ModelRegistry

reg = ModelRegistry("runs.jsonl")
run_id = reg.log(agent, X, y, tags={"experiment": "v1"})
df = reg.list_runs()
reg.load_agent(run_id)
```

---

## REST API (`physml.server`)

Start: `uvicorn physml.server:app --reload`

| Endpoint | Method | Description |
|---|---|---|
| `/train` | POST | Train agent for a user |
| `/query` | POST | Get prediction |
| `/feedback` | POST | Provide true label |
| `/report` | GET | Session summary |
| `/metrics` | GET | Prometheus text metrics |

---

## CLI

```
physml fit    <csv>  --target <col> [--out agent.pkl]
physml query  <pkl>  --data <csv>
physml report <pkl>
physml export <pkl>  --out model.json
```
