# Getting Started

## Installation

```bash
pip install physml
# With REST-server support:
pip install "physml[server]"
```

## 1 — Seed fit + prediction loop

```python
from physml import myco
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 8))
y = (X[:, 0] + X[:, 2] > 0).astype(int)

agent = myco()
agent.fit(X[:60], y[:60])   # seed with 60 labelled examples

oracle_calls = 0
for x_new, y_true in zip(X[60:], y[60:]):
    action = agent.observe(x_new)
    if action.action == "ask":
        agent.reward(x_new, y_true)
        oracle_calls += 1

print(agent.report())
print(f"Oracle called {oracle_calls} / {len(X) - 60} times")
```

## 2 — Active learning with a pool

```python
agent = myco(query_strategy="entropy")
agent.fit(X[:20], y[:20])

X_pool = X[20:]
y_pool = y[20:]

for _ in range(30):
    idx = agent.select_informative(X_pool)
    agent.reward(X_pool[idx], y_pool[idx])
    X_pool = np.delete(X_pool, idx, axis=0)
    y_pool = np.delete(y_pool, idx)
```

## 3 — Gaussian-process uncertainty

```python
agent = myco(query_strategy="gp")   # GP variance as acquisition function
agent.fit(X[:20], y[:20])
idx = agent.select_informative(X_pool)
```

## 4 — Ensemble / query-by-committee

```python
agent = myco(policy="ensemble")   # 5 bootstrap MLPs, disagreement as signal
agent.fit(X[:40], y[:40])

action = agent.observe(x_new)
print(action.confidence)          # committee disagreement, not single-model entropy
```

## 5 — Cost-aware oracle

```python
agent = myco(policy="bandit")
agent.fit(X[:40], y[:40])

# Reward bandit with a cost weight (cheaper samples annotated more freely)
agent.reward(x_new, y_true, cost=0.1)
agent.reward(x_expensive, y_expensive, cost=5.0)
```

## 6 — Drift detection

```python
agent = myco(drift_detection=True, drift_algorithm="adwin")
agent.fit(X_train, y_train)

for X_chunk, y_chunk in stream:
    agent.partial_fit(X_chunk, y_chunk)   # resets if drift detected
```

## 7 — Benchmarking

```python
from physml import benchmark_agent, myco

result = benchmark_agent(myco(), X, y, oracle_budget=80)
print(result.summary())
```

## 8 — Model registry

```python
from physml.registry import ModelRegistry

reg = ModelRegistry("my_experiments.jsonl")
agent = myco()
agent.fit(X, y)
run_id = reg.log(agent, X, y, tags={"dataset": "iris"})
print(reg.list_runs())
```

## 9 — REST server

```bash
pip install "physml[server]"
uvicorn physml.server:app --reload
```

```bash
curl -X POST http://localhost:8000/train \
     -H 'Content-Type: application/json' \
     -d '{"user_id": "alice", "X": [[1,2],[3,4]], "y": [0, 1]}'

curl http://localhost:8000/metrics     # Prometheus text
```

## 10 — CLI

```bash
physml fit   train.csv --target label --out agent.pkl
physml query agent.pkl --data new.csv
physml report agent.pkl
```
