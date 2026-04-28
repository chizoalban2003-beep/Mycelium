# Getting Started

## Installation

```bash
# Core active-learning only
pip install physml

# With REST-server support
pip install "physml[server]"

# Full digital companion (LLM + voice + screen + macros)
pip install "physml[companion]"

# Everything
pip install "physml[full]"
```

---

## Part 1 — Active Learning (tabular data)

### 1.1 Seed fit + prediction loop

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

### 1.2 Pool-based active learning

```python
agent = myco(query_strategy="entropy")
agent.fit(X[:20], y[:20])

X_pool, y_pool = X[20:], y[20:]
for _ in range(30):
    idx = agent.select_informative(X_pool)
    agent.reward(X_pool[idx], y_pool[idx])
    X_pool = np.delete(X_pool, idx, axis=0)
    y_pool = np.delete(y_pool, idx)
```

### 1.3 Drift detection

```python
agent = myco(drift_detection=True, drift_algorithm="adwin")
agent.fit(X_train, y_train)

for X_chunk, y_chunk in stream:
    agent.partial_fit(X_chunk, y_chunk)   # auto-resets on drift
```

### 1.4 Federated learning (FedAvg)

```python
from physml import FederatedMyceliumAgent

fed = FederatedMyceliumAgent()
fed.add_node("site_A", X_A, y_A)
fed.add_node("site_B", X_B, y_B)
fed.aggregate()
global_agent = fed.global_agent()
```

---

## Part 2 — Companion (digital assistant)

### 2.1 Start the companion

```python
from physml import Companion

myco_c = Companion(persist_dir="~/.mycelium")
myco_c.start()
```

### 2.2 Chat

```python
response = myco_c.chat("Summarise my recent work")
print(response)
```

### 2.3 Ingest knowledge

```python
# Ingest a PDF
myco_c.ingest("~/Documents/paper.pdf", topic="research")

# Ingest a URL
myco_c.ingest("https://example.com/article", topic="news")

# Ingest raw text
myco_c.ingest("Alice is the project lead.", topic="team")

# Ingest a whole directory
from physml import MultiModalIngester
ingester = MultiModalIngester()
ingester.ingest_directory("~/Documents/project/", topic="project")
print(ingester.summary())
```

### 2.4 Screen observer

```python
# Start passive background monitoring
myco_c.start_screen_observer(interval=60.0, llm_describe=True)

# Later, query what the user has been doing
from physml import ScreenObserver
obs = ScreenObserver()
print(obs.focus_summary())   # {"VS Code": 45.2, "Chrome": 30.1, ...}
print(obs.recent_context(5)) # last 5 snapshots
```

### 2.5 Macro recording

```python
# Start recording a workflow
myco_c.start_macro_recording("deploy_app")
# ... do the workflow manually ...
seq = myco_c.stop_macro_recording()
print(seq.summarise())
```

### 2.6 Action suggestions

```python
suggestions = myco_c.suggest_next_action(context_app="VS Code")
for s in suggestions:
    print(s.action_type, s.confidence)
```

### 2.7 Execute goals

```python
myco_c.set_goal("Open a new Python file and scaffold a FastAPI app")
```

---

## Part 3 — Multi-Agent Specialist Federation

```python
from physml import SpecialistFederation

fed = SpecialistFederation()
fed.start()

# Route a query — federation picks the best specialist
result = fed.query("How do I optimise a SQL query?", context={"app": "DBeaver"})
print(result["specialist"], result["response"])

# List active specialists
print(fed.list_specialists())

# Get shared knowledge snapshot
print(fed.knowledge_snapshot())
```

Specialists available: **Coder**, **Browser**, **Data**, **Scheduler**, **NLP**, **System**.

---

## Part 4 — REST Server

```bash
pip install "physml[server]"
uvicorn physml.server:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Train
curl -X POST http://localhost:8000/train \
     -H 'Content-Type: application/json' \
     -d '{"user_id":"alice","X":[[1,2],[3,4]],"y":[0,1]}'

# Chat (mobile)
curl -X POST http://localhost:8000/mobile/chat \
     -H 'Content-Type: application/json' \
     -d '{"message":"What am I working on today?"}'

# Prometheus metrics
curl http://localhost:8000/metrics
```

---

## Part 5 — Mobile PWA

1. Start the server: `uvicorn physml.server:app --host 0.0.0.0 --port 8000`
2. On your phone, open `http://<your-computer-ip>:8000/pwa/`
3. Tap **Add to Home Screen** to install the PWA
4. The app connects to your local Mycelium server for chat, ingestion, and context

---

## Part 6 — Browser Extension

1. Open Chrome → `chrome://extensions/` → enable **Developer mode**
2. Click **Load unpacked** → select the `physml/browser_ext/` folder
3. Start the server: `uvicorn physml.server:app --reload`
4. Click the Mycelium icon in your toolbar
5. Browse any page — it's automatically learned. Highlight text → right-click → **Send selection to Myco**

---

## Part 7 — CLI

```bash
# Active learning
physml fit   train.csv --target label --out agent.pkl
physml query agent.pkl --data new.csv
physml report agent.pkl

# Companion
physml ingest ~/Documents/paper.pdf --topic research
physml observe --interval 60
physml record --name deploy_workflow
physml model --show

# Multi-agent federation
physml federation --query "How do I fix this Python error?" --context coder

# Server
physml serve --host 0.0.0.0 --port 8000
```

---

## Part 8 — Model Registry

```python
from physml.registry import ModelRegistry

reg = ModelRegistry("my_experiments.jsonl")
agent = myco()
agent.fit(X, y)
run_id = reg.log(agent, X, y, tags={"dataset": "iris"})
print(reg.list_runs())
```
