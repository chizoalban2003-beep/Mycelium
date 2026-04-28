# API Reference

## Core Active Learning

### `myco` — flagship agent alias

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

| Method | Returns | Description |
|---|---|---|
| `fit(X, y)` | `self` | Seed fit on labelled data |
| `observe(x)` | `AgentAction` | Predict; returns ask/tell/abstain |
| `reward(x, y, cost=1.0)` | `self` | Provide true label (online update) |
| `partial_fit(X, y)` | `self` | Mini-batch update |
| `select_informative(X_pool)` | `int` | Index of most informative sample |
| `select_batch(X_pool, k)` | `list[int]` | Coreset greedy batch selection |
| `save(path)` | — | Pickle agent to disk |
| `load(path)` | `myco` | Class method — restore from pickle |
| `report()` | `dict` | Activity summary |

---

### `PhysicsPredictor`

```python
from physml import PhysicsPredictor

p = PhysicsPredictor(
    backend="neural",       # "neural" | "tree"
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

### `benchmark_agent`

```python
from physml import benchmark_agent, BenchmarkResult

result: BenchmarkResult = benchmark_agent(
    agent, X, y,
    oracle_budget=80,
    seed_size=20,
    n_trials=5,
    seed=42,
)
print(result.summary())
result.save_csv("results.csv")
```

---

### `FederatedMyceliumAgent`

```python
from physml import FederatedMyceliumAgent

fed = FederatedMyceliumAgent()
fed.add_node("hospital_A", X_A, y_A)
fed.add_node("hospital_B", X_B, y_B)
fed.aggregate()          # FedAvg over MLP weight deltas
```

---

### `DriftDetector`

```python
from physml import DriftDetector

dd = DriftDetector(algorithm="adwin")
for residual in residuals:
    if dd.update(residual):
        print("Drift detected!")
```

---

## Digital Companion

### `Companion`

```python
from physml import Companion

c = Companion(
    persist_dir="~/.mycelium",   # where state is saved
    api_key=None,                # Claude API key (or set ANTHROPIC_API_KEY)
)
c.start()
```

| Method | Description |
|---|---|
| `chat(message)` | Send a message and get a response |
| `ingest(source, topic)` | Ingest any file/URL/text into knowledge |
| `set_goal(description)` | Set and begin executing a multi-step goal |
| `start_screen_observer(interval, ...)` | Start background screen monitoring |
| `start_macro_recording(name)` | Begin recording a macro sequence |
| `stop_macro_recording()` | Stop and save the recording → `MacroSequence` |
| `suggest_next_action(context_app)` | Get imitation-learned action suggestions |
| `start_voice_interface()` | Launch voice loop (STT → LLM → TTS) |
| `personalise(**kwargs)` | Set user preferences |
| `status()` | Dict of all subsystem states |

---

### `MultiModalIngester`

Unified ingestion pipeline — routes any content into knowledge graph + vector memory.

```python
from physml import MultiModalIngester

ingester = MultiModalIngester(
    vector_memory=None,      # auto-created if None
    knowledge_graph=None,    # auto-created if None
    knowledge_extractor=None,
    user_profile=None,
    max_text_chars=50_000,
    deduplicate=True,
)

result = ingester.ingest("path/to/file.pdf", topic="research")
print(result.facts)       # extracted facts
print(result.memory_id)   # stored in vector memory
print(result.elapsed)

# Batch
results = ingester.ingest_many(["file1.txt", "file2.md"], topic="notes")

# Directory
ingester.ingest_directory("~/Documents/", topic="work", recursive=True)

print(ingester.summary())   # {"total": 23, "success": 22, "deduplicated": 1, ...}
```

**`IngestResult` fields:** `source`, `text`, `facts`, `memory_id`, `metadata`, `elapsed`, `success`, `error`

---

### `ScreenObserver`

```python
from physml import ScreenObserver

obs = ScreenObserver(
    interval=60.0,           # seconds between snapshots
    save_screenshots=False,
    llm_describe=True,       # use Claude Vision for descriptions
    ingester=None,           # auto-ingest snapshots
    vector_memory=None,
    on_snapshot=None,        # callback(ScreenSnapshot)
)

obs.start()                  # starts background daemon thread
obs.observe_once()           # take one snapshot immediately
obs.stop()

print(obs.focus_summary())   # {"VS Code": 145.2, "Chrome": 80.1}
print(obs.top_apps(5))       # top 5 apps by focus time
print(obs.recent_context(10))# last 10 ScreenSnapshot objects
```

**`ScreenSnapshot` fields:** `timestamp`, `app_name`, `window_title`, `description`, `screenshot_path`

---

### `MacroRecorder`

```python
from physml import MacroRecorder

recorder = MacroRecorder(
    skill_library=None,      # auto-save completed sequences
    save_dir="~/.mycelium/macros",
    min_steps=2,
    merge_typing=True,       # combine char-by-char keystrokes into TYPE_TEXT
)

recorder.start_recording("my_workflow")
# ... perform actions ...
seq = recorder.stop_recording()   # → MacroSequence

print(seq.summarise())
recorder.save_sequence(seq)
recorder.save_to_skill_library(seq)   # registers as reusable skill
```

**`MacroSequence` fields:** `name`, `steps`, `description`, `created_at`, `tags`, `duration`, `apps_used`

**`ActionType` constants:** `CLICK`, `DOUBLE_CLICK`, `RIGHT_CLICK`, `KEY_PRESS`, `KEY_RELEASE`, `TYPE_TEXT`, `SCROLL`, `DRAG`, `WINDOW_CHANGE`, `PAUSE`

---

### `ImitationLearner`

```python
from physml import ImitationLearner

learner = ImitationLearner(
    context_window=3,    # how many past steps to use as context
    min_sequences=1,     # minimum sequences to fit
)

learner.fit(sequences)   # list of MacroSequence

suggestions = learner.predict_next(
    context_steps=[step1, step2],
    context_app="VS Code",
    context_action="KEY_PRESS",
    top_k=3,
)
for s in suggestions:
    print(s.action_type, f"{s.confidence:.2%}")
```

**`ActionSuggestion` fields:** `action_type`, `confidence`, `x`, `y`, `text`, `app_name`

---

### `UserModel`

```python
from physml import UserModel

um = UserModel(
    user_profile=None,
    digital_soul=None,
    personalisation=None,
    vector_memory=None,
    knowledge_graph=None,
    screen_observer=None,
    macro_recorder=None,
    persist_dir="~/.mycelium/user_model",
)

# Fire any event
um.update({"type": "interaction", "role": "user", "content": "fix the bug"})
um.update({"type": "screen", "snapshot": snapshot})
um.update({"type": "goal_completed", "description": "deploy app", "steps": 4})

ctx = um.current_context()
# {"app": "VS Code", "mood": "focused", "top_topics": [...], "peak_hour": 14, ...}

prompt_snippet = um.inject_into_prompt()
patterns = um.behavioral_patterns()
```

---

### `SpecialistFederation`

```python
from physml import SpecialistFederation

fed = SpecialistFederation(
    specialists=None,         # default: Coder, Browser, Data, Scheduler, NLP, System
    knowledge_graph=None,
    vector_memory=None,
    comms=None,               # AgentComms bus
    persist_dir="~/.mycelium/federation",
)

fed.start()

result = fed.query(
    "How do I optimise this SQL query?",
    context={"app": "DBeaver", "topic": "data"},
)
print(result["specialist"])   # "Data"
print(result["response"])

# Broadcast knowledge to all specialists
fed.broadcast_fact("User prefers Python 3.12")

# Snapshot shared knowledge
print(fed.knowledge_snapshot())

# Which specialists are active
print(fed.list_specialists())
```

**Built-in specialists:** `Coder` (code + debugging), `Browser` (web + research), `Data` (SQL + analysis), `Scheduler` (calendar + reminders), `NLP` (text + documents), `System` (OS + files + shell)

---

## Infrastructure

### `ModelRegistry`

```python
from physml.registry import ModelRegistry

reg = ModelRegistry("runs.jsonl")
run_id = reg.log(agent, X, y, tags={"experiment": "v1"})
df = reg.list_runs()
reg.load_agent(run_id)
```

---

### REST API (`physml.server`)

Start: `uvicorn physml.server:app --reload`

| Endpoint | Method | Description |
|---|---|---|
| `/train` | POST | Train agent for a user |
| `/query` | POST | Get prediction |
| `/feedback` | POST | Provide true label |
| `/report` | GET | Session summary |
| `/metrics` | GET | Prometheus text metrics |
| `/chat` | POST | LLM conversation |
| `/ingest` | POST | Ingest document |
| `/mobile/chat` | POST | Mobile chat endpoint |
| `/mobile/ingest` | POST | Mobile ingest |
| `/mobile/context` | GET | Current user context |
| `/mobile/patterns` | GET | Behavioral patterns |
| `/mobile/status` | GET | All subsystem status |
| `/ext/page-visit` | POST | Browser ext page visit |
| `/ext/selection` | POST | Browser ext selection |
| `/ext/bookmark` | POST | Browser ext bookmark |
| `/ext/command` | POST | Browser ext command |
| `/pwa/` | GET | Mobile PWA app |

---

### CLI

```
physml fit       <csv>  --target <col> [--out agent.pkl]
physml query     <pkl>  --data <csv>
physml report    <pkl>
physml export    <pkl>  --out model.json
physml serve     [--host HOST] [--port PORT]
physml ingest    <source> [--topic TOPIC]
physml observe   [--interval SECS]
physml record    [--name NAME]
physml model     [--show]
physml federation [--query TEXT] [--context SPECIALIST]
```

---

### Health Check

```python
from physml.health import check

status = check()
# {"numpy": True, "sklearn": True, "anthropic": True, "whisper": False, ...}
```
