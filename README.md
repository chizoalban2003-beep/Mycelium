# 🍄 Mycelium — Local Autonomous AI Companion

[![CI](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml/badge.svg)](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/physml.svg)](https://pypi.org/project/physml/)
[![Python](https://img.shields.io/pypi/pyversions/physml.svg)](https://pypi.org/project/physml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An AI that lives on your device, learns you over time, and acts on your behalf — autonomously.**  
> No cloud. No data leaving your machine. No subscription.

---

## What is Mycelium?

Mycelium is a **137-stage autonomous digital companion** built in Python. It combines a physics-inspired ML engine with a full agent stack: voice, memory, planning, tool use, browser control, screen automation, and an autonomous goal-execution loop — all running locally.

Like the mycelium fungus that silently connects a forest, Myco works in the background — learning your patterns, watching for new data, executing multi-step goals, and notifying you when something needs attention.

---

## What it can do right now

### Learn and predict from your data
```python
from physml.companion import MyceliumCompanion

myco = MyceliumCompanion(name="Myco")
myco.start()

# Train from a CSV — model persists across restarts
myco.chat("train on sales.csv")

# Predict
myco.chat("predict 1200 45 3.2")

# Correct a wrong prediction — model updates immediately
myco.chat("that's wrong, the answer is 1850")
```

### Have a conversation — it remembers you
```python
myco.chat("My name is Alex and I work as a data scientist in London.")
myco.chat("I love Python.")

# Facts are auto-extracted and injected into every future LLM prompt
print(myco.knowledge_extractor.status())
# {'facts_stored': 3, ...}

# Manual personalisation
myco.personalise("verbosity", "concise")
myco.personalise("name", "Alex")
```

### Give it a goal — it works autonomously
```python
# Queue a multi-step goal and let it run in the background
goal_id = myco.goal_engine.add_goal(
    "Read quarterly_report.csv, train a model on it, then notify me of the results"
)
myco.goal_engine.start_loop()

# Check progress any time
print(myco.goals())
# Goals (1):
#   [completed] a1b2c3d4: Read quarterly_report.csv ... [3/3 steps] (12s)
```

### Watch directories for new data
```python
# Auto-train whenever a new CSV appears in ~/Downloads
myco.personalise("watch_dirs", ["~/Downloads"])
```

### Browse the web
```python
text = myco.browse("https://example.com/report")
```

### Plan complex tasks
```python
print(myco.plan("Launch a new product in three markets"))
# Plan for: Launch a new product...
#   1. Research target markets and competitors
#   2. Define pricing strategy for each market
#   3. Prepare localised marketing materials
#   ...
```

---

## Installation

```bash
# Core (ML + companion, no optional deps)
pip install physml

# Full companion (voice, browser, screen, notifications, OCR, file watcher)
pip install "physml[companion]"

# Everything
pip install "physml[full]"
```

**Python 3.10+ required.** No internet connection required after install.

Optional extras:

| Extra | Unlocks |
|---|---|
| `llm` | Claude API (Anthropic) for intelligent chat and LLM-driven planning |
| `voice` | Whisper speech-to-text + wake word |
| `vector` | Semantic memory (sentence-transformers) |
| `browser` | Playwright browser automation |
| `screen` | pyautogui + mss screen control |
| `notify` | plyer desktop notifications |
| `ocr` | pytesseract image text extraction |
| `watcher` | watchdog file system monitoring |
| `companion` | All of the above bundled |

---

## Quick start — the ML core

```python
from physml import PhysicsRegressor, PhysicsClassifier

# Regression
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
reg = PhysicsRegressor()
reg.fit(X, y)
print(reg.score(X, y))   # R²

# Classification
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
clf = PhysicsClassifier()
clf.fit(X, y)
print(clf.score(X, y))   # accuracy
```

---

## The GoalEngine — autonomous task execution

`GoalEngine` (Stage 137) is the autonomous loop that turns Myco from a chatbot into an agent. Give it a natural-language goal; it decomposes it, dispatches each step to the right tool, retries failures, and notifies you on completion.

```python
from physml.goal_engine import GoalEngine
from physml.task_decomposer import TaskDecomposer
from physml.notifier import Notifier

engine = GoalEngine(
    task_decomposer=TaskDecomposer(),
    notifier=Notifier(),
    state_dir="~/.mycelium/goals",
    max_retries=2,
)

# Add a goal — persisted to disk immediately
gid = engine.add_goal("Read report.pdf and extract key financial figures")

# Run now (synchronous)
record = engine.run_now(gid)
print(record.status)      # GoalStatus.COMPLETED
print(record.steps)       # [{index, description, status, output, elapsed}, ...]

# Or start the background loop — processes all pending goals automatically
engine.start_loop(interval=30)   # checks every 30 seconds
```

**Step routing** — subtask descriptions are matched to real tools:

| Keywords in step | Tool called |
|---|---|
| read / open / load / ingest | `DocumentProcessor` (CSV, PDF, JSON, Excel, URLs) |
| train / fit / learn on | `ModelManager.train_from_csv()` |
| predict / forecast / estimate | `ModelManager.predict()` |
| browse / fetch / http / url | `BrowserAgent.fetch_text()` |
| screenshot / capture screen | `ScreenAgent.screenshot()` |
| notify / alert / remind | `Notifier.send()` |
| save / persist | `companion._handle_save()` |
| search / find / look up | `VectorMemory.search()` |
| anything else | LLM reasoning (Claude) → logged |

---

## Architecture — 137 stages

Mycelium was built incrementally across 137 stages, each a standalone tested module:

```
physml/
├── Physics ML core (stages 1–28)
│   predictor.py, estimator.py, feature_engineer.py, calibration.py, ...
│
├── Autonomous agent core (stages 29–99)
│   active_learner.py, mycelium_agent.py, memory.py, knowledge_graph.py,
│   policy_optimizer.py, autonomous_agent.py, world_model.py, safety.py, ...
│
├── System integration (stages 100–128)
│   mycelium_system.py, companion.py, llm_integration.py, model_manager.py,
│   tool_bridge.py, vector_memory.py, voice_loop.py, server.py, ...
│
└── Autonomous action layer (stages 129–137)
    screen_agent.py        screen capture, mouse/keyboard control
    browser_agent.py       Playwright browser automation
    permission_manager.py  OS action gating (allow/ask/deny)
    file_watcher.py        auto-ingest new files from watched dirs
    notifier.py            desktop push notifications
    knowledge_extractor.py extract facts from conversation
    feedback_loop.py       live model corrections via partial_fit
    personalisation.py     manual config + auto-learned profile overlay
    goal_engine.py         ← NEW: persistent goal queue + autonomous loop
```

### Subsystem map

| Layer | Key classes |
|---|---|
| **ML engine** | `PhysicsPredictor`, `PhysicsRegressor`, `PhysicsClassifier`, `myco` |
| **Active learning** | `MyceliumAgent`, `ActiveLearner`, `BanditPolicy` |
| **Memory** | `KnowledgeGraph`, `VectorMemory`, `AgentMemory`, `ReplayBuffer` |
| **Planning** | `TaskDecomposer`, `PlanExecutor`, **`GoalEngine`** |
| **Perception** | `DocumentProcessor`, `ScreenAgent`, `BrowserAgent` |
| **Action** | `ToolBridge`, `LocalTaskExecutor`, `PermissionManager` |
| **Learning from you** | `KnowledgeExtractor`, `FeedbackLoop`, `UserProfileLearner` |
| **Identity** | `DigitalSoul`, `PersonalisationManager` |
| **Infrastructure** | `Notifier`, `FileWatcher`, `SecureVault`, `ModelManager` |
| **API** | `MyceliumCompanion`, FastAPI server, CLI |

---

## Command line

```bash
# Train and predict
physml fit my_data.csv --target outcome_column --out agent.pkl
physml predict agent.pkl 1.2 3.4 5.6

# Start the REST API
uvicorn physml.server:app --reload
```

---

## REST API

```bash
pip install "physml[server]"
uvicorn physml.server:app
```

```
POST /train           — train on a CSV file
POST /predict         — predict from feature values
GET  /status          — system health + model status
POST /chat            — conversational interface
```

---

## Privacy

- All processing happens on your device
- No accounts or sign-in required
- Nothing uploaded anywhere
- Every file writes to `~/.mycelium/` only
- `PermissionManager` gates all OS actions — you control what Myco can touch

---

## System requirements

| | Minimum |
|---|---|
| OS | Windows 10 / macOS 10.15 / Linux |
| Python | 3.10+ |
| RAM | 2 GB |
| Storage | 300 MB |
| Internet | Not required after install |

---

## Development

```bash
git clone https://github.com/chizoalban2003-beep/Mycelium.git
cd Mycelium
pip install -e ".[dev]"
python3 -m pytest -q
```

**Test coverage:** 19 test files, 500+ tests, all passing.

---

## Roadmap

| Next step | What it adds |
|---|---|
| Vision / VLM | Describe what's on screen; full computer-use agent |
| Scheduled goals | Cron-like recurring goals ("check my inbox every morning") |
| Multi-agent federation | Specialist agents (finance, health, code) sharing knowledge |
| Mobile edge deployment | Raspberry Pi / Android with stripped-down core |
| Goal marketplace | Community-shared goal templates |

---

## License

MIT — free to use, modify, and share.

---

<details>
<summary>Technical reference — physics ML engine</summary>

PhysML frames supervised learning as a **gel electrophoresis simulation**.

Features are treated as charged particles migrating through a viscous medium. Their "charge" is their statistical association with the target; "viscosity" is modulated by collinearity, distribution shape, and a PCR-style amplification step.

```
Raw tabular data
      │
      ▼
  Cleaning & imputation  (rolling median, MAD, winsorize)
      │
      ▼
  Feature scoring        (Pearson / Spearman / Cramér-V / KL-divergence)
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

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `plane` | `"liquid"` | Medium preset: `solid` / `liquid` / `gas` |
| `n_cycles` | 30 | Electrophoresis iterations |
| `cycle_learning_rate` | 0.18 | Per-cycle charge update rate |
| `cascade_enabled` | `True` | Multicollinearity suppression |
| `pcr_enabled` | `False` | PCR amplification of strong features |
| `quantile_transform` | `False` | Rank-normalise numeric features |
| `residual_model` | `None` | Second-stage residual corrector |
| `backend` | `"physics"` | `"physics"` or `"neural"` |

</details>
