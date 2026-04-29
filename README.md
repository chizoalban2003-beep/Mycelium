# 🍄 Mycelium — Local Autonomous AI Companion

[![CI](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml/badge.svg)](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/physml.svg)](https://pypi.org/project/physml/)
[![Python](https://img.shields.io/pypi/pyversions/physml.svg)](https://pypi.org/project/physml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An AI that lives on your device, learns you over time, and acts on your behalf — autonomously.**  
> No cloud. No data leaving your machine. No subscription.

---

## What is Mycelium?

Mycelium is a **145-stage autonomous digital companion** built in Python. It combines a physics-inspired ML engine with a full agent stack: voice interaction, semantic memory, goal planning, browser control, screen automation, desktop task execution, and multi-channel messaging — all running locally.

Like the mycelium fungus that silently connects a forest, Myco works in the background — learning your patterns, watching for new data, executing multi-step goals, sending messages on your behalf, and notifying you when something needs attention.

---

## What it can do right now

### Learn from any kind of content (v1.1)
```python
myco = MyceliumCompanion()
myco.start()

# Ingest a document, code file, URL, or raw text — all stored in semantic memory
myco.ingest("research_notes.pdf")
myco.ingest("https://arxiv.org/abs/2305.12345")
myco.ingest("My name is Alex and I specialise in distributed systems.")
myco.ingest("src/main.py")   # learns your code patterns

# Or from the CLI
# physml ingest research_notes.pdf --topic research
```

### Watch and learn what you do on-screen (v1.1)
```python
# Background screen observer — quietly tracks what you're working on
observer = myco.start_screen_observer(interval=60.0)
# ...
print(observer.top_apps(5))
# [("VSCode", 3620.0), ("Chrome", 1200.0), ("Terminal", 800.0)]
```

### Record actions and replay them (v1.1)
```python
# Record a task you perform manually
myco.start_macro_recording("rename_reports")
# ... user renames files, types text, clicks around ...
seq = myco.stop_macro_recording()
# Macro is auto-saved as a reusable Skill

# Myco learns from your recordings and proactively suggests actions
suggestions = myco.suggest_next_action(context_app="Finder")
for s in suggestions:
    print(s.action_type, f"{s.confidence:.0%}")
# double_click  72%
# type_text     18%
```

### Run fully offline — no API key needed (v1.3)
```bash
# Install and start ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2        # ~2GB text model
ollama pull llava           # ~4GB vision model (optional)
ollama serve                # start the local server
```

```python
from physml.llm import LocalLLM

llm = LocalLLM()            # auto-detects ollama
print(llm.available)        # True when ollama is running
result = llm.complete("Explain Python decorators")
print(result.text)

# Vision analysis (requires llava)
result = llm.vision_chat(image_b64, "What UI elements do you see?")
```

Myco automatically uses your local ollama when `ANTHROPIC_API_KEY` is not set — no code changes needed.

### Automate any UI task with vision (v1.3)
```python
from physml import VisionAgent

va = VisionAgent()          # auto-detects Claude Vision or ollama/llava

# Analyse what's on screen
result = va.analyse_current_screen()
print(result.description)
print(result.elements)      # detected buttons, inputs, menus with coordinates

# Find and click any element by description
va.find_and_click("Save button")
va.find_and_click("search box")

# Wait until something appears
va.watch_for("confirmation dialog", timeout=10.0)

# Ask vision model how to perform a goal step
step = va.describe_goal_step("send email", "compose message")
print(step["action"], step["target_description"])
```

### Ask specialist agents for domain-specific help (v1.2)
```python
from physml import SpecialistFederation

fed = SpecialistFederation()
fed.start()

# Routes automatically to the right specialist
result = fed.query("How do I optimise this SQL join?")
print(f"[{result['specialist']}] {result['response']}")
# [Data] Use an index on the join column...

result = fed.query("debug my Python exception", context={"app": "VS Code"})
# [Coder] Check the traceback for...

# Broadcast a fact — all specialists learn it
fed.broadcast_fact("User works with PostgreSQL 16 and Python 3.12")
```

### Use it from your phone (v1.2 — Mobile PWA)
```bash
# Start server on your computer
uvicorn physml.server:app --host 0.0.0.0 --port 8000

# On your phone: open http://<your-ip>:8000/pwa/
# Tap "Add to Home Screen" → installed as a native-feeling app
```

The mobile PWA includes: chat, knowledge ingestion, context view, behavioral patterns, and current task intent — all connecting to your local Myco server over Wi-Fi.

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

myco.personalise("verbosity", "concise")
myco.personalise("name", "Alex")
```

### Give it a goal — it works autonomously
```python
goal_id = myco.goal_engine.add_goal(
    "Read quarterly_report.csv, train a model on it, then notify me of the results"
)
myco.goal_engine.start_loop()

print(myco.goals())
# Goals (1):
#   [completed] a1b2c3d4: Read quarterly_report.csv ... [3/3 steps] (12s)

# Goals also learn from the past — similar goals reuse proven step sequences
myco.goal_engine.add_goal("Analyse annual_report.csv and summarise it")
# → automatically reuses the steps that worked for quarterly_report.csv
```

### Send messages and notifications
```python
# CommBridge — routes to the right channel automatically
myco.chat("send email to alice@corp.com subject 'Q3 results' body 'Done, see attached'")
myco.chat("text +15551234567 hey just finished the report")
myco.chat("post to slack: analysis complete, check the dashboard")
myco.chat("whatsapp Bob: can we reschedule to Thursday?")

# Direct API
myco.comm_bridge.send_email("alice@corp.com", "Q3 results", "Done!")
myco.comm_bridge.send_sms("+15551234567", "Done!")
myco.comm_bridge.send_slack("Analysis complete")
```

### Control your desktop
```python
# DesktopBridge — everyday computer tasks via natural language
myco.chat("open file ~/Documents/report.pdf")
myco.chat("list files in ~/Downloads")
myco.chat("copy 'Meeting notes from today' to clipboard")
myco.chat("take a screenshot")
myco.chat("open app Chrome")

# With MYCO_ALLOW_WRITES=1:
myco.chat("write file ~/output.txt with content 'analysis complete'")
myco.chat("delete file ~/tmp/old_report.csv")
```

### Talk to it — voice in, voice out
```python
# Activate the voice loop (requires faster-whisper + sounddevice + pyttsx3)
myco.start_voice(wake_word="hey myco", speak_response=True)
# → Myco listens, transcribes, responds, and speaks — all locally

# Stop voice
myco.stop_voice()

# Or use VoiceInterface (simpler, speech_recognition + pyttsx3)
myco.start_voice_interface(tts=True, language="en-US")
```

### Use the LLM + action dispatch layer directly

```python
from physml.llm import ClaudeClient, PromptSystem, ActionDispatcher
from physml.conversation_store import ConversationStore

client = ClaudeClient()          # reads ANTHROPIC_API_KEY
store = ConversationStore("~/.mycelium/conversations/demo.json")
ps = PromptSystem(client=client)
dispatcher = ActionDispatcher(store=store, client=client)

# Route intent and execute action in one step
action = ps.route("train on sales.csv")
print(action.intent)             # "train"
response = dispatcher.dispatch(action)
print(response)                  # "Trained on 'sales.csv': 1200 samples ..."

# Voice interface with graceful degradation
from physml.voice import VoiceInterface
vi = VoiceInterface(prompt_system=ps, dispatcher=dispatcher, tts=False)
print(vi.available)              # True if speech_recognition installed
vi.run_once("show me a report")  # single text turn (no mic needed)
```

### Run benchmark experiments with Claude analysis

```python
from physml.experiment_runner import ExperimentRunner
from physml.llm import ClaudeClient

runner = ExperimentRunner()
summary = runner.run(task="regression", n_samples=200)
print(summary)   # BenchmarkSummary(task=regression, best_R²=0.9412, ...)

# Get a plain-English analysis from Claude
client = ClaudeClient()
analysis = runner.analyze_with_llm(summary, client)
print(analysis)  # "The best configuration used a liquid-phase medium with 10 cycles..."
```

### Scheduled goals — recurring tasks
```python
# Run a goal every morning
myco.schedule_goal("Check ~/Downloads for new CSV files and auto-train", schedule="daily")

# Every 30 minutes
myco.schedule_goal("Take a screenshot and save to ~/myco-snapshots", schedule="every 30 minutes")

myco.scheduler.start()   # background thread picks up schedules automatically
```

### Watch directories for new data
```python
myco.personalise("watch_dirs", ["~/Downloads"])
# Auto-trains whenever a new CSV appears
```

### Browse the web and process documents
```python
text = myco.browse("https://example.com/report")
myco.chat("read ~/docs/annual_report.pdf and summarise key points")
```

### REST API + streaming web UI
```bash
uvicorn physml.server:app
# Open http://localhost:8000 → full web chat UI with goal panel + streaming
```

```
GET  /goals               list all goals
POST /goals               queue a new goal
GET  /goals/{id}          get goal by ID
DELETE /goals/{id}        cancel a goal
GET  /schedules           list schedules
POST /schedules           add a recurring schedule
POST /chat                chat (JSON)
POST /chat/stream         streaming SSE chat (tokens as they arrive)
GET  /digest              24-hour activity digest
GET  /voice/status        voice loop status
POST /voice/start         start voice loop
POST /voice/stop          stop voice loop
GET  /comm/status         CommBridge channel config status
GET  /desktop/status      DesktopBridge capability status
GET  /metrics             Prometheus metrics
```

### Daily digest
```python
print(myco.daily_digest())
# === Myco Daily Digest ===
# TL;DR: Productive day — 3 goals completed, model trained on 1,200 rows.
#
# Goals (last 24 h):
#   Completed : 3
#   Failed    : 0
#   Pending   : 1
# Schedules : 2 registered, 2 enabled
# Model     : trained (1200 rows)
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

| Extra | Unlocks |
|---|---|
| `llm` | Claude API (Anthropic) for intelligent chat and LLM-driven planning |
| `voice` | faster-whisper STT + pyttsx3 TTS + wake word |
| `vector` | Semantic memory (sentence-transformers) |
| `browser` | Playwright browser + WhatsApp Web automation |
| `screen` | pyautogui + mss screen control |
| `notify` | plyer desktop notifications |
| `ocr` | pytesseract image text extraction |
| `watcher` | watchdog file system monitoring |
| `sms` | twilio SDK for SMS sending |
| `companion` | All of the above bundled |

---

## Configuration

Copy `.env.example` to `.env` and fill in what you need:

```bash
cp .env.example .env
```

Key variables:

```env
# LLM (optional — unlocks intelligent chat + planning)
ANTHROPIC_API_KEY=sk-ant-...

# Email sending
MYCO_EMAIL_HOST=smtp.gmail.com
MYCO_EMAIL_USER=you@gmail.com
MYCO_EMAIL_PASS=your-app-password

# SMS (requires pip install twilio)
TWILIO_ACCOUNT_SID=ACxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxx
TWILIO_FROM_NUMBER=+15551234567

# Slack
MYCO_SLACK_WEBHOOK=https://hooks.slack.com/services/...

# Allow file write/delete operations
MYCO_ALLOW_WRITES=0
```

---

## Quick start — the ML core

```python
from physml import PhysicsRegressor, PhysicsClassifier

from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
reg = PhysicsRegressor()
reg.fit(X, y)
print(reg.score(X, y))   # R²
```

---

## Architecture — 145 stages

Built incrementally, every stage a standalone tested module:

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
├── Autonomous action layer (stages 129–142)
│   screen_agent.py        screen capture, mouse/keyboard control
│   browser_agent.py       Playwright browser automation
│   permission_manager.py  OS action gating (allow/ask/deny)
│   file_watcher.py        auto-ingest new files from watched dirs
│   notifier.py            desktop push notifications
│   knowledge_extractor.py extract facts from conversation
│   feedback_loop.py       live model corrections via partial_fit
│   personalisation.py     manual config + auto-learned profile overlay
│   goal_engine.py         persistent goal queue + autonomous loop
│   scheduled_goals.py     cron-like recurring goals
│   goal_feedback.py       learn from past goal outcomes (Stage 139)
│   server.py              Goals/Schedules REST API + SSE streaming + digest
│
└── Digital action layer (stages 143–145)
    comm_bridge.py          email, SMS, Slack, WhatsApp messaging
    desktop_bridge.py       file I/O, clipboard, app launch, shell, screen
    companion.py            start_voice() / stop_voice() + REST voice endpoints
```

### Subsystem map

| Layer | Key classes |
|---|---|
| **ML engine** | `PhysicsPredictor`, `PhysicsRegressor`, `PhysicsClassifier`, `myco` |
| **Active learning** | `MyceliumAgent`, `ActiveLearner`, `BanditPolicy` |
| **Memory** | `KnowledgeGraph`, `VectorMemory`, `AgentMemory`, `ReplayBuffer` |
| **Planning** | `TaskDecomposer`, `PlanExecutor`, `GoalEngine`, `GoalFeedbackStore` |
| **Perception** | `DocumentProcessor`, `ScreenAgent`, `BrowserAgent` |
| **Action** | `ToolBridge`, `LocalTaskExecutor`, `PermissionManager` |
| **Communication** | `CommBridge` (email/SMS/Slack/WhatsApp), `Notifier` |
| **Desktop control** | `DesktopBridge` (files, clipboard, apps, shell, screen) |
| **Voice** | `VoiceLoop`, `VoiceInputAdapter`, `VoiceOutputAdapter` |
| **Learning from you** | `KnowledgeExtractor`, `FeedbackLoop`, `UserProfileLearner` |
| **Identity** | `DigitalSoul`, `PersonalisationManager` |
| **Infrastructure** | `FileWatcher`, `SecureVault`, `ModelManager`, `ScheduledGoals` |
| **API** | `MyceliumCompanion`, FastAPI server + SSE streaming, CLI |
| **LLM layer** | `ClaudeClient`, `PromptSystem`, `ActionDispatcher`, `physml chat`, `physml voice` |

### GoalEngine step routing

| Keywords in step | Tool called |
|---|---|
| read / open / load / ingest | `DocumentProcessor` (CSV, PDF, JSON, Excel, URLs) |
| train / fit / learn on | `ModelManager.train_from_csv()` |
| predict / forecast / estimate | `ModelManager.predict()` |
| send email / email | `CommBridge.parse_and_send_email()` |
| send sms / text message | `CommBridge.parse_and_send_sms()` |
| whatsapp | `CommBridge.send_whatsapp()` |
| slack | `CommBridge.send_slack()` |
| open file / read file | `DesktopBridge.read_file()` |
| write file / save file | `DesktopBridge.write_file()` |
| open app | `DesktopBridge.open_app()` |
| copy to clipboard | `DesktopBridge.copy_to_clipboard()` |
| screenshot / capture screen | `DesktopBridge.screenshot()` |
| run command / execute | `DesktopBridge.run_shell()` |
| browse / fetch / http | `BrowserAgent.fetch_text()` |
| notify / alert / remind | `Notifier.send()` |
| digest / daily summary | `companion.daily_digest()` |
| anything else | LLM reasoning (Claude) → logged |

---

## Docker / deployment

```bash
# Start the full stack (API + worker)
docker compose up

# Worker only (background goal processing, no HTTP server)
python scripts/run_worker.py
```

Environment via `.env` file (see `.env.example`).

---

## Command line

```bash
# Train a model on a CSV file
physml fit my_data.csv --target outcome_column --out agent.pkl

# Predict with a saved agent
physml query agent.pkl test.csv --out predictions.csv

# Print a report of a saved agent
physml report agent.pkl

# Export predictions to CSV
physml export agent.pkl test.csv --out predictions.csv

# Also invokable as a Python module (no PATH setup required)
python3 -m physml --help

# Natural-language REPL (executes train/predict/report/save actions)
physml chat --session myproject
# you> train on sales.csv
# myco> Trained on 'sales.csv': 1200 samples, 5 features, target='revenue'.
# you> predict 1500 40 3.5
# myco> Prediction: 1.0  confidence: 82%
# you> report
# myco> n_samples_seen_: 1200 ...
# you> save
# myco> Agent saved to 'agent.pkl'

# Voice interaction loop (SR → Whisper offline → text fallback)
# Install offline STT: pip install openai-whisper sounddevice
physml voice --session myproject --language en-US
# → SR first, falls back to Whisper (offline), then to text input

# Run benchmark experiments on synthetic data
physml experiment --task classification --quick
# → runs 1 config, prints BenchmarkSummary + Claude analysis (if API key set)

# Ask Claude to explain a saved agent
physml explain agent.pkl

# Ingest any file, URL, or text
physml ingest notes.pdf --topic research
physml ingest "My name is Alex and I work as a data scientist."

# Start background screen observer (tracks what you work on)
physml observe --interval 30

# Record a macro (requires: pip install pynput)
physml record open_browser

# Show the current user model (context + patterns)
physml model

# Start the REST API + web UI (includes mobile API + browser extension endpoints)
uvicorn physml.server:app --reload

# Browser extension: load physml/browser_ext/ as unpacked Chrome/Firefox extension
# Then the extension posts to http://localhost:8000/ext/* automatically
```

---

## Privacy

- All processing happens on your device
- No accounts or sign-in required
- Nothing uploaded anywhere
- Every file writes to `~/.mycelium/` only
- `PermissionManager` gates all OS actions — you control what Myco can touch
- `MYCO_ALLOW_WRITES=0` by default — file writes require explicit opt-in

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

# Fast tests (CI default — excludes CPU-heavy slow tests)
python3 -m pytest tests/ -q --timeout=90 -m "not slow"

# All tests
python3 -m pytest tests/ -q --timeout=120
```

**Test coverage:** 22 test files, 550+ tests, all passing.

---

## What's shipped vs what's next

### Shipped ✅

| Version | Feature |
|---|---|
| v1.0 | Active learning core, GoalEngine, voice interface, REST API, CLI |
| v1.1 | Multi-modal ingestion, screen observer, macro recorder, imitation learner, user model, browser extension, mobile API |
| v1.2 | Specialist federation (6 domain agents), mobile PWA, comprehensive docs, PyPI distribution packages |
| v1.3 | Local LLM (ollama + llama.cpp), VisionAgent (computer-use), server auth hardening + rate limiting |

### Next steps

| Feature | What it adds |
|---|---|
| Vision / VLM | Full computer-use agent — describe and act on any screen |
| Mobile edge deployment | Raspberry Pi / Android with stripped-down core |
| Goal marketplace | Community-shared goal templates and skill packs |
| WhatsApp/Telegram native API | Replace Playwright automation with official APIs |
| PyPI publish | Push v1.2.0 tag → GitHub Actions auto-publishes to PyPI |
| Fine-tuned local LLM | `llama.cpp` / `ollama` backend for fully-offline LLM |

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

### Benchmark Results

| Dataset | Samples | Features | Final Accuracy |
|---|---|---|---|
| iris | 150 | 4 | 93.8% |
| breast_cancer | 569 | 30 | 87.6% |
| wine | 178 | 13 | 56.3% |

Reproduce: `python benchmarks/run_benchmarks.py`

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
