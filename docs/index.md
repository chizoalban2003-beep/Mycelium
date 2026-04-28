# PhysML / Mycelium

> **Physics-Inspired Autonomous AI Companion**

[![CI](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml/badge.svg)](https://github.com/chizoalban2003-beep/Mycelium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/physml.svg)](https://pypi.org/project/physml/)
[![Python](https://img.shields.io/pypi/pyversions/physml.svg)](https://pypi.org/project/physml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Mycelium** (package: `physml`) is a local autonomous AI assistant that learns from
everything on your device — documents, browsing, screen activity, keyboard/mouse actions —
and helps you get digital work done faster.  No cloud required.  Your data stays local.

It grew from a physics-inspired active-learning core (`myco`) into a full **digital companion**
with multi-modal ingestion, imitation learning, a browser extension, a mobile PWA, and a
multi-agent specialist federation.

---

## What It Does

| Capability | Description |
|---|---|
| **Active learning** | `myco` — trains on tabular data, asks for labels only when uncertain |
| **Multi-modal ingestion** | Learn from PDFs, code, URLs, images, audio — anything |
| **Screen observation** | Background monitor tracks what you work on, focus time, app usage |
| **Macro recording** | Records your keyboard/mouse actions; learns to suggest and replay them |
| **Imitation learning** | Policy model trained on your actions predicts what you'll do next |
| **User model** | Unified profile aggregating all learning streams |
| **LLM companion** | Claude-powered conversation, goal execution, voice interface |
| **Browser extension** | Chrome/Firefox extension — send pages, selections, bookmarks to Myco |
| **Mobile PWA** | Progressive Web App installable on iOS/Android, talks to local server |
| **Specialist federation** | Multi-agent routing: Coder, Browser, Data, Scheduler, NLP agents share knowledge |
| **REST API** | FastAPI server with Prometheus metrics, mobile endpoints, browser extension endpoints |
| **CLI** | `physml` command for all major operations |

---

## Quick Start

```bash
pip install physml
# Full digital companion:
pip install "physml[companion]"
```

```python
from physml import Companion

myco = Companion()
myco.start()              # boots all subsystems
myco.chat("What can you do?")
myco.ingest("~/Documents/project_brief.pdf")
myco.start_screen_observer()
```

See [Getting Started](getting_started.md) for step-by-step walkthroughs.

---

## Architecture

```
physml
├── myco                      ← active-learning agent (physics core)
├── Companion                 ← orchestrator, boots all subsystems
│   ├── MultiModalIngester    ← PDF / code / URL / image / audio → knowledge
│   ├── ScreenObserver        ← background screen monitor
│   ├── MacroRecorder         ← action sequence recorder
│   ├── ImitationLearner      ← policy model trained on macros
│   ├── UserModel             ← unified user representation
│   ├── SpecialistFederation  ← multi-agent specialist routing
│   ├── GoalEngine            ← plan + execute multi-step goals
│   ├── SkillLibrary          ← reusable named automation skills
│   ├── VectorMemory          ← semantic memory with embeddings
│   └── KnowledgeGraph        ← fact store with entity relations
├── server (FastAPI)          ← REST API, mobile endpoints, ext endpoints
├── browser_ext/              ← Chrome/Firefox MV3 extension
├── static/pwa/               ← Mobile Progressive Web App
└── cli                       ← physml command-line interface
```

---

## Feature Matrix

| Feature | Install extra | Status |
|---|---|---|
| Active learning | *(core)* | Stable |
| Multi-modal ingestion | `[full]` | Stable |
| Screen observer | `[screen]` | Stable |
| Macro recorder + imitation | `[companion]` | Stable |
| LLM companion (Claude) | `[llm]` | Stable |
| Voice interface | `[voice]` | Stable |
| Browser extension | `[server]` | Stable |
| Mobile PWA | `[server]` | Stable |
| Specialist federation | *(core)* | Stable |
| Federated ML (FedAvg) | *(core)* | Stable |
| Drift detection | *(core)* | Stable |
| Privacy (DP) | *(core)* | Stable |

---

See [API Reference](api_reference.md) for the full class/method catalogue.
