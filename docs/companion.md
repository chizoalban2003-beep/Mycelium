# Companion — Digital Assistant Guide

`Companion` is the top-level orchestrator that boots all subsystems and gives you
a single conversational interface to your local Mycelium assistant.

---

## Quick Start

```bash
pip install "physml[companion]"
export ANTHROPIC_API_KEY=sk-ant-...   # optional — enables LLM features
```

```python
from physml import Companion

c = Companion(persist_dir="~/.mycelium")
c.start()
print(c.chat("What can you do for me?"))
```

---

## Configuration

```python
c = Companion(
    persist_dir="~/.mycelium",   # where all state is persisted
    api_key=None,                # ANTHROPIC_API_KEY env var used if None
)
```

All subsystems are **lazily initialised** — they start on first use, not at `Companion()` call time.
Calling `start()` pre-warms everything so the first chat isn't slow.

---

## Learning From Your Documents

```python
# Single file (PDF, DOCX, TXT, MD, PY, JS, CSV, image, audio URL, ...)
c.ingest("~/Documents/quarterly_report.pdf", topic="finance")

# Directory
c.ingest("~/code/myproject/", topic="codebase")

# URL
c.ingest("https://docs.python.org/3/library/asyncio.html", topic="python")

# Raw text
c.ingest("Meeting notes: Alice will lead the backend refactor.", topic="meetings")
```

Everything is stored in both **vector memory** (semantic search) and the
**knowledge graph** (entity + fact relationships).  The LLM companion automatically
searches this knowledge when answering questions.

---

## Screen Observer

```python
# Start passive background monitoring (60-second snapshots)
c.start_screen_observer(
    interval=60.0,
    save_screenshots=False,   # set True to keep PNG files
    llm_describe=True,        # use Claude Vision API for descriptions
)
```

The observer detects your active app and window title on Linux (xdotool),
macOS (osascript), and Windows (ctypes).  Each snapshot is automatically
ingested into vector memory so you can ask questions like:

```python
c.chat("What have I been working on for the past hour?")
```

---

## Macro Recording and Imitation Learning

Record a workflow once, replay it on demand:

```python
# Record
c.start_macro_recording("weekly_report")
# ... open Excel, pull data, paste into template, email ... (do it manually)
seq = c.stop_macro_recording()
print(seq.summarise())   # prints steps, duration, apps used
```

After recording several sequences, the imitation learner trains automatically
and can suggest your next action:

```python
suggestions = c.suggest_next_action(context_app="Excel")
# [ActionSuggestion(action_type='KEY_PRESS', confidence=0.82, text='Ctrl+C'), ...]
```

---

## Goal Execution

Set a natural-language goal and Myco will plan and execute it:

```python
c.set_goal("Find all PDF invoices in ~/Downloads and rename them with the invoice date")
```

Completed goals are automatically saved as reusable **skills** in `SkillLibrary`,
so the same goal runs instantly next time.

---

## User Model

The UserModel aggregates everything into a single profile used to personalise
all responses:

```python
ctx = c.user_model.current_context()
# {
#   "app": "VS Code",
#   "mood": "focused",
#   "top_topics": ["Python", "FastAPI", "testing"],
#   "peak_hour": 14,
#   "verbosity": "concise",
#   "session_seconds": 3600
# }

print(c.user_model.inject_into_prompt())
# → "You are assisting a developer currently working in VS Code..."
```

---

## Voice Interface

```bash
pip install "physml[voice]"
```

```python
c.start_voice_interface()
# Now speak naturally — Myco listens, responds with voice output
```

Whisper (offline STT) is used when `faster-whisper` is installed, falling back to
`speech_recognition` and then to text input.

---

## Persistence

Everything persists to `persist_dir` (default `~/.mycelium`):

| Subdirectory | Contents |
|---|---|
| `vector_memory/` | Embeddings + text chunks |
| `knowledge_graph/` | Entity and fact graph |
| `macros/` | Recorded action sequences |
| `user_model/` | User profile, event log |
| `skills/` | Saved skills from SkillLibrary |
| `conversations/` | Chat history |

---

## Specialist Federation

```python
result = c.federation.query(
    "Explain this stack trace",
    context={"app": "Terminal", "topic": "debugging"},
)
print(f"[{result['specialist']}] {result['response']}")
```

See [API Reference](api_reference.md#specialistfederation) for full details.
