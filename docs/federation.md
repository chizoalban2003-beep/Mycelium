# Specialist Federation

`SpecialistFederation` wires multiple domain-specialist agents together through a
shared knowledge bus (`AgentComms`), shared `KnowledgeGraph`, and shared `VectorMemory`.
Queries are automatically routed to the best specialist based on topic and context.

---

## Architecture

```
SpecialistFederation
├── AgentComms bus          ← inter-agent messaging
├── Shared KnowledgeGraph   ← facts visible to all specialists
├── Shared VectorMemory     ← semantic memory visible to all
└── Specialists
    ├── Coder      ← code generation, debugging, refactoring
    ├── Browser    ← web research, link summarisation
    ├── Data       ← SQL, CSV, data analysis, statistics
    ├── Scheduler  ← calendar, reminders, time planning
    ├── NLP        ← text summarisation, translation, extraction
    └── System     ← file management, OS tasks, shell commands
```

When you call `fed.query(...)`, the federation:
1. Analyses the query topic and context app
2. Selects the best specialist (or two if uncertain)
3. Injects shared knowledge into the specialist's prompt
4. Returns the response + which specialist answered

---

## Quick Start

```python
from physml import SpecialistFederation

fed = SpecialistFederation()
fed.start()

# Simple query
result = fed.query("How do I write a context manager in Python?")
print(f"[{result['specialist']}] {result['response']}")
# [Coder] A context manager uses __enter__ and __exit__...

# Context-aware query
result = fed.query(
    "What's wrong with this query?",
    context={"app": "DBeaver", "topic": "SQL"},
)
# → routed to Data specialist

# Push a fact to all specialists
fed.broadcast_fact("User's primary language is Python 3.12")

# Snapshot all shared knowledge
snapshot = fed.knowledge_snapshot()
```

---

## Routing Logic

Topic keywords trigger specialist routing:

| Keywords | Specialist |
|---|---|
| `code`, `function`, `bug`, `debug`, `error`, `class`, `import`, `test` | Coder |
| `search`, `website`, `url`, `article`, `browser`, `web`, `link`, `research` | Browser |
| `data`, `sql`, `query`, `csv`, `table`, `chart`, `statistics`, `analyse` | Data |
| `schedule`, `calendar`, `remind`, `meeting`, `appointment`, `deadline`, `plan` | Scheduler |
| `summarise`, `translate`, `extract`, `rewrite`, `grammar`, `text`, `document` | NLP |
| `file`, `folder`, `terminal`, `command`, `shell`, `install`, `system`, `os` | System |

If no specialist matches, the query goes to the **NLP** specialist as a default.

Context `app` also influences routing:
- VS Code / PyCharm / vim → Coder
- Chrome / Firefox / Safari → Browser
- DBeaver / Tableau / Excel → Data
- Terminal / iTerm / bash → System

---

## Knowledge Sharing

Every response from any specialist is automatically:
1. Stored in the shared `VectorMemory` so all agents can retrieve it later
2. Added to the shared `KnowledgeGraph` (facts extracted by `KnowledgeExtractor`)
3. Broadcast via `AgentComms` so other specialists can subscribe

```python
# Subscribe a custom handler to all specialist responses
def on_response(message):
    print(f"[{message.sender}] → {message.topic}: {message.content[:60]}")

fed.comms.subscribe("response", "my_handler")
fed.comms.register_handler("my_handler", on_response)
```

---

## Custom Specialists

```python
from physml.specialist_federation import Specialist

class SecuritySpecialist(Specialist):
    name = "Security"
    topics = ["vulnerability", "cve", "exploit", "auth", "jwt", "ssl", "tls"]
    apps = ["Burp Suite", "Wireshark"]

    def respond(self, query: str, context: dict, knowledge: str) -> str:
        prompt = f"You are a security expert.\n\nKnowledge:\n{knowledge}\n\nQuery: {query}"
        return self._llm_call(prompt)

fed = SpecialistFederation(specialists=[
    *SpecialistFederation.default_specialists(),
    SecuritySpecialist(),
])
```

---

## Integration With Companion

The `Companion` class automatically creates and starts a `SpecialistFederation`:

```python
from physml import Companion

c = Companion()
c.start()

result = c.federation.query("Explain this error", context={"app": "Terminal"})
```

Or via CLI:

```bash
physml federation --query "How do I optimise this loop?" --context coder
```
