"""Stage 146 — SpecialistFederation: multi-agent specialist routing and knowledge sharing.

Routes natural-language queries to domain-specialist agents (Coder, Browser, Data,
Scheduler, NLP, System) based on topic keywords and context app.  All specialists
share a common KnowledgeGraph and VectorMemory so facts learned by one are
available to all.

Architecture
------------
::

    SpecialistFederation
    ├── AgentComms bus            ← inter-agent messaging
    ├── Shared KnowledgeGraph     ← entity/fact store
    ├── Shared VectorMemory       ← semantic memory
    └── Specialists (6 default)
        ├── Coder     — code, debugging, refactoring
        ├── Browser   — web research, link summarisation
        ├── Data      — SQL, analysis, statistics
        ├── Scheduler — calendar, reminders, planning
        ├── NLP       — text summarisation, extraction
        └── System    — files, shell, OS tasks

Usage
-----
::

    from physml import SpecialistFederation

    fed = SpecialistFederation()
    fed.start()

    result = fed.query("How do I optimise this SQL join?")
    print(result["specialist"], result["response"])

    fed.broadcast_fact("User prefers Python 3.12 and type hints")
    print(fed.knowledge_snapshot())
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword → specialist routing table
# ---------------------------------------------------------------------------
_KEYWORD_MAP: Dict[str, str] = {
    # Coder
    "code": "Coder", "function": "Coder", "bug": "Coder", "debug": "Coder",
    "error": "Coder", "class": "Coder", "import": "Coder", "test": "Coder",
    "refactor": "Coder", "syntax": "Coder", "compile": "Coder",
    "python": "Coder", "javascript": "Coder", "typescript": "Coder",
    "java": "Coder", "rust": "Coder", "golang": "Coder", "api": "Coder",
    "async": "Coder", "await": "Coder", "exception": "Coder",
    # Browser
    "search": "Browser", "website": "Browser", "url": "Browser",
    "article": "Browser", "browser": "Browser", "web": "Browser",
    "link": "Browser", "research": "Browser", "google": "Browser",
    "http": "Browser", "download": "Browser", "scrape": "Browser",
    # Data
    "data": "Data", "sql": "Data", "query": "Data", "csv": "Data",
    "table": "Data", "chart": "Data", "statistics": "Data",
    "analyse": "Data", "analyze": "Data", "dataframe": "Data",
    "excel": "Data", "pandas": "Data", "database": "Data", "db": "Data",
    "join": "Data", "aggregate": "Data", "pivot": "Data",
    # Scheduler
    "schedule": "Scheduler", "calendar": "Scheduler", "remind": "Scheduler",
    "meeting": "Scheduler", "appointment": "Scheduler", "deadline": "Scheduler",
    "plan": "Scheduler", "tomorrow": "Scheduler", "next week": "Scheduler",
    "reminder": "Scheduler", "event": "Scheduler", "due": "Scheduler",
    # NLP
    "summarise": "NLP", "summarize": "NLP", "translate": "NLP",
    "extract": "NLP", "rewrite": "NLP", "grammar": "NLP", "text": "NLP",
    "document": "NLP", "read": "NLP", "essay": "NLP", "paragraph": "NLP",
    "email": "NLP", "letter": "NLP", "report": "NLP",
    # System
    "file": "System", "folder": "System", "terminal": "System",
    "command": "System", "shell": "System", "install": "System",
    "system": "System", "os": "System", "disk": "System", "process": "System",
    "rename": "System", "copy": "System", "move": "System", "delete": "System",
    "directory": "System", "path": "System", "permission": "System",
}

_APP_MAP: Dict[str, str] = {
    "vs code": "Coder", "vscode": "Coder", "pycharm": "Coder", "vim": "Coder",
    "neovim": "Coder", "nvim": "Coder", "emacs": "Coder", "sublime": "Coder",
    "atom": "Coder", "intellij": "Coder", "eclipse": "Coder",
    "chrome": "Browser", "firefox": "Browser", "safari": "Browser",
    "edge": "Browser", "brave": "Browser",
    "dbeaver": "Data", "tableau": "Data", "power bi": "Data",
    "excel": "Data", "jupyter": "Data",
    "terminal": "System", "iterm": "System", "konsole": "System",
    "bash": "System", "zsh": "System", "powershell": "System",
    "word": "NLP", "google docs": "NLP", "notion": "NLP", "obsidian": "NLP",
    "calendar": "Scheduler", "outlook": "Scheduler", "google calendar": "Scheduler",
}

_DEFAULT_SPECIALIST = "NLP"


# ---------------------------------------------------------------------------
# Specialist base class
# ---------------------------------------------------------------------------
@dataclass
class Specialist:
    """Domain specialist agent with LLM fallback."""

    name: str
    topics: List[str] = field(default_factory=list)
    apps: List[str] = field(default_factory=list)
    _llm: Any = field(default=None, repr=False)

    def _system_prompt(self) -> str:
        return (
            f"You are {self.name}, a specialist AI assistant focused on {', '.join(self.topics) or self.name.lower()} tasks. "
            "Be concise, practical, and accurate.  Prefer code examples when helpful."
        )

    def respond(self, query: str, context: dict, knowledge: str) -> str:
        if self._llm is not None:
            try:
                system = self._system_prompt()
                if knowledge:
                    system += f"\n\nRelevant knowledge:\n{knowledge[:2000]}"
                return self._llm.complete(query, system_prompt=system)
            except Exception as exc:
                _logger.debug("LLM call failed for %s: %s", self.name, exc)

        parts = [f"[{self.name}]"]
        if knowledge:
            parts.append(f"Based on available knowledge:\n{knowledge[:500]}")
        parts.append(
            f"I'm the {self.name} specialist. To fully answer '{query[:80]}...', "
            f"please ensure the LLM subsystem is configured (set ANTHROPIC_API_KEY)."
        )
        return "\n".join(parts)


def _make_defaults() -> List[Specialist]:
    return [
        Specialist(
            name="Coder",
            topics=["code", "programming", "debugging", "testing", "refactoring"],
            apps=["VS Code", "PyCharm", "vim", "Terminal"],
        ),
        Specialist(
            name="Browser",
            topics=["web research", "link summarisation", "online content"],
            apps=["Chrome", "Firefox", "Safari"],
        ),
        Specialist(
            name="Data",
            topics=["SQL", "data analysis", "statistics", "visualisation"],
            apps=["DBeaver", "Tableau", "Excel", "Jupyter"],
        ),
        Specialist(
            name="Scheduler",
            topics=["calendar", "reminders", "planning", "time management"],
            apps=["Calendar", "Outlook", "Notion"],
        ),
        Specialist(
            name="NLP",
            topics=["text summarisation", "writing", "translation", "extraction"],
            apps=["Google Docs", "Word", "Obsidian"],
        ),
        Specialist(
            name="System",
            topics=["file management", "shell commands", "OS tasks", "installation"],
            apps=["Terminal", "Finder", "Explorer"],
        ),
    ]


# ---------------------------------------------------------------------------
# FederationMessage (lightweight, no pydantic)
# ---------------------------------------------------------------------------
@dataclass
class FederationMessage:
    sender: str
    topic: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SpecialistFederation
# ---------------------------------------------------------------------------
class SpecialistFederation:
    """Multi-agent specialist federation with shared knowledge.

    Parameters
    ----------
    specialists : list[Specialist] or None
        Custom list of specialists.  Defaults to the 6 built-in specialists.
    knowledge_graph : KnowledgeGraph or None
        Shared fact store.  Auto-created when None.
    vector_memory : VectorMemory or None
        Shared semantic memory.  Auto-created when None.
    comms : AgentComms or None
        Message bus.  Auto-created when None.
    llm : LLMIntegration or None
        LLM backend injected into each specialist.
    persist_dir : str or None
        Directory for persisting shared knowledge.
    """

    def __init__(
        self,
        specialists: Optional[List[Specialist]] = None,
        knowledge_graph: Any = None,
        vector_memory: Any = None,
        comms: Any = None,
        llm: Any = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        self._specialists: List[Specialist] = specialists or _make_defaults()
        self._kg = knowledge_graph
        self._vm = vector_memory
        self._comms = comms
        self._llm = llm
        self._persist_dir = persist_dir
        self._log: List[FederationMessage] = []
        self._facts: List[str] = []
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._kg = self._get_kg()
        self._vm = self._get_vm()
        self._comms = self._get_comms()
        for spec in self._specialists:
            spec._llm = self._llm
            self._comms.subscribe("broadcast_fact", spec.name)
        self._started = True
        _logger.info(
            "SpecialistFederation started with specialists: %s",
            [s.name for s in self._specialists],
        )

    def _get_kg(self) -> Any:
        if self._kg is not None:
            return self._kg
        try:
            from physml.knowledge_graph import KnowledgeGraph
            return KnowledgeGraph()
        except Exception:
            return _NullKG()

    def _get_vm(self) -> Any:
        if self._vm is not None:
            return self._vm
        try:
            from physml.vector_memory import VectorMemory
            return VectorMemory()
        except Exception:
            return _NullVM()

    def _get_comms(self) -> Any:
        if self._comms is not None:
            return self._comms
        try:
            from physml.agent_comms import AgentComms
            return AgentComms()
        except Exception:
            return _NullComms()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _route(self, query: str, context: Optional[dict]) -> Specialist:
        query_lower = query.lower()
        context = context or {}

        scores: Dict[str, int] = {s.name: 0 for s in self._specialists}

        for kw, specialist_name in _KEYWORD_MAP.items():
            if kw in query_lower:
                scores[specialist_name] = scores.get(specialist_name, 0) + 1

        app = str(context.get("app", "")).lower()
        for app_kw, specialist_name in _APP_MAP.items():
            if app_kw in app:
                scores[specialist_name] = scores.get(specialist_name, 0) + 3

        topic_hint = str(context.get("topic", "")).lower()
        for kw, specialist_name in _KEYWORD_MAP.items():
            if kw in topic_hint:
                scores[specialist_name] = scores.get(specialist_name, 0) + 2

        best_name = max(scores, key=lambda k: scores[k])
        if scores[best_name] == 0:
            best_name = _DEFAULT_SPECIALIST

        for spec in self._specialists:
            if spec.name == best_name:
                return spec
        return self._specialists[0]

    def _retrieve_knowledge(self, query: str, n: int = 5) -> str:
        parts: List[str] = []
        try:
            if hasattr(self._vm, "search"):
                results = self._vm.search(query, top_k=n)
                for r in results or []:
                    text = r.get("text", r) if isinstance(r, dict) else str(r)
                    parts.append(str(text)[:300])
        except Exception:
            pass

        try:
            if hasattr(self._kg, "search_facts"):
                facts = self._kg.search_facts(query, limit=n)
                for f in facts or []:
                    parts.append(str(f)[:200])
        except Exception:
            pass

        parts.extend(self._facts[-10:])
        return "\n".join(parts[:n * 2])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def query(
        self,
        query: str,
        context: Optional[dict] = None,
        top_k_knowledge: int = 5,
    ) -> Dict[str, Any]:
        """Route a query to the best specialist and return the response.

        Parameters
        ----------
        query : str
            Natural-language question or instruction.
        context : dict or None
            Optional context with keys like ``app``, ``topic``, ``file``.
        top_k_knowledge : int
            Number of shared knowledge snippets to inject.

        Returns
        -------
        dict
            ``{"specialist": str, "response": str, "elapsed": float}``
        """
        if not self._started:
            self.start()

        t0 = time.time()
        specialist = self._route(query, context)
        knowledge = self._retrieve_knowledge(query, top_k_knowledge)

        try:
            response = specialist.respond(query, context or {}, knowledge)
        except Exception as exc:
            response = f"[{specialist.name}] Error: {exc}"
            _logger.warning("Specialist %s failed: %s", specialist.name, exc)

        elapsed = time.time() - t0
        msg = FederationMessage(
            sender=specialist.name,
            topic="response",
            content=response,
            metadata={"query": query[:100], "elapsed": elapsed},
        )
        self._log.append(msg)

        try:
            self._comms.publish(msg.sender, msg.topic, msg.content)
        except Exception:
            pass

        try:
            self._store_response(query, response, specialist.name)
        except Exception:
            pass

        return {"specialist": specialist.name, "response": response, "elapsed": elapsed}

    def _store_response(self, query: str, response: str, specialist: str) -> None:
        combined = f"Q: {query}\nA ({specialist}): {response}"
        content_hash = hashlib.md5(combined.encode()).hexdigest()[:8]
        try:
            if hasattr(self._vm, "add"):
                self._vm.add(combined, metadata={"specialist": specialist, "hash": content_hash})
        except Exception:
            pass
        try:
            if hasattr(self._kg, "add_fact"):
                self._kg.add_fact(f"{specialist} answered: {response[:200]}")
        except Exception:
            pass

    def broadcast_fact(self, fact: str) -> None:
        """Push a fact to all specialists and shared knowledge stores."""
        self._facts.append(fact)
        try:
            if hasattr(self._kg, "add_fact"):
                self._kg.add_fact(fact)
        except Exception:
            pass
        try:
            if hasattr(self._vm, "add"):
                self._vm.add(fact, metadata={"source": "broadcast"})
        except Exception:
            pass
        try:
            self._comms.publish("federation", "broadcast_fact", fact)
        except Exception:
            pass
        _logger.debug("Fact broadcast: %s", fact[:80])

    def list_specialists(self) -> List[str]:
        """Return names of all registered specialists."""
        return [s.name for s in self._specialists]

    def knowledge_snapshot(self) -> Dict[str, Any]:
        """Return a summary of shared knowledge."""
        return {
            "facts": self._facts[-20:],
            "log_size": len(self._log),
            "specialists": self.list_specialists(),
            "last_query": (
                self._log[-1].metadata.get("query") if self._log else None
            ),
        }

    def recent_log(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the n most recent federation messages."""
        return [
            {
                "sender": m.sender,
                "topic": m.topic,
                "content": m.content[:200],
                "timestamp": m.timestamp,
            }
            for m in self._log[-n:]
        ]

    @staticmethod
    def default_specialists() -> List[Specialist]:
        """Return the default 6 built-in specialists."""
        return _make_defaults()


# ---------------------------------------------------------------------------
# Null stubs for graceful degradation
# ---------------------------------------------------------------------------
class _NullKG:
    def add_fact(self, *a: Any, **kw: Any) -> None: ...
    def search_facts(self, *a: Any, **kw: Any) -> List: return []


class _NullVM:
    def add(self, *a: Any, **kw: Any) -> None: ...
    def search(self, *a: Any, **kw: Any) -> List: return []


class _NullComms:
    def subscribe(self, *a: Any, **kw: Any) -> None: ...
    def publish(self, *a: Any, **kw: Any) -> None: ...
