"""physml.llm.prompt_system — Natural language routing via Claude or rule-based fallback.

:class:`PromptSystem` is the single entry-point for turning a user's free-text
prompt into a structured action that can be dispatched to the right physml
subsystem.  Two backends are supported:

1. **Claude** (when the SDK is available and ``ANTHROPIC_API_KEY`` is set):
   Uses a structured tool-call to extract intent + payload from the prompt.
2. **Rule-based fallback** (always available):
   Uses :class:`~physml.nl_router.NaturalLanguageRouter` (TF-IDF / keyword
   matching) to classify intent and regex to extract entities.

The output is a :class:`PromptAction` with:

* ``intent`` — one of ``"train"``, ``"predict"``, ``"report"``, ``"read_doc"``,
  ``"run_task"``, ``"add_goal"``, ``"show_goals"``, ``"help"``, ``"unknown"``.
* ``payload`` — dict of extracted entities (paths, numbers, key-value pairs).
* ``confidence`` — 0–1 match confidence.
* ``raw_text`` — the original user input.
* ``via_llm`` — ``True`` when Claude was used for routing.

Usage::

    from physml.llm.prompt_system import PromptSystem

    ps = PromptSystem()
    action = ps.route("train a model on /data/sales.csv")
    print(action.intent)        # "train"
    print(action.payload)       # {"path": "/data/sales.csv"}

    # With a companion attached (enables richer context)
    ps = PromptSystem(companion=my_companion)
    action = ps.route("predict the revenue for next quarter using values 1.5 2.3")
    print(action.intent)        # "predict"
    print(action.payload)       # {"numbers": [1.5, 2.3]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Intent catalogue — matches physml subsystems
# ---------------------------------------------------------------------------

_INTENTS: Dict[str, List[str]] = {
    "train": [
        "train on", "learn from", "fit the model", "teach me", "build a model",
        "train a model", "load data", "import csv", "fine-tune on",
    ],
    "predict": [
        "predict", "forecast", "estimate", "what is", "classify", "infer",
        "run inference", "give me a prediction", "what will",
    ],
    "report": [
        "show report", "give me a report", "show stats", "how is the model",
        "model status", "performance", "accuracy", "what have you learned",
        "show me the results", "summary",
    ],
    "read_doc": [
        "read file", "open file", "read document", "load document",
        "read pdf", "open csv", "process document", "extract from",
    ],
    "run_task": [
        "run command", "execute", "do task", "run script", "shell",
        "run program", "execute task", "run this",
    ],
    "add_goal": [
        "add a goal", "set a goal", "create goal", "new goal", "queue goal",
        "i want you to", "your goal is", "remind me to", "schedule",
    ],
    "show_goals": [
        "show goals", "list goals", "what are my goals", "pending goals",
        "goals status", "what goals",
    ],
    "memory": [
        "what do you remember", "show memory", "what have you stored",
        "my preferences", "user profile", "what do you know about me",
    ],
    "help": [
        "help", "what can you do", "commands", "capabilities",
        "how do i", "tutorial", "guide",
    ],
    "save": [
        "save the model", "save session", "persist", "write to disk", "backup",
    ],
}

# ---------------------------------------------------------------------------
# Tool definition for Claude-based routing
# ---------------------------------------------------------------------------

_ROUTING_TOOL: Dict[str, Any] = {
    "name": "route_intent",
    "description": (
        "Classify the user's intent and extract structured entities from their message. "
        "Return the most specific intent and any relevant payload fields."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(_INTENTS.keys()) + ["unknown"],
                "description": "The user's primary intent.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence 0–1.",
            },
            "payload": {
                "type": "object",
                "description": (
                    "Extracted entities: 'path' (file path), 'numbers' (list of floats), "
                    "'target_column' (str), 'goal_description' (str), 'kv' (key-value pairs)."
                ),
            },
        },
        "required": ["intent", "confidence", "payload"],
    },
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PromptAction:
    """A structured action derived from a user prompt.

    Attributes
    ----------
    intent : str
        The classified intent (e.g. ``"train"``, ``"predict"``).
    confidence : float
        Confidence of the classification (0–1).
    payload : dict
        Extracted entities (paths, numbers, goal text, etc.).
    raw_text : str
        The original user input.
    via_llm : bool
        ``True`` when Claude was used for the routing decision.
    """

    intent: str
    confidence: float
    payload: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    via_llm: bool = False


# ---------------------------------------------------------------------------
# PromptSystem
# ---------------------------------------------------------------------------


class PromptSystem:
    """Natural language router with Claude + rule-based fallback.

    Parameters
    ----------
    client : ClaudeClient or None
        If ``None`` and the SDK is available, a :class:`ClaudeClient` is
        created automatically.
    companion : any, optional
        A ``MyceliumCompanion`` instance.  When provided, its ``llm``
        attribute is used as the routing backend.
    min_confidence : float
        Minimum confidence to accept a rule-based intent match.
    """

    def __init__(
        self,
        client: Any = None,
        companion: Any = None,
        min_confidence: float = 0.15,
    ) -> None:
        self._companion = companion
        self.min_confidence = min_confidence

        # Resolve the Claude client
        if client is not None:
            self._client = client
        elif companion is not None and getattr(companion, "llm", None) is not None:
            # Wrap the LLMIntegration as an ad-hoc bridge
            self._client = _LLMIntegrationBridge(companion.llm)
        else:
            try:
                from physml.llm.claude_client import ClaudeClient
                self._client = ClaudeClient()
            except Exception:
                self._client = None

        # Rule-based fallback router
        self._rule_router = _build_rule_router(min_confidence)

    # ------------------------------------------------------------------
    # Main routing method
    # ------------------------------------------------------------------

    def route(self, text: str) -> PromptAction:
        """Route *text* to a :class:`PromptAction`.

        Tries Claude first; falls back to rule-based routing on failure.

        Parameters
        ----------
        text : str
            Raw user input.

        Returns
        -------
        PromptAction
        """
        text = str(text).strip()
        if not text:
            return PromptAction("unknown", 0.0, {}, text)

        # Try LLM-based routing first
        if self._client is not None and getattr(self._client, "available", False):
            action = self._route_via_llm(text)
            if action is not None:
                return action

        # Fall back to rule-based routing
        return self._route_via_rules(text)

    # ------------------------------------------------------------------
    # LLM routing
    # ------------------------------------------------------------------

    def _route_via_llm(self, text: str) -> Optional[PromptAction]:
        try:
            system = (
                "You are a routing assistant for the Mycelium local AI agent. "
                "Your job is to classify the user's intent and extract entities. "
                "Call the route_intent tool with the best classification."
            )
            result = self._client.tool_call(
                user_message=text,
                tools=[_ROUTING_TOOL],
                system=system,
            )
            if result.tool_calls:
                tc = result.tool_calls[0]
                inp = tc.get("input", {})
                return PromptAction(
                    intent=inp.get("intent", "unknown"),
                    confidence=float(inp.get("confidence", 0.9)),
                    payload=inp.get("payload", {}),
                    raw_text=text,
                    via_llm=True,
                )
        except Exception as exc:
            _logger.debug("PromptSystem LLM routing failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Rule-based routing
    # ------------------------------------------------------------------

    def _route_via_rules(self, text: str) -> PromptAction:
        from physml.nl_router import _extract_entities

        routed = self._rule_router.route(text)
        entities = _extract_entities(text)

        payload: Dict[str, Any] = {}
        if "numbers" in entities:
            payload["numbers"] = entities["numbers"]
        if "paths" in entities:
            payload["path"] = entities["paths"][0]
            payload["paths"] = entities["paths"]
        if "quoted" in entities:
            payload["quoted"] = entities["quoted"]
        if "kv" in entities:
            payload["kv"] = entities["kv"]

        return PromptAction(
            intent=routed.intent,
            confidence=routed.confidence,
            payload=payload,
            raw_text=text,
            via_llm=False,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def describe_intent(self, intent: str) -> str:
        """Return a human-readable description of an intent."""
        _descriptions = {
            "train": "Train a model on data",
            "predict": "Run a prediction",
            "report": "Show model status / report",
            "read_doc": "Read / process a document",
            "run_task": "Execute a local task",
            "add_goal": "Queue a new autonomous goal",
            "show_goals": "List current goals",
            "memory": "Query conversation memory",
            "help": "Show help",
            "save": "Save the current session",
            "unknown": "Unknown intent",
        }
        return _descriptions.get(intent, f"Intent: {intent}")

    def __repr__(self) -> str:
        llm_ok = self._client is not None and getattr(self._client, "available", False)
        return f"PromptSystem(llm={llm_ok}, min_confidence={self.min_confidence})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_rule_router(min_confidence: float = 0.15) -> Any:
    """Build and return a pre-populated NaturalLanguageRouter."""
    from physml.nl_router import NaturalLanguageRouter, Intent

    router = NaturalLanguageRouter(min_confidence=min_confidence)
    for name, examples in _INTENTS.items():
        router.register(Intent(name=name, examples=examples))
    return router


class _LLMIntegrationBridge:
    """Adapts physml.llm_integration.LLMIntegration to the ClaudeClient interface."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    @property
    def available(self) -> bool:
        return getattr(self._llm, "available", False)

    def tool_call(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        history: Any = None,
    ) -> Any:
        from dataclasses import dataclass, field

        @dataclass
        class _TC:
            tool_calls: list = field(default_factory=list)
            text: str = ""
            available: bool = False
            error: Optional[str] = None

        try:
            result = self._llm.chat(
                user_message=user_message,
                history=history or [],
                system=system,
                tools=tools,
            )
            return _TC(
                tool_calls=result.tool_calls,
                text=result.text,
                available=result.available,
                error=result.error,
            )
        except Exception as exc:
            return _TC(error=str(exc))
