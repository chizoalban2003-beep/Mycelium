"""physml.llm.action_dispatcher — Wire PromptSystem intents to real physml operations.

:class:`ActionDispatcher` takes a :class:`~physml.llm.prompt_system.PromptAction`
and executes the corresponding physml operation, returning a plain-text response
suitable for display in a REPL or chat interface.

Supported intents
-----------------
train       — fit a MyceliumAgent on a CSV file
predict     — run a prediction with loaded agent
report      — print agent report
help        — list available commands
show_goals  — show goal engine state
add_goal    — queue a new goal
memory      — show conversation history summary
save        — save the current agent to disk
unknown     — fallback message

Usage::

    from physml.llm.action_dispatcher import ActionDispatcher
    from physml.llm import PromptSystem, ClaudeClient

    agent = MyceliumAgent.load("agent.pkl")
    store = ConversationStore("~/.mycelium/conversations/default.json")
    client = ClaudeClient()

    dispatcher = ActionDispatcher(agent=agent, store=store, client=client)
    ps = PromptSystem(client=client)

    action = ps.route("train a model on sales.csv")
    response = dispatcher.dispatch(action)
    print(response)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physml._log import get_logger

if TYPE_CHECKING:
    from physml.llm.prompt_system import PromptAction

_logger = get_logger(__name__)

_HELP_TEXT = """Available commands (plain English or shortcuts):

  train <file.csv>           — fit a model on a CSV file
  predict <values>           — predict with current model, e.g. "predict 1.5 2.3"
  report                     — show model status and statistics
  save                       — save the current model to disk
  show goals / list goals    — show goal engine state
  add goal <description>     — queue a new autonomous goal
  memory / history           — show conversation history summary
  remember that <key>=<val>  — store a user fact (e.g. "remember that name=Alex")
  help                       — show this message

Special REPL commands:
  /history                   — print full conversation history
  /clear                     — clear conversation history
  exit / quit                — exit the REPL
"""


class ActionDispatcher:
    """Dispatch a :class:`PromptAction` to real physml operations.

    Parameters
    ----------
    agent : MyceliumAgent or None
        A trained (or untrained) agent.  When ``None``, train/predict
        operations prompt the user to load one.
    store : ConversationStore or None
        Persistent conversation history, used for the ``memory`` intent.
    client : ClaudeClient or None
        Optional LLM client for generating richer fallback responses.
    agent_path : str or None
        Path to save/load the agent (default: ``"agent.pkl"``).
    user_memory : UserMemory or None
        Persistent user facts store.  Created automatically when ``None``.
    """

    def __init__(
        self,
        agent: Any = None,
        store: Any = None,
        client: Any = None,
        agent_path: str = "agent.pkl",
        user_memory: Any = None,
    ) -> None:
        self.agent = agent
        self.store = store
        self.client = client
        self.agent_path = agent_path
        if user_memory is not None:
            self.user_memory = user_memory
        else:
            from physml.llm.memory_store import UserMemory
            self.user_memory = UserMemory()

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def dispatch(self, action: "PromptAction") -> str:
        """Execute *action* and return a plain-text response.

        Parameters
        ----------
        action : PromptAction
            Structured intent + payload from :class:`~physml.llm.prompt_system.PromptSystem`.

        Returns
        -------
        str
        """
        intent = action.intent
        payload = action.payload or {}

        try:
            if intent == "train":
                return self._do_train(payload, action.raw_text)
            elif intent == "predict":
                return self._do_predict(payload, action.raw_text)
            elif intent == "report":
                return self._do_report()
            elif intent == "help":
                return _HELP_TEXT.strip()
            elif intent == "show_goals":
                return self._do_show_goals()
            elif intent == "add_goal":
                return self._do_add_goal(payload, action.raw_text)
            elif intent == "memory":
                return self._do_memory()
            elif intent == "save":
                return self._do_save()
            elif intent == "remember":
                return self._handle_remember(payload, action.raw_text)
            else:
                return self._do_unknown(action)
        except Exception as exc:
            _logger.warning("ActionDispatcher.dispatch error: %s", exc)
            return f"Error executing {intent!r}: {exc}"

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _do_train(self, payload: dict, raw_text: str) -> str:
        path = payload.get("path") or payload.get("paths", [None])[0] if payload.get("paths") else None

        if not path:
            # Try to extract from raw text with a simple heuristic
            import re
            m = re.search(r"[\w./~-]+\.csv", raw_text)
            if m:
                path = m.group(0)

        if not path:
            return (
                "I need a CSV file path to train on.\n"
                "Example: 'train on sales.csv' or 'train on /data/train.csv'"
            )

        try:
            import pandas as pd
            from pathlib import Path

            p = Path(path)
            if not p.exists():
                return f"File not found: {path!r}. Please provide a valid CSV path."

            df = pd.read_csv(p)
            if df.empty:
                return f"CSV file {path!r} is empty."

            # Infer target column: last column by default
            target = payload.get("target_column") or df.columns[-1]
            if target not in df.columns:
                return (
                    f"Target column {target!r} not found. "
                    f"Available columns: {list(df.columns)}"
                )

            y = df[target].to_numpy()
            X = df.drop(columns=[target]).to_numpy(dtype=float)

            from physml.mycelium_agent import MyceliumAgent
            self.agent = MyceliumAgent()
            self.agent.fit(X, y)

            n_samples, n_features = X.shape
            return (
                f"Trained on {path!r}: {n_samples} samples, {n_features} features, "
                f"target={target!r}.\n"
                f"Agent is ready. Try: 'predict <values>' or 'report'"
            )
        except ImportError:
            return "pandas is required for CSV training: pip install pandas"
        except Exception as exc:
            return f"Training failed: {exc}"

    def _do_predict(self, payload: dict, raw_text: str) -> str:
        if self.agent is None:
            return (
                "No agent loaded. Train one first with: 'train on <file.csv>'"
            )

        numbers = payload.get("numbers") or []

        if not numbers:
            # Try to parse from raw text
            import re
            found = re.findall(r"-?\d+(?:\.\d+)?", raw_text)
            numbers = [float(x) for x in found] if found else []

        if not numbers:
            return (
                "I need numeric feature values to predict.\n"
                "Example: 'predict 1.5 2.3 -0.7'"
            )

        try:
            import numpy as np
            X = np.array(numbers, dtype=float).reshape(1, -1)
            action = self.agent.observe(X)
            conf_pct = f"{action.confidence:.0%}" if action.confidence is not None else "?"
            pred = action.prediction
            uncertain = " (uncertain — consider providing a label)" if action.action == "ask" else ""
            return (
                f"Prediction: {pred!r}  confidence: {conf_pct}{uncertain}\n"
                f"Features used: {numbers}"
            )
        except Exception as exc:
            return f"Prediction failed: {exc}"

    def _do_report(self) -> str:
        if self.agent is None:
            return "No agent loaded. Train one first with: 'train on <file.csv>'"
        try:
            report = self.agent.report()
            lines = []
            for k, v in report.items():
                if isinstance(v, dict):
                    lines.append(f"{k}:")
                    for kk, vv in v.items():
                        lines.append(f"  {kk}: {vv}")
                else:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines) if lines else "Agent report: (no data)"
        except Exception as exc:
            return f"Report error: {exc}"

    def _do_show_goals(self) -> str:
        try:
            from physml.goal_engine import GoalEngine  # noqa: F401
            return "Goal engine not attached to this session. Use MyceliumCompanion.goals() for full goal tracking."
        except Exception:
            return "Goal engine not available in this session."

    def _do_add_goal(self, payload: dict, raw_text: str) -> str:
        desc = payload.get("goal_description") or raw_text
        return (
            f"Goal noted: {desc!r}\n"
            f"(Full goal execution requires MyceliumCompanion. "
            f"Use companion.add_goal({desc!r}) for autonomous execution.)"
        )

    def _do_memory(self) -> str:
        lines = []

        # Show user facts first
        mem_text = self.user_memory.inject_into_prompt()
        if mem_text:
            lines.append(mem_text)
        else:
            lines.append("No user facts stored yet. Use 'remember that name=Alex' to store facts.")

        if self.store is None:
            lines.append("\nNo conversation store attached to this session.")
            return "\n".join(lines)
        try:
            s = self.store.summary()
            total = s.get("total_turns", 0)
            user_t = s.get("user_turns", 0)
            asst_t = s.get("assistant_turns", 0)
            intents = s.get("intents", {})
            lines.append(
                f"\nConversation history: {total} turns ({user_t} user, {asst_t} assistant)"
            )
            if intents:
                top = sorted(intents.items(), key=lambda x: -x[1])[:5]
                lines.append("Top intents: " + ", ".join(f"{k}({v})" for k, v in top))
            if total > 0:
                recent = list(self.store)[-3:]
                lines.append("Recent turns:")
                for t in recent:
                    role = t.get("role", "?")
                    content = t.get("content", "")[:80]
                    lines.append(f"  {role}: {content}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Memory summary error: {exc}"

    def _handle_remember(self, payload: dict, raw_text: str) -> str:
        """Store a user fact from a 'remember that' intent.

        Parses key=value pairs from payload or raw_text.
        Also handles natural phrasing like "my name is Alex" or "call me Bob".
        """
        import re

        # Try payload kv first
        kv = payload.get("kv") or {}
        if isinstance(kv, dict) and kv:
            for k, v in kv.items():
                self.user_memory.remember(str(k), str(v))
            stored = ", ".join(f"{k}={v}" for k, v in kv.items())
            return f"Got it! I'll remember: {stored}"

        # Try key=value pattern in raw text
        m = re.search(r"(\w[\w\s]*)=([^,\n]+)", raw_text)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            self.user_memory.remember(key, value)
            return f"Got it! I'll remember that {key} = {value}"

        # Try "my name is X" / "call me X"
        name_m = re.search(
            r"(?:my name is|call me|i am|i'm)\s+([A-Za-z][\w\s]*)", raw_text, re.IGNORECASE
        )
        if name_m:
            name = name_m.group(1).strip()
            self.user_memory.remember("name", name)
            return f"Got it! I'll remember that your name is {name}."

        # Generic: store the whole text under a "note" key with index
        existing = self.user_memory.summary()
        idx = sum(1 for k in existing if k.startswith("note"))
        key = f"note{idx + 1}" if idx > 0 else "note"
        self.user_memory.remember(key, raw_text)
        return f"Noted and stored: {raw_text!r}"

    def _do_save(self) -> str:
        if self.agent is None:
            return "No agent to save. Train one first with: 'train on <file.csv>'"
        try:
            self.agent.save(self.agent_path)
            return f"Agent saved to {self.agent_path!r}"
        except Exception as exc:
            return f"Save failed: {exc}"

    def _do_unknown(self, action: "PromptAction") -> str:
        if self.client is not None and getattr(self.client, "available", False):
            try:
                history = []
                if self.store is not None:
                    history = self.store.to_messages(max_turns=10)
                result = self.client.chat(action.raw_text, history=history)
                if result.text:
                    return result.text
            except Exception as exc:
                _logger.debug("ActionDispatcher LLM fallback error: %s", exc)

        from physml.llm.prompt_system import PromptSystem
        ps = PromptSystem()
        desc = ps.describe_intent(action.intent)
        return (
            f"I understood your request as: {desc!r} "
            f"(confidence={action.confidence:.2f})\n"
            f"Type 'help' to see what I can do."
        )

    def __repr__(self) -> str:
        has_agent = self.agent is not None
        has_store = self.store is not None
        n_facts = len(self.user_memory.summary())
        return f"ActionDispatcher(agent={has_agent}, store={has_store}, user_facts={n_facts})"
