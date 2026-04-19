"""Stage 136 — PersonalisationManager: explicit + learned user configuration.

Two personalisation paths run simultaneously:

1. **Manual** — user edits ``~/.mycelium/config.json`` or calls
   ``companion.personalise(key, value)`` at any time.  Changes take
   effect immediately without restarting.

2. **Automatic** — every chat, correction, and document ingestion
   updates ``UserProfileLearner``, ``KnowledgeGraph``, and
   ``VectorMemory`` transparently.

This class consolidates both paths and provides a single ``profile()``
view of what the agent knows about the user.

Configurable keys (manual)
--------------------------
- ``name`` — user's name (shown in greetings)
- ``language`` — preferred response language (e.g. ``"en"``, ``"es"``)
- ``verbosity`` — ``"concise"``, ``"normal"``, or ``"verbose"``
- ``agent_name`` — name to give the companion
- ``wake_word`` — voice activation word (default ``"hey myco"``)
- ``timezone`` — e.g. ``"Europe/London"``
- ``notifications`` — ``true``/``false``
- ``watch_dirs`` — list of directories for the file watcher
- ``permissions`` — dict of ``{action: "allow"|"ask"|"deny"}``
- ``theme`` — UI theme for web chat (``"dark"`` or ``"light"``)

Usage
-----
::

    from physml.personalisation import PersonalisationManager

    pm = PersonalisationManager()
    pm.set("name", "Alex")
    pm.set("verbosity", "concise")

    print(pm.get("name"))       # "Alex"
    print(pm.profile())         # full profile dict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "name": None,
    "language": "en",
    "verbosity": "normal",
    "agent_name": "Myco",
    "wake_word": "hey myco",
    "timezone": "UTC",
    "notifications": True,
    "watch_dirs": [],
    "permissions": {},
    "theme": "dark",
}

_VALID_VERBOSITY = {"concise", "normal", "verbose"}


class PersonalisationManager:
    """Unified manual + auto-learned user personalisation.

    Parameters
    ----------
    config_path : str, default "~/.mycelium/config.json"
        Where to persist manual settings.
    user_profile : UserProfileLearner or None
        Auto-learned preference store to read alongside manual config.
    knowledge_graph : KnowledgeGraph or None
        Where auto-extracted facts are stored.
    """

    def __init__(
        self,
        config_path: str = "~/.mycelium/config.json",
        user_profile: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self._path = Path(config_path).expanduser()
        self._config: Dict[str, Any] = dict(_DEFAULTS)
        self._user_profile = user_profile
        self._kg = knowledge_graph
        self._load()

    # ------------------------------------------------------------------
    # Manual config API
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Set a personalisation key and persist immediately."""
        if key == "verbosity" and value not in _VALID_VERBOSITY:
            raise ValueError(f"verbosity must be one of {_VALID_VERBOSITY}")
        self._config[key] = value
        self._save()
        _logger.info("PersonalisationManager: set %s=%r", key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a personalisation value (manual config takes priority)."""
        return self._config.get(key, default)

    def update(self, mapping: Dict[str, Any]) -> None:
        """Batch-update multiple keys."""
        for k, v in mapping.items():
            self.set(k, v)

    def reset(self, key: Optional[str] = None) -> None:
        """Reset one key (or all) to defaults."""
        if key:
            self._config[key] = _DEFAULTS.get(key)
        else:
            self._config = dict(_DEFAULTS)
        self._save()

    # ------------------------------------------------------------------
    # Profile view
    # ------------------------------------------------------------------

    def profile(self) -> Dict[str, Any]:
        """Return a unified profile dict combining manual + auto-learned data."""
        result: Dict[str, Any] = dict(self._config)

        # Overlay auto-learned data from UserProfileLearner
        if self._user_profile is not None:
            try:
                auto = self._user_profile.summary()
                result["learned_topics"] = auto.get("top_topics", [])
                result["interaction_count"] = auto.get("interaction_count", 0)
                result["feedback_score"] = auto.get("feedback_score", 0.0)
                # Use learned name if manual name not set
                if not result.get("name"):
                    prefs = auto.get("preferences", {})
                    result["name"] = prefs.get("name")
            except Exception:
                pass

        # Overlay facts from KnowledgeGraph
        if self._kg is not None:
            try:
                facts = {}
                for node in self._kg.nodes_by_type("user_fact"):
                    props = node.properties
                    pred = props.get("predicate")
                    obj = props.get("object")
                    if pred and obj:
                        facts[pred] = obj
                if facts:
                    result["known_facts"] = facts
                    if not result.get("name") and "name" in facts:
                        result["name"] = facts["name"]
            except Exception:
                pass

        return result

    def greeting(self) -> str:
        """Return a personalised greeting string."""
        name = self.get("name")
        agent = self.get("agent_name", "Myco")
        if name:
            return f"Hello {name}! I'm {agent}."
        return f"Hello! I'm {agent}, your personal AI companion."

    def system_prompt_additions(self) -> str:
        """Return a snippet to inject into the LLM system prompt."""
        parts = []
        p = self.profile()
        if p.get("name"):
            parts.append(f"The user's name is {p['name']}.")
        if p.get("language") and p["language"] != "en":
            parts.append(f"Respond in {p['language']}.")
        if p.get("verbosity") == "concise":
            parts.append("Be very concise — no more than 2-3 sentences per response.")
        elif p.get("verbosity") == "verbose":
            parts.append("Give detailed, thorough explanations.")
        facts = p.get("known_facts", {})
        if facts:
            fact_lines = "; ".join(f"{k}={v}" for k, v in list(facts.items())[:5])
            parts.append(f"Known user facts: {fact_lines}.")
        topics = p.get("learned_topics", [])
        if topics:
            parts.append(f"The user's main interests are: {', '.join(topics[:4])}.")
        return " ".join(parts)

    def keys(self) -> List[str]:
        """All configurable keys."""
        return list(_DEFAULTS.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._config, indent=2))
        except Exception as exc:
            _logger.debug("PersonalisationManager save failed: %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._config.update(data)
        except Exception as exc:
            _logger.debug("PersonalisationManager load failed: %s", exc)
