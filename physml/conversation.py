"""Stage 108 — ConversationManager: multi-turn dialogue tracker.

Maintains conversation history with speaker attribution, context-window
management (keep last N turns), topic tracking, and session serialization.
Integrates with NaturalLanguageRouter to route commands and track intents.

Usage
-----
::

    from physml.conversation import ConversationManager

    mgr = ConversationManager(max_history=50)
    mgr.add_turn("user", "predict sales for next week")
    mgr.add_turn("agent", "I predict 1250 units with 85% confidence")
    ctx = mgr.context()   # recent turns as text
    intent = mgr.last_intent
    mgr.save("convo.json")
    mgr.load("convo.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class Turn:
    """A single conversation turn.

    Attributes
    ----------
    speaker : str
        ``"user"`` or ``"agent"`` (or any label).
    text : str
        The utterance.
    timestamp : float
        Unix time of the turn.
    intent : str or None
        Routed intent, if resolved.
    entities : dict
        Extracted entities (from NaturalLanguageRouter).
    metadata : dict
        Arbitrary extra data.
    """

    speaker: str
    text: str
    timestamp: float = field(default_factory=time.time)
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Multi-turn dialogue tracker.

    Parameters
    ----------
    max_history : int, default 50
        Maximum number of turns to keep in memory.
    router : NaturalLanguageRouter or None
        When provided, user turns are automatically routed to extract
        intent and entities.
    context_turns : int, default 10
        Number of most-recent turns included in :meth:`context`.
    """

    def __init__(
        self,
        max_history: int = 50,
        router: Any = None,
        context_turns: int = 10,
    ) -> None:
        self.max_history = int(max_history)
        self.router = router
        self.context_turns = int(context_turns)
        self._turns: List[Turn] = []
        self._topics: Dict[str, int] = {}
        self._last_intent: Optional[str] = None
        self._session_id: str = str(int(time.time()))

    # ------------------------------------------------------------------
    # Adding turns
    # ------------------------------------------------------------------

    def add_turn(
        self,
        speaker: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Turn:
        """Record a new turn.

        Parameters
        ----------
        speaker : str
            ``"user"``, ``"agent"``, or any label.
        text : str
        metadata : dict, optional

        Returns
        -------
        Turn
        """
        intent = None
        entities: Dict[str, Any] = {}

        if speaker == "user" and self.router is not None:
            try:
                result = self.router.route(text)
                intent = result.intent
                entities = result.entities
                self._last_intent = intent
                # track topic from intent
                if intent and intent != "unknown":
                    self._topics[intent] = self._topics.get(intent, 0) + 1
            except Exception as e:
                _logger.warning("ConversationManager: routing failed: %s", e)

        turn = Turn(
            speaker=speaker,
            text=text,
            timestamp=time.time(),
            intent=intent,
            entities=entities,
            metadata=metadata or {},
        )
        self._turns.append(turn)

        # Evict oldest turns if over limit
        if len(self._turns) > self.max_history:
            self._turns = self._turns[-self.max_history:]

        return turn

    # ------------------------------------------------------------------
    # Context and state
    # ------------------------------------------------------------------

    def context(self, n: Optional[int] = None) -> str:
        """Return the last *n* turns as a formatted text block.

        Parameters
        ----------
        n : int, optional
            Number of turns to include.  Defaults to ``context_turns``.

        Returns
        -------
        str
        """
        n = n or self.context_turns
        recent = self._turns[-n:] if n else self._turns
        lines = [f"[{t.speaker}] {t.text}" for t in recent]
        return "\n".join(lines)

    @property
    def last_intent(self) -> Optional[str]:
        """The most recently resolved intent (``None`` if no routing yet)."""
        return self._last_intent

    @property
    def turns(self) -> List[Turn]:
        """All stored turns (read-only view)."""
        return list(self._turns)

    def top_topics(self, n: int = 5) -> List[str]:
        """Return the *n* most frequently occurring intents/topics.

        Parameters
        ----------
        n : int, default 5

        Returns
        -------
        list of str
        """
        sorted_topics = sorted(self._topics.items(), key=lambda kv: -kv[1])
        return [t for t, _ in sorted_topics[:n]]

    def clear(self) -> None:
        """Clear conversation history (keeps session ID)."""
        self._turns.clear()
        self._topics.clear()
        self._last_intent = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist conversation to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        data = {
            "session_id": self._session_id,
            "max_history": self.max_history,
            "context_turns": self.context_turns,
            "topics": self._topics,
            "turns": [asdict(t) for t in self._turns],
        }
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _logger.info("ConversationManager: saved %d turns to %s", len(self._turns), p)

    def load(self, path: str) -> None:
        """Load a conversation from a JSON file.

        Parameters
        ----------
        path : str
            Source file path.
        """
        p = Path(path).expanduser()
        data = json.loads(p.read_text(encoding="utf-8"))
        self._session_id = data.get("session_id", self._session_id)
        self.max_history = data.get("max_history", self.max_history)
        self.context_turns = data.get("context_turns", self.context_turns)
        self._topics = data.get("topics", {})
        self._turns = [
            Turn(
                speaker=t["speaker"],
                text=t["text"],
                timestamp=t.get("timestamp", 0.0),
                intent=t.get("intent"),
                entities=t.get("entities", {}),
                metadata=t.get("metadata", {}),
            )
            for t in data.get("turns", [])
        ]
        _logger.info("ConversationManager: loaded %d turns from %s", len(self._turns), p)

    def summary(self) -> Dict[str, Any]:
        """Return a brief stats summary."""
        return {
            "session_id": self._session_id,
            "n_turns": len(self._turns),
            "last_intent": self._last_intent,
            "top_topics": self.top_topics(3),
        }

    def __repr__(self) -> str:
        return (
            f"ConversationManager("
            f"n_turns={len(self._turns)}, "
            f"max_history={self.max_history})"
        )
