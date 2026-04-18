"""Stage 119 — DigitalSoul: agent identity and personality layer.

The agent's persistent "self".  Stores:
* Agent name and persona.
* Creation date and cumulative learning stats.
* Life events log (``"first prediction"``, ``"learned from 500 samples"``).
* Goals and values (user-defined).
* A mood / homeostasis state that influences response style.

This is what makes the agent feel like *your* companion.

Usage
-----
::

    from physml.digital_soul import DigitalSoul

    soul = DigitalSoul(soul_path="~/.mycelium/soul.json")
    soul.name = "Myco"
    soul.record_event("first_prediction", details={"target": "sales"})
    print(soul.stats)        # {"total_predictions": 1, "days_alive": 0, ...}
    print(soul.mood)         # "curious" | "confident" | "uncertain" | "learning"
    soul.update_mood(homeostasis_score=0.8)
    print(soul.life_story())
    soul.save()
    soul.load()
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class LifeEvent:
    """A single logged event in the agent's life story.

    Attributes
    ----------
    event_type : str
        Short event identifier (e.g. ``"first_prediction"``).
    timestamp : float
        Unix time of the event.
    details : dict
        Arbitrary event-specific data.
    description : str
        Optional human-readable description.
    """

    event_type: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


_MOOD_TABLE = {
    # homeostasis_score range → mood
    (0.8, 1.0): "confident",
    (0.6, 0.8): "curious",
    (0.4, 0.6): "learning",
    (0.0, 0.4): "uncertain",
}


def _score_to_mood(score: float) -> str:
    for (lo, hi), mood in _MOOD_TABLE.items():
        if lo <= score <= hi:
            return mood
    return "uncertain"


class DigitalSoul:
    """Persistent agent identity and personality.

    Parameters
    ----------
    soul_path : str, default "~/.mycelium/soul.json"
        Path to the JSON soul file.
    name : str, default "Myco"
        Agent name.
    """

    def __init__(
        self,
        soul_path: str = "~/.mycelium/soul.json",
        name: str = "Myco",
    ) -> None:
        self.soul_path = Path(soul_path).expanduser()
        self._name: str = name
        self._created_at: float = time.time()
        self._events: List[LifeEvent] = []
        self._goals: List[str] = []
        self._values: List[str] = []
        self._mood: str = "curious"
        self._homeostasis_score: float = 0.7
        self._stats: Dict[str, Any] = {
            "total_predictions": 0,
            "total_training_rounds": 0,
            "total_interactions": 0,
        }

        # Try loading existing soul
        if self.soul_path.exists():
            try:
                self.load()
            except Exception as e:
                _logger.warning("DigitalSoul: could not load soul: %s", e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def mood(self) -> str:
        """Current mood: ``"confident"``, ``"curious"``, ``"learning"``, or ``"uncertain"``."""
        return self._mood

    @property
    def stats(self) -> Dict[str, Any]:
        """Cumulative statistics."""
        created = datetime.fromtimestamp(self._created_at, tz=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        days = (now - created).days
        return {**self._stats, "days_alive": days}

    @property
    def events(self) -> List[LifeEvent]:
        """All recorded life events."""
        return list(self._events)

    @property
    def goals(self) -> List[str]:
        return list(self._goals)

    @property
    def values(self) -> List[str]:
        return list(self._values)

    # ------------------------------------------------------------------
    # Events and mood
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> LifeEvent:
        """Log a life event.

        Parameters
        ----------
        event_type : str
        details : dict, optional
        description : str, optional

        Returns
        -------
        LifeEvent
        """
        event = LifeEvent(
            event_type=event_type,
            details=details or {},
            description=description,
        )
        self._events.append(event)

        # Update stats
        if "prediction" in event_type:
            self._stats["total_predictions"] = self._stats.get("total_predictions", 0) + 1
        if "train" in event_type or "learn" in event_type:
            self._stats["total_training_rounds"] = self._stats.get("total_training_rounds", 0) + 1
        self._stats["total_interactions"] = self._stats.get("total_interactions", 0) + 1

        return event

    def update_mood(self, homeostasis_score: float) -> str:
        """Update the mood based on a homeostasis score.

        Parameters
        ----------
        homeostasis_score : float
            Score in [0, 1].  Higher = more stable / confident.

        Returns
        -------
        str
            New mood string.
        """
        self._homeostasis_score = max(0.0, min(1.0, float(homeostasis_score)))
        self._mood = _score_to_mood(self._homeostasis_score)
        return self._mood

    def set_goal(self, goal: str) -> None:
        """Add a goal string."""
        if goal not in self._goals:
            self._goals.append(goal)

    def set_value(self, value: str) -> None:
        """Add a value string."""
        if value not in self._values:
            self._values.append(value)

    def increment_stat(self, key: str, amount: int = 1) -> None:
        """Increment a named statistic.

        Parameters
        ----------
        key : str
        amount : int, default 1
        """
        self._stats[key] = self._stats.get(key, 0) + amount

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------

    def life_story(self, max_events: int = 20) -> str:
        """Generate a human-readable narrative of the agent's life.

        Parameters
        ----------
        max_events : int, default 20

        Returns
        -------
        str
        """
        created = datetime.fromtimestamp(self._created_at).strftime("%Y-%m-%d")
        s = self.stats
        lines = [
            f"My name is {self._name}.",
            f"I was created on {created} and I am {s['days_alive']} day(s) old.",
            f"Current mood: {self._mood} (homeostasis: {self._homeostasis_score:.2f}).",
            f"I have made {s.get('total_predictions', 0)} predictions and "
            f"completed {s.get('total_training_rounds', 0)} training round(s).",
        ]
        if self._goals:
            lines.append("Goals: " + "; ".join(self._goals[:5]) + ".")
        if self._values:
            lines.append("Values: " + "; ".join(self._values[:5]) + ".")
        if self._events:
            lines.append("")
            lines.append("Key life events:")
            for ev in self._events[-max_events:]:
                ts = datetime.fromtimestamp(ev.timestamp).strftime("%Y-%m-%d %H:%M")
                desc = ev.description or ev.event_type
                lines.append(f"  [{ts}] {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the soul to disk."""
        self.soul_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self._name,
            "created_at": self._created_at,
            "mood": self._mood,
            "homeostasis_score": self._homeostasis_score,
            "stats": self._stats,
            "goals": self._goals,
            "values": self._values,
            "events": [asdict(e) for e in self._events],
        }
        self.soul_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _logger.info("DigitalSoul: saved to %s", self.soul_path)

    def load(self) -> None:
        """Load the soul from disk."""
        raw = self.soul_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        self._name = data.get("name", self._name)
        self._created_at = data.get("created_at", self._created_at)
        self._mood = data.get("mood", "curious")
        self._homeostasis_score = data.get("homeostasis_score", 0.7)
        self._stats = data.get("stats", self._stats)
        self._goals = data.get("goals", [])
        self._values = data.get("values", [])
        self._events = [
            LifeEvent(
                event_type=e["event_type"],
                timestamp=e.get("timestamp", 0.0),
                details=e.get("details", {}),
                description=e.get("description", ""),
            )
            for e in data.get("events", [])
        ]
        _logger.info("DigitalSoul: loaded from %s", self.soul_path)

    def summary(self) -> Dict[str, Any]:
        """Return a brief summary dict."""
        return {
            "name": self._name,
            "mood": self._mood,
            "stats": self.stats,
            "n_events": len(self._events),
            "goals": self._goals[:5],
        }

    def __repr__(self) -> str:
        return (
            f"DigitalSoul("
            f"name={self._name!r}, "
            f"mood={self._mood!r}, "
            f"events={len(self._events)})"
        )
