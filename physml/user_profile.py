"""Stage 113 — UserProfileLearner: persistent user preference learning.

Learns and persists the user's preferences, domain vocabulary, interaction
patterns, and feedback history.  Builds a lightweight user model that
personalises agent responses.  Backed by a JSON file on disk.

Tracks:
* Preferred topics (ranked by frequency).
* Explicit preference key-value pairs.
* Feedback history (positive / negative / correction).
* Time-of-day interaction patterns.

Usage
-----
::

    from physml.user_profile import UserProfileLearner

    profile = UserProfileLearner(profile_path="~/.mycelium/profile.json")
    profile.record_interaction(intent="predict", feedback="positive", topic="sales")
    profile.set_preference("verbosity", "concise")
    profile.get_preference("verbosity")    # → "concise"
    profile.top_topics(n=5)               # → ["sales", ...]
    profile.save()
    profile.load()
    summary = profile.summary()
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


class UserProfileLearner:
    """Persist and learn user preferences and interaction patterns.

    Parameters
    ----------
    profile_path : str, default "~/.mycelium/profile.json"
        Path to the JSON profile file.
    auto_save : bool, default False
        If ``True``, automatically call :meth:`save` after each mutation.
    """

    def __init__(
        self,
        profile_path: str = "~/.mycelium/profile.json",
        auto_save: bool = False,
    ) -> None:
        self.profile_path = Path(profile_path).expanduser()
        self.auto_save = auto_save

        # Internal state
        self._preferences: Dict[str, Any] = {}
        self._topic_counts: Counter = Counter()
        self._feedback_history: List[Dict[str, Any]] = []
        self._time_of_day_counts: Counter = Counter()  # hour → count
        self._interaction_count: int = 0

        # Try to load existing profile
        if self.profile_path.exists():
            try:
                self.load()
            except Exception as e:
                _logger.warning("UserProfileLearner: could not load profile: %s", e)

    # ------------------------------------------------------------------
    # Recording interactions
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        intent: Optional[str] = None,
        feedback: Optional[str] = None,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single user interaction.

        Parameters
        ----------
        intent : str, optional
            The intent/action performed.
        feedback : str, optional
            ``"positive"``, ``"negative"``, or ``"correction"``.
        topic : str, optional
            Domain topic (e.g. ``"sales"``, ``"finance"``).
        metadata : dict, optional
            Arbitrary extra data.
        """
        hour = datetime.now().hour
        self._time_of_day_counts[str(hour)] += 1
        self._interaction_count += 1

        if topic:
            self._topic_counts[topic] += 1
        if intent:
            self._topic_counts[intent] += 1

        if feedback:
            entry: Dict[str, Any] = {
                "timestamp": time.time(),
                "intent": intent,
                "topic": topic,
                "feedback": feedback,
                "metadata": metadata or {},
            }
            self._feedback_history.append(entry)
            # Keep last 500 entries
            if len(self._feedback_history) > 500:
                self._feedback_history = self._feedback_history[-500:]

        if self.auto_save:
            self.save()

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        """Set an explicit user preference.

        Parameters
        ----------
        key : str
            Preference name (e.g. ``"verbosity"``).
        value : any
            Preference value.
        """
        self._preferences[key] = value
        if self.auto_save:
            self.save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a preference value.

        Parameters
        ----------
        key : str
        default : any, optional
            Returned when *key* is not set.

        Returns
        -------
        any
        """
        return self._preferences.get(key, default)

    def remove_preference(self, key: str) -> None:
        """Remove a preference entry."""
        self._preferences.pop(key, None)
        if self.auto_save:
            self.save()

    # ------------------------------------------------------------------
    # Topic analysis
    # ------------------------------------------------------------------

    def top_topics(self, n: int = 5) -> List[str]:
        """Return the *n* most frequently occurring topics.

        Parameters
        ----------
        n : int, default 5

        Returns
        -------
        list of str
        """
        return [t for t, _ in self._topic_counts.most_common(n)]

    def most_active_hour(self) -> Optional[int]:
        """Return the hour of day (0–23) when the user is most active.

        Returns ``None`` if no data.
        """
        if not self._time_of_day_counts:
            return None
        return int(self._time_of_day_counts.most_common(1)[0][0])

    # ------------------------------------------------------------------
    # Feedback analysis
    # ------------------------------------------------------------------

    def feedback_score(self) -> float:
        """Return ratio of positive feedback (0.0–1.0).

        Returns ``0.5`` when no feedback exists.
        """
        if not self._feedback_history:
            return 0.5
        pos = sum(1 for f in self._feedback_history if f.get("feedback") == "positive")
        return pos / len(self._feedback_history)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the profile to disk."""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "preferences": self._preferences,
            "topic_counts": dict(self._topic_counts),
            "feedback_history": self._feedback_history,
            "time_of_day_counts": dict(self._time_of_day_counts),
            "interaction_count": self._interaction_count,
        }
        self.profile_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _logger.info("UserProfileLearner: saved profile to %s", self.profile_path)

    def load(self) -> None:
        """Load the profile from disk."""
        raw = self.profile_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        self._preferences = data.get("preferences", {})
        self._topic_counts = Counter(data.get("topic_counts", {}))
        self._feedback_history = data.get("feedback_history", [])
        self._time_of_day_counts = Counter(data.get("time_of_day_counts", {}))
        self._interaction_count = data.get("interaction_count", 0)
        _logger.info("UserProfileLearner: loaded profile from %s", self.profile_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "interaction_count": self._interaction_count,
            "top_topics": self.top_topics(5),
            "feedback_score": round(self.feedback_score(), 3),
            "most_active_hour": self.most_active_hour(),
            "preferences": dict(self._preferences),
            "n_feedback": len(self._feedback_history),
        }

    def __repr__(self) -> str:
        return (
            f"UserProfileLearner("
            f"interactions={self._interaction_count}, "
            f"topics={len(self._topic_counts)})"
        )
