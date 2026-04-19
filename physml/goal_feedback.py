"""Stage 139 — GoalFeedbackStore: learn from past goal outcomes.

Closes the autonomous learning loop: every completed or failed goal is
persisted as an outcome record.  When GoalEngine encounters a new goal it
queries this store for similar past goals and—if strong matches exist—reuses
their successful step sequences instead of re-running the full decomposer.

This means Myco gets faster and more reliable at goals it has seen before,
without requiring any LLM calls.

Classes
-------
GoalOutcome
    Immutable record of one goal's execution result.
GoalFeedbackStore
    Persist, query, and prune outcome records.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with by from is are was were be been "
    "being have has had do does did will would could should may might shall can".split()
)


def _keywords(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if w not in _STOP_WORDS and len(w) > 2)


def _similarity(a: str, b: str) -> float:
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)  # Jaccard


@dataclass
class GoalOutcome:
    """Record of one goal execution."""

    goal_id: str
    description: str
    status: str                          # "completed" | "failed" | "blocked"
    steps: List[str]                     # ordered step descriptions
    successful_steps: List[str]          # steps that returned status "ok"
    timestamp: float = field(default_factory=time.time)
    elapsed: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoalOutcome":
        return cls(
            goal_id=d["goal_id"],
            description=d["description"],
            status=d["status"],
            steps=d.get("steps", []),
            successful_steps=d.get("successful_steps", []),
            timestamp=d.get("timestamp", 0.0),
            elapsed=d.get("elapsed", 0.0),
            error=d.get("error"),
        )


class GoalFeedbackStore:
    """Persist and query goal execution outcomes.

    Parameters
    ----------
    state_dir : str
        Directory for ``feedback.json``.  Created on first write.
    max_outcomes : int
        Maximum records to keep; oldest are pruned when the cap is exceeded.
    min_similarity : float
        Jaccard threshold for ``find_similar`` to return a match.
    """

    def __init__(
        self,
        state_dir: str = "~/.mycelium/goals",
        max_outcomes: int = 500,
        min_similarity: float = 0.3,
    ) -> None:
        self._state_dir = Path(state_dir).expanduser()
        self.max_outcomes = max_outcomes
        self.min_similarity = min_similarity
        self._outcomes: List[GoalOutcome] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, goal_record: object) -> None:
        """Persist the outcome of a completed/failed GoalRecord.

        Parameters
        ----------
        goal_record : GoalRecord
            The finished goal from GoalEngine.
        """
        try:
            steps = [s.get("description", "") for s in getattr(goal_record, "steps", [])]
            successful = [
                s.get("description", "")
                for s in getattr(goal_record, "steps", [])
                if s.get("status") == "ok"
            ]
            outcome = GoalOutcome(
                goal_id=goal_record.id,
                description=goal_record.description,
                status=goal_record.status.value
                if hasattr(goal_record.status, "value")
                else str(goal_record.status),
                steps=steps,
                successful_steps=successful,
                elapsed=getattr(goal_record, "elapsed", 0.0),
                error=getattr(goal_record, "error", None),
            )
            self._outcomes.append(outcome)
            if len(self._outcomes) > self.max_outcomes:
                self._outcomes = self._outcomes[-self.max_outcomes :]
            self._save()
            _logger.debug(
                "GoalFeedbackStore: recorded outcome for %s (%s)",
                goal_record.id,
                outcome.status,
            )
        except Exception as exc:
            _logger.debug("GoalFeedbackStore: record error: %s", exc)

    def find_similar(
        self,
        description: str,
        n: int = 3,
        status_filter: str = "completed",
    ) -> List[GoalOutcome]:
        """Return up to *n* past outcomes similar to *description*.

        Parameters
        ----------
        description : str
            The new goal description.
        n : int
            Maximum results to return.
        status_filter : str or None
            Only return outcomes with this status.  Pass ``None`` to include all.

        Returns
        -------
        list[GoalOutcome]
            Sorted by similarity descending.
        """
        candidates = [
            o for o in self._outcomes
            if status_filter is None or o.status == status_filter
        ]
        scored = [
            (o, _similarity(description, o.description))
            for o in candidates
        ]
        scored = [(o, s) for o, s in scored if s >= self.min_similarity]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [o for o, _ in scored[:n]]

    def best_steps_for(self, description: str) -> List[str]:
        """Return the successful step sequence from the best matching past goal.

        Returns an empty list if no good match is found.
        """
        matches = self.find_similar(description, n=1)
        if not matches:
            return []
        best = matches[0]
        steps = best.successful_steps or best.steps
        _logger.info(
            "GoalFeedbackStore: reusing %d steps from past goal %s (sim=%.2f)",
            len(steps),
            best.goal_id,
            _similarity(description, best.description),
        )
        return steps

    def stats(self) -> dict:
        """Summary counts by status."""
        from collections import Counter
        counts = Counter(o.status for o in self._outcomes)
        return {
            "total": len(self._outcomes),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "blocked": counts.get("blocked", 0),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            path = self._state_dir / "feedback.json"
            path.write_text(
                json.dumps([o.to_dict() for o in self._outcomes], indent=2)
            )
        except Exception as exc:
            _logger.debug("GoalFeedbackStore: save error: %s", exc)

    def _load(self) -> None:
        path = self._state_dir / "feedback.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._outcomes = [GoalOutcome.from_dict(d) for d in data]
            _logger.debug(
                "GoalFeedbackStore: loaded %d outcomes", len(self._outcomes)
            )
        except Exception as exc:
            _logger.debug("GoalFeedbackStore: load error: %s", exc)
