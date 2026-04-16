"""Stage 65 — GoalConditionedPolicy.

Allows the agent to condition its behaviour on a structured or
natural-language goal description.  Goals are encoded as fixed-length
embeddings (hashed bag-of-words for zero-dependency portability) and
appended to state vectors before prediction.  The policy learns which
actions best satisfy each goal type from labelled experience.

Classes
-------
GoalSpec
    Dataclass encoding a structured goal.
GoalConditionedPolicy
    Maps (state, goal) → preferred action index using a lightweight
    online multi-class classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier


@dataclass
class GoalSpec:
    """A structured goal specification.

    Parameters
    ----------
    description : str
        Human-readable goal text.
    target_metric : str
        The metric to optimise (e.g. ``"accuracy"``, ``"reward"``,
        ``"f1"``).
    threshold : float
        Minimum acceptable value of *target_metric* to consider the goal
        achieved.
    extra : dict
        Optional additional constraints.
    """

    description: str
    target_metric: str = "accuracy"
    threshold: float = 0.8
    extra: dict[str, Any] = field(default_factory=dict)

    def achieved(self, metrics: dict[str, float]) -> bool:
        """Return True if the goal threshold is met."""
        return metrics.get(self.target_metric, 0.0) >= self.threshold


class GoalConditionedPolicy:
    """Goal-conditioned action policy.

    Encodes a goal as a hashed embedding and appends it to the state
    before scoring candidate actions with an online SGD classifier.

    Parameters
    ----------
    n_actions : int
        Number of available discrete actions.
    embedding_dim : int
        Dimension of the goal embedding vector.
    """

    def __init__(self, n_actions: int = 2, embedding_dim: int = 16) -> None:
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self._clf = SGDClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            n_jobs=1,
        )
        self._fitted = False
        self._classes = np.arange(n_actions)

    # ------------------------------------------------------------------
    # Goal encoding
    # ------------------------------------------------------------------

    def encode_goal(self, goal: GoalSpec | str) -> np.ndarray:
        """Convert a goal to a fixed-length embedding vector.

        Uses a deterministic hashed bag-of-words approach so no NLP
        library is required.
        """
        text = goal.description if isinstance(goal, GoalSpec) else str(goal)
        vec = np.zeros(self.embedding_dim, dtype=np.float64)
        for i, word in enumerate(text.lower().split()):
            idx = hash(word) % self.embedding_dim
            vec[idx] += 1.0 / (i + 1)
        # Normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        # Append threshold if GoalSpec
        if isinstance(goal, GoalSpec):
            vec[0] = goal.threshold
        return vec

    def _make_input(
        self, state: np.ndarray, goal: GoalSpec | str
    ) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).ravel()
        g = self.encode_goal(goal)
        return np.concatenate([s, g]).reshape(1, -1)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, goal: GoalSpec | str) -> int:
        """Return the recommended action for a (state, goal) pair.

        Falls back to action 0 if the policy has not been fitted yet.
        """
        if not self._fitted:
            return 0
        x = self._make_input(state, goal)
        return int(self._clf.predict(x)[0])

    def update(
        self,
        state: np.ndarray,
        goal: GoalSpec | str,
        action: int,
    ) -> "GoalConditionedPolicy":
        """Online update from a (state, goal) → action example."""
        x = self._make_input(state, goal)
        self._clf.partial_fit(x, [action], classes=self._classes)
        self._fitted = True
        return self

    def action_scores(
        self, state: np.ndarray, goal: GoalSpec | str
    ) -> np.ndarray:
        """Return log-probability scores for all actions."""
        if not self._fitted:
            return np.zeros(self.n_actions)
        x = self._make_input(state, goal)
        return self._clf.predict_log_proba(x)[0]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_actions": self.n_actions,
            "embedding_dim": self.embedding_dim,
        }
