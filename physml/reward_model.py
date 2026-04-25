"""Stage 90 — RewardModel: learn reward functions from demonstrations.

Allows the agent to infer a reward signal from labelled (state, action, reward)
demonstrations rather than relying on a hand-coded reward function.

Classes
-------
RewardSample
    A single demonstration (state, action, observed_reward).
RewardModel
    Fits a regression model on demonstrations and predicts rewards
    for unseen (state, action) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class RewardSample:
    """One demonstration data-point.

    Attributes
    ----------
    state : list[float]
        Feature vector describing the environment state.
    action : float | int
        The action taken.
    reward : float
        The observed reward for (state, action).
    """

    state: List[float]
    action: float
    reward: float


class RewardModel:
    """Learns a reward function from demonstration data.

    Parameters
    ----------
    model : Any, optional
        A scikit-learn regression estimator.  Defaults to
        ``Ridge`` if *None*.

    Attributes
    ----------
    samples_ : list[RewardSample]
        All demonstrations added so far.
    fitted_ : bool
        Whether the model has been fitted at least once.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        if model is None:
            from sklearn.linear_model import Ridge

            model = Ridge()
        self._model = model
        self.samples_: List[RewardSample] = []
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    def add_sample(self, sample: RewardSample) -> None:
        """Append a demonstration to the buffer."""
        self.samples_.append(sample)

    def add_samples(self, samples: List[RewardSample]) -> None:
        """Append multiple demonstrations."""
        self.samples_.extend(samples)

    # ------------------------------------------------------------------
    def fit(self) -> "RewardModel":
        """Fit the reward model on all buffered demonstrations.

        Returns
        -------
        RewardModel
            *self* for chaining.

        Raises
        ------
        ValueError
            If no samples have been added.
        """
        if not self.samples_:
            raise ValueError("No samples to fit on.")
        X, y = self._build_matrix()
        self._model.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, state: List[float], action: float) -> float:
        """Predict the reward for a (state, action) pair.

        Parameters
        ----------
        state : list[float]
            Environment state features.
        action : float
            The action to evaluate.

        Returns
        -------
        float
            Estimated reward.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")
        x = np.array(list(state) + [action], dtype=float).reshape(1, -1)
        return float(self._model.predict(x)[0])

    # ------------------------------------------------------------------
    def _build_matrix(self):
        X = np.array(
            [list(s.state) + [s.action] for s in self.samples_], dtype=float
        )
        y = np.array([s.reward for s in self.samples_], dtype=float)
        return X, y

    def __repr__(self) -> str:  # pragma: no cover
        return f"RewardModel(samples={len(self.samples_)}, fitted={self.fitted_})"
