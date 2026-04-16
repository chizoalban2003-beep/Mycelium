"""Stage 15 — Contextual bandit policy for ask/predict decisions.

:class:`ContextualBandit` replaces the heuristic adaptive-threshold with a
small logistic-regression policy that learns *when* to ask vs predict from
experience.

The bandit observes a feature vector ``x`` (the input sample concatenated with
a homeostasis score) and outputs an ask-probability.  After each labelling
event the policy is updated using a REINFORCE-style reward signal:

    reward = accuracy_improvement_after_labelling

The logistic-regression policy is updated incrementally with scikit-learn's
``SGDClassifier`` (``partial_fit``).

Usage
-----
Direct use is optional — :class:`~physml.mycelium_agent.MyceliumAgent` enables
the bandit via ``policy="bandit"``:

::

    from physml import myco
    agent = myco(policy="bandit")
    agent.fit(X_seed, y_seed)
    action = agent.observe(X_new)

Alternatively use the bandit directly:

::

    from physml.bandit import ContextualBandit
    bandit = ContextualBandit(n_features=5)
    prob = bandit.ask_probability(x, homeostasis=0.7)
    # ... after observing outcome ...
    bandit.update(x, homeostasis=0.7, reward=0.1)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ContextualBandit:
    """Logistic-regression contextual bandit for ask/predict decisions.

    The policy maintains a binary ``SGDClassifier`` where:
    * Class 0 = predict (don't ask)
    * Class 1 = ask

    The feature vector fed to the bandit is
    ``[x_normalised..., homeostasis]``.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector.
    learning_rate : float, default 0.01
        SGD step size.
    exploration_rate : float, default 0.1
        Probability of random exploration (epsilon-greedy).
    warm_ask_threshold : float, default 0.5
        Initial fallback ask-probability before the bandit has been
        trained (used for the first ``min_samples`` steps).
    min_samples : int, default 5
        Minimum number of update calls before the bandit's predictions
        are used (warm-up period uses ``warm_ask_threshold``).
    """

    def __init__(
        self,
        n_features: int,
        *,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
        warm_ask_threshold: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.exploration_rate = float(exploration_rate)
        self.warm_ask_threshold = float(warm_ask_threshold)
        self.min_samples = int(min_samples)
        self._n_updates: int = 0
        self._clf: Any = None
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask_probability(self, x: np.ndarray, homeostasis: float) -> float:
        """Return the probability that the bandit recommends asking for a label.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,) or (1, n_features)
        homeostasis : float in [0, 1]

        Returns
        -------
        float in [0, 1]
        """
        if self._n_updates < self.min_samples or self._clf is None:
            return self.warm_ask_threshold

        # Epsilon-greedy exploration
        if self._rng.random() < self.exploration_rate:
            return float(self._rng.random())

        feat = self._featurise(x, homeostasis)
        try:
            proba = self._clf.predict_proba(feat.reshape(1, -1))
            # Class 1 = ask
            classes = list(self._clf.classes_)
            ask_idx = classes.index(1) if 1 in classes else -1
            return float(proba[0, ask_idx]) if ask_idx >= 0 else 0.5
        except Exception:
            return self.warm_ask_threshold

    def update(
        self,
        x: np.ndarray,
        homeostasis: float,
        reward: float,
        asked: bool = True,
    ) -> None:
        """Update the bandit policy with an observed reward.

        Parameters
        ----------
        x : np.ndarray
        homeostasis : float
        reward : float
            Positive reward signals that asking was correct; negative that
            asking was unnecessary.  Typical value: accuracy improvement in
            [-1, 1].
        asked : bool
            Whether the "ask" action was taken.
        """
        from sklearn.linear_model import SGDClassifier  # type: ignore

        if self._clf is None:
            self._clf = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=self.learning_rate,
                random_state=0,
                max_iter=1,
                tol=None,
            )

        feat = self._featurise(x, homeostasis).reshape(1, -1)
        # REINFORCE: label = 1 (ask) if reward > 0 and we asked,
        #            label = 0 (predict) otherwise
        label = int(asked and reward > 0)
        # Weight by absolute reward magnitude
        sample_weight = np.array([max(0.01, abs(reward))])
        try:
            self._clf.partial_fit(feat, [label], classes=[0, 1], sample_weight=sample_weight)
        except Exception:
            pass
        self._n_updates += 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _featurise(self, x: np.ndarray, homeostasis: float) -> np.ndarray:
        """Concatenate (normalised) x with homeostasis score."""
        x_flat = np.asarray(x, dtype=float).ravel()
        # Normalise each feature to [-1, 1] range (simple clip + scale)
        x_norm = np.clip(x_flat, -5.0, 5.0) / 5.0
        return np.append(x_norm, float(homeostasis))
