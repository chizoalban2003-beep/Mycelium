"""Stage 80 — ActiveLearner: uncertainty-based active learning.

Selects the most informative unlabelled samples for human annotation using
four pluggable query strategies:

* ``"least_confident"``  — picks the sample whose predicted probability for
  the most-likely class is *lowest*.
* ``"margin"``           — picks the sample with the smallest gap between
  the top-two predicted probabilities.
* ``"entropy"``          — picks the sample with the highest Shannon entropy
  over the predicted probability distribution.
* ``"qbc"``              — Query-By-Committee: trains a small ensemble of
  sub-models on the labelled pool and selects the sample with the highest
  disagreement (vote entropy) across committee members.

Classes
-------
QueryResult
    Metadata about one active-learning query round.
ActiveLearner
    Iterative active learner with pluggable query strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QueryResult:
    """Metadata about one active-learning query round.

    Attributes
    ----------
    strategy : str
        Name of the query strategy used.
    query_indices : list[int]
        Indices (into the unlabelled pool) of the selected samples.
    scores : list[float]
        Informativeness score for each selected sample.
    n_labelled : int
        Total number of labelled samples *after* this round.
    n_unlabelled : int
        Number of unlabelled samples remaining after this round.
    elapsed_s : float
        Wall-clock duration of the query call.
    """

    strategy: str
    query_indices: list[int]
    scores: list[float]
    n_labelled: int
    n_unlabelled: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query_indices": self.query_indices,
            "n_selected": len(self.query_indices),
            "n_labelled": self.n_labelled,
            "n_unlabelled": self.n_unlabelled,
            "elapsed_s": round(self.elapsed_s, 4),
        }


class ActiveLearner:
    """Iterative active learner with pluggable query strategies.

    Parameters
    ----------
    estimator : Any
        A sklearn-compatible estimator with ``fit(X, y)`` and
        ``predict_proba(X)``.
    strategy : str, default ``"entropy"``
        One of ``"least_confident"``, ``"margin"``, ``"entropy"``,
        ``"qbc"``.
    n_query : int, default 10
        Number of samples to select per query round.
    committee_size : int, default 5
        Number of sub-models for the ``"qbc"`` strategy.  Each sub-model
        is trained on a bootstrap resample of the labelled pool.
    random_state : int, default 0

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from physml.active_learner import ActiveLearner
    >>> X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    >>> learner = ActiveLearner(LogisticRegression(max_iter=300), strategy="entropy")
    >>> learner.initialize(X[:50], y[:50], X[50:])
    >>> result = learner.query()
    >>> len(result.query_indices) == 10
    True
    """

    _STRATEGIES = frozenset({"least_confident", "margin", "entropy", "qbc"})

    def __init__(
        self,
        estimator: Any,
        *,
        strategy: str = "entropy",
        n_query: int = 10,
        committee_size: int = 5,
        random_state: int = 0,
    ) -> None:
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(self._STRATEGIES)}, got {strategy!r}"
            )
        self.estimator = estimator
        self.strategy = strategy
        self.n_query = int(n_query)
        self.committee_size = int(committee_size)
        self.random_state = int(random_state)

        self._rng = np.random.default_rng(random_state)
        self._X_labelled: np.ndarray | None = None
        self._y_labelled: np.ndarray | None = None
        self._X_pool: np.ndarray | None = None
        self._history: list[QueryResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(
        self,
        X_labelled: Any,
        y_labelled: Any,
        X_pool: Any,
    ) -> None:
        """Seed the learner with an initial labelled set and unlabelled pool.

        Parameters
        ----------
        X_labelled : array-like of shape (n_labelled, n_features)
        y_labelled : array-like of shape (n_labelled,)
        X_pool : array-like of shape (n_pool, n_features)
            Unlabelled candidate samples.
        """
        self._X_labelled = np.asarray(X_labelled, dtype=float)
        self._y_labelled = np.asarray(y_labelled)
        self._X_pool = np.asarray(X_pool, dtype=float)
        self._fit_estimator()

    def query(self) -> QueryResult:
        """Select the next batch of informative samples from the pool.

        Returns
        -------
        QueryResult
        """
        self._check_initialised()
        t0 = time.time()

        scores = self._score_pool()
        n_select = min(self.n_query, len(self._X_pool))
        # Largest scores = most informative
        idx = np.argsort(scores)[::-1][:n_select]
        selected_scores = scores[idx].tolist()

        result = QueryResult(
            strategy=self.strategy,
            query_indices=idx.tolist(),
            scores=[round(s, 4) for s in selected_scores],
            n_labelled=len(self._X_labelled),
            n_unlabelled=len(self._X_pool) - n_select,
            elapsed_s=time.time() - t0,
        )
        self._history.append(result)
        return result

    def update(self, indices: list[int], y_new: Any) -> None:
        """Add newly labelled samples from the pool to the labelled set.

        Parameters
        ----------
        indices : list[int]
            Indices returned by :meth:`query`.
        y_new : array-like of shape (len(indices),)
            True labels for the selected pool samples.
        """
        self._check_initialised()
        idx = np.asarray(indices, dtype=int)
        y_new = np.asarray(y_new)

        X_new = self._X_pool[idx]
        mask = np.ones(len(self._X_pool), dtype=bool)
        mask[idx] = False

        self._X_labelled = np.vstack([self._X_labelled, X_new])
        self._y_labelled = np.concatenate([self._y_labelled, y_new])
        self._X_pool = self._X_pool[mask]

        self._fit_estimator()

    def score(self, X: Any, y: Any) -> float:
        """Return accuracy of the current model on *(X, y)*."""
        self._check_initialised()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        preds = self.estimator.predict(X)
        return float(np.mean(preds == y))

    @property
    def history(self) -> list[QueryResult]:
        """All query results in order."""
        return list(self._history)

    @property
    def n_labelled(self) -> int:
        """Current size of the labelled set."""
        return 0 if self._X_labelled is None else len(self._X_labelled)

    @property
    def n_pool(self) -> int:
        """Current size of the unlabelled pool."""
        return 0 if self._X_pool is None else len(self._X_pool)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_initialised(self) -> None:
        if self._X_labelled is None:
            raise RuntimeError("Call initialize() first.")

    def _fit_estimator(self) -> None:
        import copy

        self.estimator = copy.deepcopy(self.estimator)
        self.estimator.fit(self._X_labelled, self._y_labelled)

    def _score_pool(self) -> np.ndarray:
        if self.strategy == "qbc":
            return self._qbc_scores()
        proba = self.estimator.predict_proba(self._X_pool)
        if self.strategy == "least_confident":
            return 1.0 - proba.max(axis=1)
        if self.strategy == "margin":
            sorted_p = np.sort(proba, axis=1)
            return 1.0 - (sorted_p[:, -1] - sorted_p[:, -2])
        # entropy (default)
        proba = np.clip(proba, 1e-9, 1.0)
        return -np.sum(proba * np.log(proba), axis=1)

    def _qbc_scores(self) -> np.ndarray:
        import copy

        n = len(self._X_pool)
        n_labelled = len(self._X_labelled)
        vote_matrix: list[np.ndarray] = []

        for seed in range(self.committee_size):
            rng = np.random.default_rng(self.random_state + seed)
            idx = rng.integers(0, n_labelled, size=n_labelled)
            X_boot = self._X_labelled[idx]
            y_boot = self._y_labelled[idx]
            member = copy.deepcopy(self.estimator)
            try:
                member.fit(X_boot, y_boot)
                votes = member.predict(self._X_pool)
            except Exception:
                votes = self.estimator.predict(self._X_pool)
            vote_matrix.append(votes)

        # Vote entropy: how often committee members disagree
        vote_arr = np.stack(vote_matrix, axis=1)  # (n_pool, committee_size)
        classes = np.unique(vote_arr)
        entropy = np.zeros(n)
        for cls in classes:
            p = (vote_arr == cls).mean(axis=1)
            p = np.clip(p, 1e-9, 1.0)
            entropy -= p * np.log(p)
        return entropy
