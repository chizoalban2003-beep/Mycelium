"""Stage 101 — NeuralSearchEngine: lightweight neural architecture search.

Explores a discrete space of MLP configurations (layer widths and depths)
and selects the architecture that achieves the highest cross-validated score
on a provided dataset, without requiring external AutoML frameworks.

Classes
-------
SearchResult
    Outcome of one architecture evaluation trial.
NeuralSearchEngine
    Manages the search loop and exposes the best-found architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Outcome of a single architecture trial.

    Attributes
    ----------
    trial_id : int
        Zero-based trial index.
    hidden_layers : tuple of int
        Layer widths evaluated in this trial.
    score : float
        Cross-validated accuracy or R² achieved.
    train_time : float
        Wall-clock seconds taken to fit and evaluate.
    metadata : dict
        Arbitrary supplementary information.
    """

    trial_id: int
    hidden_layers: Tuple[int, ...]
    score: float
    train_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NeuralSearchEngine
# ---------------------------------------------------------------------------


class NeuralSearchEngine:
    """Discrete neural architecture search engine.

    Evaluates a configurable set of MLP hidden-layer configurations using
    cross-validation and tracks the best-scoring architecture.

    Parameters
    ----------
    search_space : list of tuple, optional
        Explicit list of ``(width, …)`` tuples to evaluate.  When *None*
        a small default grid is used.
    cv : int, default 3
        Number of cross-validation folds.
    max_iter : int, default 50
        Maximum MLP training iterations per trial (keeps tests fast).
    random_state : int or None, default None
        Seed for reproducibility.
    task : {"classification", "regression"}, default "classification"
        Controls which sklearn estimator is used internally.
    """

    _DEFAULT_SPACE: List[Tuple[int, ...]] = [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ]

    def __init__(
        self,
        search_space: Optional[List[Tuple[int, ...]]] = None,
        cv: int = 3,
        max_iter: int = 50,
        random_state: Optional[int] = None,
        task: str = "classification",
    ) -> None:
        self.search_space: List[Tuple[int, ...]] = search_space or list(self._DEFAULT_SPACE)
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self.task = task
        self._results: List[SearchResult] = []
        self._best: Optional[SearchResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        search_space: Optional[List[Tuple[int, ...]]] = None,
    ) -> SearchResult:
        """Run the architecture search.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        search_space : list of tuple, optional
            Override the instance-level search space for this call.

        Returns
        -------
        SearchResult
            The result for the best-scoring architecture found.
        """
        import time

        from sklearn.model_selection import cross_val_score
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        space = search_space or self.search_space

        for trial_id, hidden_layers in enumerate(space):
            t0 = time.perf_counter()
            if self.task == "regression":
                est = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                scoring = "r2"
            else:
                est = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                scoring = "accuracy"

            scores = cross_val_score(est, X, y, cv=self.cv, scoring=scoring)
            elapsed = time.perf_counter() - t0
            result = SearchResult(
                trial_id=trial_id,
                hidden_layers=tuple(hidden_layers),
                score=float(scores.mean()),
                train_time=elapsed,
            )
            self._results.append(result)
            if self._best is None or result.score > self._best.score:
                self._best = result

        return self._best  # type: ignore[return-value]

    @property
    def best_result(self) -> Optional[SearchResult]:
        """Best :class:`SearchResult` found so far, or *None*."""
        return self._best

    @property
    def all_results(self) -> List[SearchResult]:
        """All trial results in evaluation order."""
        return list(self._results)

    def reset(self) -> None:
        """Clear all search history."""
        self._results = []
        self._best = None
