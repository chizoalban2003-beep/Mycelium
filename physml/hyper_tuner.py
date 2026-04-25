"""Stage 70 — HyperTuner: autonomous hyperparameter self-tuning.

Wires :class:`~physml.automl.AutoMLOptimizer` into the agent's self-improve
cycle so the agent can periodically re-search its own hyperparameters using
held-out validation performance as the objective.  Best configurations are
optionally stored in a :class:`~physml.knowledge_graph.KnowledgeGraph` for
cross-session persistence.

Classes
-------
HyperTuner
    Wraps an agent with periodic AutoML-driven hyperparameter search.
TuneResult
    Per-tuning-round snapshot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TuneResult:
    """Snapshot from one autonomous tuning round.

    Attributes
    ----------
    round_idx : int
        Zero-based tuning-round index.
    best_params : dict
        Best hyperparameter configuration found.
    best_score : float
        Cross-validated score for the best configuration.
    n_candidates : int
        Number of configurations evaluated.
    elapsed_s : float
        Wall-clock seconds for this tuning round.
    stored_in_graph : bool
        Whether the result was written to the KnowledgeGraph.
    """

    round_idx: int
    best_params: dict[str, Any]
    best_score: float
    n_candidates: int
    elapsed_s: float
    stored_in_graph: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialisable dict for JSON export."""
        return {
            "round": self.round_idx,
            "best_params": self.best_params,
            "best_score": round(self.best_score, 4),
            "n_candidates": self.n_candidates,
            "elapsed_s": round(self.elapsed_s, 3),
            "stored_in_graph": self.stored_in_graph,
        }


class HyperTuner:
    """Autonomous hyperparameter self-tuning loop.

    Wraps any agent with an :class:`~physml.automl.AutoMLOptimizer`-backed
    search that fires every *tune_every* improvement steps (or on explicit
    :meth:`tune` calls).  Best configs are applied to the agent's predictor
    and optionally persisted in a :class:`~physml.knowledge_graph.KnowledgeGraph`.

    Parameters
    ----------
    agent : Any
        Agent exposing ``fit(X, y)`` and ``predict(X)``.  If the agent also
        exposes ``self_improve(X, y)``, that is called after each tune round.
    param_grid : dict or None
        Search grid for :class:`~physml.automl.AutoMLOptimizer`.  When
        ``None``, the optimizer's default grid is used.
    n_candidates : int, default 6
        Number of candidate configurations per tuning round.
    tune_every : int, default 5
        Number of ``maybe_tune()`` calls between actual search rounds.
    knowledge_graph : KnowledgeGraph or None
        When provided, best configs are stored as nodes under the
        ``"hyper_tune"`` topic.
    random_state : int or None, default None

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> from physml import MyceliumAgent
    >>> from physml.hyper_tuner import HyperTuner
    >>> X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    >>> agent = MyceliumAgent()
    >>> tuner = HyperTuner(agent, tune_every=1)
    >>> result = tuner.tune(X[:200], y[:200])
    >>> print(result.best_score)
    """

    def __init__(
        self,
        agent: Any,
        *,
        param_grid: dict[str, list[Any]] | None = None,
        n_candidates: int = 6,
        tune_every: int = 5,
        knowledge_graph: Any | None = None,
        random_state: int | None = None,
    ) -> None:
        self.agent = agent
        self.param_grid = param_grid
        self.n_candidates = max(1, int(n_candidates))
        self.tune_every = max(1, int(tune_every))
        self.knowledge_graph = knowledge_graph
        self.random_state = random_state

        self._history: list[TuneResult] = []
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        X: Any,
        y: Any,
        estimator: Any | None = None,
    ) -> TuneResult:
        """Run one hyperparameter search round and apply best params to agent.

        Parameters
        ----------
        X, y : array-like
            Training data for the search.
        estimator : sklearn estimator or None
            Base estimator for the search.  Defaults to
            :class:`~physml.automl.AutoMLOptimizer`'s default.

        Returns
        -------
        TuneResult
        """
        from physml.automl import AutoMLOptimizer

        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        optimizer = AutoMLOptimizer(
            param_grid=self.param_grid,
            n_candidates=self.n_candidates,
            random_state=self.random_state,
        )
        best_params = optimizer.fit(X_arr, y_arr, estimator=estimator)
        best_score = optimizer.best_score_
        n_candidates = len(optimizer.cv_results_)

        # Apply best params to agent predictor if possible
        self._apply_params(best_params)

        elapsed = time.perf_counter() - t0

        # Optionally persist to KnowledgeGraph
        stored = False
        if self.knowledge_graph is not None:
            stored = self._store_to_graph(best_params, best_score)

        result = TuneResult(
            round_idx=len(self._history),
            best_params=best_params,
            best_score=best_score,
            n_candidates=n_candidates,
            elapsed_s=elapsed,
            stored_in_graph=stored,
        )
        self._history.append(result)
        return result

    def maybe_tune(self, X: Any, y: Any, estimator: Any | None = None) -> TuneResult | None:
        """Tune only every ``tune_every`` calls; otherwise return None.

        Intended for integration into an ongoing training loop.
        """
        self._call_count += 1
        if self._call_count % self.tune_every == 0:
            return self.tune(X, y, estimator=estimator)
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[TuneResult]:
        """Ordered list of per-round tuning snapshots."""
        return list(self._history)

    def best_result(self) -> TuneResult | None:
        """Return the tuning round with the highest ``best_score``."""
        if not self._history:
            return None
        return max(self._history, key=lambda r: r.best_score)

    def summary(self) -> dict[str, Any]:
        """High-level summary of all tuning rounds."""
        scores = [r.best_score for r in self._history]
        return {
            "n_rounds": len(self._history),
            "best_score_ever": round(max(scores), 4) if scores else None,
            "latest_best_params": self._history[-1].best_params if self._history else {},
            "knowledge_graph_enabled": self.knowledge_graph is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_params(self, params: dict[str, Any]) -> None:
        """Apply ``params`` to the agent's predictor if possible."""
        if not params:
            return
        # Try direct attribute set on agent
        for k, v in params.items():
            try:
                setattr(self.agent, k, v)
            except (AttributeError, TypeError):
                pass
        # Try inner predictor
        predictor = getattr(self.agent, "_predictor", None)
        if predictor is not None:
            for k, v in params.items():
                try:
                    setattr(predictor, k, v)
                except (AttributeError, TypeError):
                    pass

    def _store_to_graph(self, params: dict[str, Any], score: float) -> bool:
        """Write tuning result to the KnowledgeGraph."""
        try:
            import time as _time

            node_name = f"hyper_tune_{len(self._history)}"
            self.knowledge_graph.add_node(
                node_name,
                node_type="hyper_tune",
                best_params=str(params),
                best_score=score,
                timestamp=_time.time(),
            )
            return True
        except Exception:
            return False
