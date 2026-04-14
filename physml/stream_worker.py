"""Stage 35 — ParallelDataStream: concurrent mini-batch processing.

Provides :class:`ParallelDataStream` which processes a list of
``(X_chunk, y_chunk)`` pairs in parallel using either a thread pool or a
process pool from :mod:`concurrent.futures`.  Supports:

* :meth:`~ParallelDataStream.map` — apply any function to each chunk in
  parallel and collect results.
* :meth:`~ParallelDataStream.fit_stream` — stream-fit a predictor across
  all chunks, optionally using FedAvg-style weight averaging.
"""

from __future__ import annotations

import concurrent.futures
import copy
from typing import Any, Callable

import numpy as np


class ParallelDataStream:
    """Parallel mini-batch streaming using :mod:`concurrent.futures`.

    Parameters
    ----------
    chunks : list[tuple[np.ndarray, np.ndarray]]
        List of ``(X_chunk, y_chunk)`` pairs.
    n_workers : int, default 4
        Number of parallel workers.
    backend : {"thread", "process"}, default "thread"
        Parallelism backend.  ``"thread"`` is suitable for sklearn estimators
        (which release the GIL during fit).  ``"process"`` may be used for
        CPU-bound tasks that do not release the GIL.
    """

    def __init__(
        self,
        chunks: list[tuple[np.ndarray, np.ndarray]],
        n_workers: int = 4,
        backend: str = "thread",
    ) -> None:
        self.chunks = list(chunks)
        self.n_workers = max(1, int(n_workers))
        self.backend = str(backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_stream(self, predictor: Any, sequential: bool = False) -> Any:
        """Stream-fit *predictor* across all chunks.

        Parameters
        ----------
        predictor : sklearn-compatible estimator
            Must expose either ``partial_fit`` or ``fit``.
        sequential : bool, default False
            When ``True``, fall back to sequential ``partial_fit`` (or
            ``fit``) calls.  When ``False``, chunks are fitted in parallel
            and weights are aggregated using FedAvg-style averaging.

        Returns
        -------
        predictor (mutated in-place when possible).
        """
        if sequential or not self.chunks:
            return self._fit_sequential(predictor)
        return self._fit_parallel(predictor)

    def map(self, fn: Callable, use_parallel: bool = True) -> list:
        """Apply *fn(X_chunk, y_chunk)* to each chunk and collect results.

        Parameters
        ----------
        fn : Callable[[np.ndarray, np.ndarray], Any]
        use_parallel : bool, default True

        Returns
        -------
        list — one result per chunk, in chunk order.
        """
        if not use_parallel or self.n_workers == 1:
            return [fn(X, y) for X, y in self.chunks]

        executor_cls = self._executor_cls()
        with executor_cls(max_workers=self.n_workers) as executor:
            futures = [executor.submit(fn, X, y) for X, y in self.chunks]
            return [f.result() for f in futures]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _executor_cls(self) -> type:
        if self.backend == "process":
            return concurrent.futures.ProcessPoolExecutor
        return concurrent.futures.ThreadPoolExecutor

    def _fit_sequential(self, predictor: Any) -> Any:
        """Fall back to sequential partial_fit / fit over all chunks."""
        for X_chunk, y_chunk in self.chunks:
            if hasattr(predictor, "partial_fit"):
                predictor.partial_fit(X_chunk, y_chunk)
            else:
                predictor.fit(X_chunk, y_chunk)
        return predictor

    def _fit_parallel(self, predictor: Any) -> Any:
        """Fit independent copies on each chunk, then FedAvg-aggregate."""

        def _fit_one(chunk: tuple[np.ndarray, np.ndarray]) -> Any:
            X, y = chunk
            local = copy.deepcopy(predictor)
            if hasattr(local, "partial_fit"):
                local.partial_fit(X, y)
            else:
                local.fit(X, y)
            return local

        executor_cls = self._executor_cls()
        with executor_cls(max_workers=self.n_workers) as executor:
            local_models = list(executor.map(_fit_one, self.chunks))

        # FedAvg: average first-available weight attribute
        self._fedavg(predictor, local_models)
        return predictor

    @staticmethod
    def _fedavg(global_model: Any, local_models: list[Any]) -> None:
        """Average weights from *local_models* into *global_model* in-place."""
        if not local_models:
            return

        # Multi-layer MLP: coefs_ (list of arrays)
        if hasattr(global_model, "coefs_"):
            valid = [m for m in local_models if hasattr(m, "coefs_")]
            if valid:
                n = len(valid)
                for i in range(len(valid[0].coefs_)):
                    avg = sum(m.coefs_[i] for m in valid) / n
                    # Ensure global model has coefs_ (may need dummy fit first)
                    if hasattr(global_model, "coefs_"):
                        global_model.coefs_[i] = avg
            return

        # Linear model: coef_
        if hasattr(global_model, "coef_"):
            valid = [m for m in local_models if hasattr(m, "coef_")]
            if valid:
                avg = sum(m.coef_ for m in valid) / len(valid)
                global_model.coef_ = avg
            return

        # Fallback: copy the first local model's state wholesale
        if local_models:
            first = local_models[0]
            for attr in vars(first):
                if not attr.startswith("__"):
                    try:
                        setattr(global_model, attr, getattr(first, attr))
                    except AttributeError:
                        pass
