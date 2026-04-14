"""Stage 51 — MetaLearner: strategy selector via performance history.

The ``MetaLearner`` learns, over time, which
(``query_strategy``, ``policy``) combinations perform best for different
dataset *profiles* (size, dimensionality, class balance) — without any
external AutoML dependency.

Algorithm
---------
1. Every time an agent finishes a task (user calls :meth:`record`), the
   meta-learner stores a ``(dataset_profile, config, score)`` tuple.
2. When asked :meth:`recommend` for a new dataset profile, it retrieves the
   top-k most similar historical profiles (cosine similarity on normalised
   feature vectors) and returns the configuration with the highest
   weighted average score.
3. If fewer than ``min_history`` entries exist, it falls back to the
   default hard-coded recommendation.

Usage
-----
::

    from physml.meta_learner import MetaLearner

    ml = MetaLearner()

    # After a training run:
    ml.record(
        X=X_train,
        y=y_train,
        config={"query_strategy": "entropy", "policy": "adaptive"},
        score=0.87,
    )

    # Before starting a new run:
    rec = ml.recommend(X_new, y_new)
    print(rec)
    # {"query_strategy": "entropy", "policy": "adaptive"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_DEFAULT_CONFIG: dict[str, Any] = {
    "query_strategy": "entropy",
    "policy": "adaptive",
}


@dataclass
class _Entry:
    profile: np.ndarray          # normalised 1-D float vector
    config: dict[str, Any]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MetaLearner:
    """Online meta-learner that recommends agent configurations.

    Parameters
    ----------
    k : int, default 5
        Number of nearest neighbours used for score aggregation.
    min_history : int, default 3
        Minimum stored entries before recommendation departs from the
        hard-coded default.
    decay : float, default 0.95
        Multiplicative weight decay for older entries
        (entries recorded earlier are down-weighted).
    """

    def __init__(
        self,
        k: int = 5,
        min_history: int = 3,
        decay: float = 0.95,
    ) -> None:
        self.k = k
        self.min_history = min_history
        self.decay = decay

        self._entries: list[_Entry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: dict[str, Any],
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a performance observation.

        Parameters
        ----------
        X, y : dataset used for the run
        config : configuration dict (``query_strategy``, ``policy``, …)
        score : scalar performance metric (higher = better, e.g. accuracy)
        metadata : optional extra info (dataset name, timestamp, …)
        """
        profile = self._dataset_profile(X, y)
        entry = _Entry(
            profile=profile,
            config=dict(config),
            score=float(score),
            metadata=dict(metadata or {}),
        )
        self._entries.append(entry)

    def recommend(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Recommend a configuration for a new dataset.

        Parameters
        ----------
        X, y : the new dataset (used to compute the profile)

        Returns
        -------
        dict
            Best configuration dict found in history, or a hard-coded
            default when history is too short.
        """
        if len(self._entries) < self.min_history:
            return dict(_DEFAULT_CONFIG)

        profile = self._dataset_profile(X, y)
        similarities = self._cosine_similarities(profile)

        # Top-k indices
        k = min(self.k, len(self._entries))
        top_k_idx = np.argsort(similarities)[::-1][:k]

        # Recency weights (most recent = highest weight)
        n = len(self._entries)
        recency = np.array([self.decay ** (n - 1 - i) for i in range(n)])

        # Aggregate scores per config key (json-serialisable repr)
        config_scores: dict[str, list[float]] = {}
        config_map: dict[str, dict[str, Any]] = {}
        for idx in top_k_idx:
            entry = self._entries[idx]
            key = str(sorted(entry.config.items()))
            weight = float(similarities[idx]) * float(recency[idx])
            config_scores.setdefault(key, []).append(weight * entry.score)
            config_map[key] = entry.config

        best_key = max(config_scores, key=lambda k: sum(config_scores[k]))
        return dict(config_map[best_key])

    def history_size(self) -> int:
        """Number of stored performance entries."""
        return len(self._entries)

    def top_configs(self, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k configs by raw score (no weighting)."""
        sorted_entries = sorted(self._entries, key=lambda e: e.score, reverse=True)
        return [{"config": e.config, "score": e.score} for e in sorted_entries[:k]]

    def dataset_profile(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Return a human-readable profile dict for *X*, *y*."""
        p = self._dataset_profile(X, y)
        keys = [
            "log_n_samples", "log_n_features", "class_balance",
            "mean_feature_corr", "target_std_norm",
        ]
        return dict(zip(keys, p.tolist()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dataset_profile(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute a normalised 5-D profile vector for a dataset."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape

        # 1. log(n_samples) / log(1e6) — normalised to ~[0,1]
        feat_n = np.log1p(n) / np.log1p(1e6)

        # 2. log(n_features) / log(1e4)
        feat_d = np.log1p(d) / np.log1p(1e4)

        # 3. Class balance (gini impurity for classifiers, 0 for regressors)
        feat_balance = self._gini(y)

        # 4. Mean absolute feature correlation (approximated)
        if n >= 4 and d >= 2:
            sample = X[:min(200, n)]
            corr = np.corrcoef(sample.T)
            if corr.ndim == 2:
                mask = ~np.eye(d, dtype=bool)
                feat_corr = float(np.mean(np.abs(corr[mask])))
            else:
                feat_corr = 0.0
        else:
            feat_corr = 0.0

        # 5. Normalised target variance
        try:
            y_float = y.astype(float)
            std = float(np.std(y_float))
            mean_abs = float(np.mean(np.abs(y_float))) or 1.0
            feat_target = min(std / mean_abs, 10.0) / 10.0
        except (ValueError, TypeError):
            feat_target = 0.0

        vec = np.array([feat_n, feat_d, feat_balance, feat_corr, feat_target], dtype=float)
        # L2-normalise
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    def _cosine_similarities(self, profile: np.ndarray) -> np.ndarray:
        """Cosine similarity between *profile* and all stored profiles."""
        stored = np.array([e.profile for e in self._entries])  # (n, d)
        dots = stored @ profile
        norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(profile)
        norms = np.where(norms == 0, 1.0, norms)
        return dots / norms

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """Gini impurity, clipped to [0, 1]."""
        try:
            vals, counts = np.unique(y, return_counts=True)
            if len(vals) <= 1:
                return 0.0
            probs = counts / counts.sum()
            return float(1.0 - np.sum(probs ** 2))
        except Exception:
            return 0.0
