"""Stage 84 — ModelZoo: curated preset model configurations.

Provides a searchable registry of popular, well-tuned sklearn estimator
configurations (presets) retrievable by name, task type, and performance
tier.  Each preset is a lightweight factory: calling it returns a fresh
unfitted estimator instance.

Classes
-------
ZooEntry
    Metadata record for one model preset.
ModelZoo
    Registry of labelled model presets with search and retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ZooEntry:
    """Metadata for one model preset in the zoo.

    Attributes
    ----------
    name : str
        Unique identifier (e.g. ``"lr_l2"``, ``"rf_100"``).
    task : str
        ``"classification"``, ``"regression"``, or ``"any"``.
    tier : str
        Speed / accuracy tier: ``"fast"``, ``"balanced"``, ``"accurate"``.
    description : str
        Short human-readable description.
    tags : list[str]
        Searchable keyword tags.
    factory : Callable[[], Any]
        Zero-argument callable returning a fresh estimator instance.
    """

    name: str
    task: str
    tier: str
    description: str
    tags: list[str]
    factory: Callable[[], Any]

    def build(self) -> Any:
        """Instantiate and return a fresh (unfitted) estimator."""
        return self.factory()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "tier": self.tier,
            "description": self.description,
            "tags": self.tags,
        }

    def __repr__(self) -> str:
        return (
            f"ZooEntry(name={self.name!r}, task={self.task!r}, "
            f"tier={self.tier!r})"
        )


class ModelZoo:
    """Registry of named model presets with search and retrieval.

    A default zoo is populated automatically on first instantiation with
    a curated set of logistic regression, random forest, gradient boosting,
    and linear regression presets.  Custom entries can be added via
    :meth:`register`.

    Parameters
    ----------
    include_defaults : bool, default True
        Populate the zoo with built-in presets on construction.

    Example
    -------
    >>> from physml.model_zoo import ModelZoo
    >>> zoo = ModelZoo()
    >>> len(zoo) >= 5
    True
    >>> entry = zoo.get("lr_fast")
    >>> entry is not None
    True
    >>> model = entry.build()
    >>> hasattr(model, "fit")
    True
    """

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._entries: dict[str, ZooEntry] = {}
        if include_defaults:
            self._populate_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, entry: ZooEntry) -> None:
        """Add a :class:`ZooEntry` to the registry.

        Parameters
        ----------
        entry : ZooEntry
            An entry whose :attr:`ZooEntry.name` is used as the key.
            Overwrites any existing entry with the same name.
        """
        self._entries[entry.name] = entry

    def get(self, name: str) -> ZooEntry | None:
        """Retrieve an entry by name, or ``None`` if not found."""
        return self._entries.get(name)

    def build(self, name: str) -> Any:
        """Instantiate the estimator for preset *name*.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"No model preset named {name!r} in the zoo.")
        return entry.build()

    def search(
        self,
        *,
        task: str | None = None,
        tier: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ZooEntry]:
        """Search the registry with optional filters.

        Parameters
        ----------
        task : str or None
            Filter by task (``"classification"``, ``"regression"``,
            ``"any"``).  Entries with task ``"any"`` match any filter value.
        tier : str or None
            Filter by tier (``"fast"``, ``"balanced"``, ``"accurate"``).
        tags : list[str] or None
            Return only entries that contain *all* the given tags.

        Returns
        -------
        list[ZooEntry]
        """
        results = list(self._entries.values())

        if task is not None:
            results = [
                e for e in results if e.task == task or e.task == "any"
            ]
        if tier is not None:
            results = [e for e in results if e.tier == tier]
        if tags:
            for tag in tags:
                results = [e for e in results if tag in e.tags]

        return results

    def list_names(self) -> list[str]:
        """Return all registered preset names."""
        return list(self._entries.keys())

    def summary(self) -> list[dict[str, Any]]:
        """Return a list of metadata dicts for all presets."""
        return [e.as_dict() for e in self._entries.values()]

    def compare(self, names: list[str], X: Any, y: Any) -> list[dict[str, Any]]:
        """Quick benchmark: fit and score each named preset on *(X, y)*.

        Uses a 60/40 train-test split.  Returns results sorted by score
        (descending).

        Parameters
        ----------
        names : list[str]
            Preset names to compare.
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        list[dict]
            Each dict has keys ``name``, ``score``, ``elapsed_s``.
        """
        import numpy as np

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(X)
        split = int(n * 0.6)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        results = []
        for name in names:
            entry = self._entries.get(name)
            if entry is None:
                continue
            t0 = time.time()
            model = entry.build()
            try:
                model.fit(X_tr, y_tr)
                score = float(model.score(X_te, y_te))
            except Exception:
                score = float("nan")
            results.append(
                {
                    "name": name,
                    "score": round(score, 4),
                    "elapsed_s": round(time.time() - t0, 4),
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    # ------------------------------------------------------------------
    # Default presets
    # ------------------------------------------------------------------

    def _populate_defaults(self) -> None:
        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            return  # sklearn not available — skip defaults

        defaults: list[ZooEntry] = [
            ZooEntry(
                name="lr_fast",
                task="classification",
                tier="fast",
                description="Logistic Regression with L2 penalty, max_iter=200",
                tags=["linear", "fast", "interpretable"],
                factory=lambda: LogisticRegression(max_iter=200, random_state=0),
            ),
            ZooEntry(
                name="lr_balanced",
                task="classification",
                tier="balanced",
                description="Logistic Regression with balanced class weights",
                tags=["linear", "balanced", "imbalanced"],
                factory=lambda: LogisticRegression(
                    max_iter=500, class_weight="balanced", random_state=0
                ),
            ),
            ZooEntry(
                name="rf_fast",
                task="classification",
                tier="fast",
                description="Random Forest with 50 trees",
                tags=["ensemble", "fast", "tree"],
                factory=lambda: RandomForestClassifier(
                    n_estimators=50, random_state=0, n_jobs=-1
                ),
            ),
            ZooEntry(
                name="rf_accurate",
                task="classification",
                tier="accurate",
                description="Random Forest with 200 trees and tuned depth",
                tags=["ensemble", "accurate", "tree"],
                factory=lambda: RandomForestClassifier(
                    n_estimators=200, max_depth=None, random_state=0, n_jobs=-1
                ),
            ),
            ZooEntry(
                name="gbt_balanced",
                task="classification",
                tier="balanced",
                description="Gradient Boosting classifier, 100 trees",
                tags=["ensemble", "boosting", "balanced"],
                factory=lambda: GradientBoostingClassifier(
                    n_estimators=100, random_state=0
                ),
            ),
            ZooEntry(
                name="gbt_accurate",
                task="classification",
                tier="accurate",
                description="Gradient Boosting classifier, 300 trees, deep",
                tags=["ensemble", "boosting", "accurate"],
                factory=lambda: GradientBoostingClassifier(
                    n_estimators=300, max_depth=5, random_state=0
                ),
            ),
            ZooEntry(
                name="knn_fast",
                task="classification",
                tier="fast",
                description="K-Nearest Neighbours, k=5",
                tags=["instance-based", "fast", "nonparametric"],
                factory=lambda: KNeighborsClassifier(n_neighbors=5),
            ),
            ZooEntry(
                name="dt_fast",
                task="classification",
                tier="fast",
                description="Decision Tree (max_depth=5)",
                tags=["tree", "fast", "interpretable"],
                factory=lambda: DecisionTreeClassifier(
                    max_depth=5, random_state=0
                ),
            ),
            ZooEntry(
                name="ridge_fast",
                task="regression",
                tier="fast",
                description="Ridge Regression, alpha=1.0",
                tags=["linear", "fast", "regression"],
                factory=lambda: Ridge(alpha=1.0),
            ),
            ZooEntry(
                name="rf_reg_balanced",
                task="regression",
                tier="balanced",
                description="Random Forest Regressor, 100 trees",
                tags=["ensemble", "balanced", "regression", "tree"],
                factory=lambda: RandomForestRegressor(
                    n_estimators=100, random_state=0, n_jobs=-1
                ),
            ),
            ZooEntry(
                name="gbt_reg_accurate",
                task="regression",
                tier="accurate",
                description="Gradient Boosting Regressor, 200 trees",
                tags=["ensemble", "boosting", "accurate", "regression"],
                factory=lambda: GradientBoostingRegressor(
                    n_estimators=200, random_state=0
                ),
            ),
        ]

        for entry in defaults:
            self._entries[entry.name] = entry
