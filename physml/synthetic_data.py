"""Stage 60 — SyntheticDataGenerator: generate labelled tabular data for
testing, pre-training, and data augmentation.

Supported distributions
-----------------------
* ``"gaussian"`` — mixture of Gaussians (classification/regression).
* ``"moons"``    — two interleaved half-moons (binary classification).
* ``"blobs"``    — isotropic Gaussian blobs (multi-class classification).
* ``"regression"`` — linear combination + optional non-linearity + noise.

Key class
---------
:class:`SyntheticDataGenerator`

Usage
-----
::

    from physml.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(task="classification", n_classes=3,
                                 n_features=10, random_state=42)
    X, y = gen.generate(n_samples=500)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SyntheticDataGenerator:
    """Generate synthetic tabular datasets for ML experiments.

    Parameters
    ----------
    task : str, default "classification"
        Either ``"classification"`` or ``"regression"``.
    distribution : str, default "gaussian"
        Data-generation distribution; one of ``"gaussian"``, ``"moons"``,
        ``"blobs"``, ``"regression"``.
    n_features : int, default 10
        Number of input features.
    n_classes : int, default 2
        Number of target classes (classification only).
    noise : float, default 0.1
        Standard deviation of additive Gaussian noise on features.
    class_sep : float, default 1.0
        Controls distance between class centres (larger = easier).
    random_state : int | None, default None
        Seed for reproducibility.
    """

    _VALID_DISTRIBUTIONS = {"gaussian", "moons", "blobs", "regression"}

    def __init__(
        self,
        task: str = "classification",
        distribution: str = "gaussian",
        n_features: int = 10,
        n_classes: int = 2,
        noise: float = 0.1,
        class_sep: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
        if distribution not in self._VALID_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {sorted(self._VALID_DISTRIBUTIONS)}, "
                f"got {distribution!r}"
            )
        self.task = task
        self.distribution = distribution
        self.n_features = max(1, n_features)
        self.n_classes = max(2, n_classes)
        self.noise = max(0.0, noise)
        self.class_sep = class_sep
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._n_generated: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Generate *n_samples* labelled examples.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        """
        n_samples = max(1, n_samples)
        if self.distribution == "gaussian":
            X, y = self._gaussian(n_samples)
        elif self.distribution == "moons":
            X, y = self._moons(n_samples)
        elif self.distribution == "blobs":
            X, y = self._blobs(n_samples)
        else:  # "regression"
            X, y = self._regression(n_samples)

        self._n_generated += n_samples
        return X, y

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_synthetic: int = 100,
        noise_scale: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment an existing dataset by adding Gaussian-perturbed copies.

        Parameters
        ----------
        X, y : existing labelled data.
        n_synthetic : int
            Number of synthetic samples to add.
        noise_scale : float | None
            Noise std for augmented samples; defaults to ``self.noise``.

        Returns
        -------
        X_aug, y_aug : augmented arrays (original + synthetic).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        scale = noise_scale if noise_scale is not None else self.noise
        idx = self._rng.integers(0, len(X), size=n_synthetic)
        X_syn = X[idx] + self._rng.normal(0.0, scale, (n_synthetic, X.shape[1]))
        y_syn = y[idx]
        return np.vstack([X, X_syn]), np.concatenate([y, y_syn])

    @property
    def n_generated(self) -> int:
        """Total samples generated across all ``generate()`` calls."""
        return self._n_generated

    def reset(self) -> None:
        """Reset RNG and sample counter (seeds are preserved)."""
        self._rng = np.random.default_rng(self.random_state)
        self._n_generated = 0

    def describe(self) -> dict[str, Any]:
        """Return a metadata dict describing this generator."""
        return {
            "task": self.task,
            "distribution": self.distribution,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "noise": self.noise,
            "class_sep": self.class_sep,
            "random_state": self.random_state,
            "n_generated": self._n_generated,
        }

    # ------------------------------------------------------------------
    # Private generators
    # ------------------------------------------------------------------

    def _gaussian(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Mixture of Gaussians, one centre per class."""
        centres = (
            self._rng.normal(0, self.class_sep, (self.n_classes, self.n_features))
        )
        per_class = n // self.n_classes
        remainder = n - per_class * self.n_classes
        X_parts, y_parts = [], []
        for c, centre in enumerate(centres):
            count = per_class + (1 if c < remainder else 0)
            samples = self._rng.normal(0, self.noise, (count, self.n_features)) + centre
            X_parts.append(samples)
            y_parts.append(np.full(count, c, dtype=int))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        if self.task == "regression":
            y = X[:, 0] * 2.0 + self._rng.normal(0, self.noise, n)
        return X, y

    def _moons(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Two interleaved half-moons (binary, first 2 features)."""
        half = n // 2
        t0 = self._rng.uniform(0, np.pi, half)
        t1 = self._rng.uniform(np.pi, 2 * np.pi, n - half)
        X0 = np.column_stack([np.cos(t0), np.sin(t0)])
        X1 = np.column_stack([1.0 + np.cos(t1), -np.sin(t1)])
        if self.n_features > 2:
            extra0 = self._rng.normal(0, self.noise, (half, self.n_features - 2))
            extra1 = self._rng.normal(0, self.noise, (n - half, self.n_features - 2))
            X0 = np.hstack([X0, extra0])
            X1 = np.hstack([X1, extra1])
        X = np.vstack([X0, X1])
        X += self._rng.normal(0, self.noise, X.shape)
        y = np.concatenate([np.zeros(half, int), np.ones(n - half, int)])
        if self.task == "regression":
            y = X[:, 0] + self._rng.normal(0, self.noise, n)
        return X, y

    def _blobs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Isotropic Gaussian blobs, one per class."""
        return self._gaussian(n)  # Gaussian with unit std per feature

    def _regression(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Linear regression with optional non-linear interaction terms."""
        X = self._rng.normal(0, 1, (n, self.n_features))
        coef = self._rng.normal(0, 1, self.n_features)
        y = X @ coef + self._rng.normal(0, self.noise, n)
        return X, y

    def __repr__(self) -> str:
        return (
            f"SyntheticDataGenerator(task={self.task!r}, "
            f"distribution={self.distribution!r}, "
            f"n_features={self.n_features}, n_classes={self.n_classes})"
        )
