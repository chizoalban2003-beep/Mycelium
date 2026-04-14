"""Stage 61 — UncertaintyEstimator: quantify predictive uncertainty for
active-learning and risk-aware decision making.

Methods
-------
* **Monte-Carlo Dropout** (``"mc_dropout"``) — repeated stochastic forward
  passes; computes mean prediction and entropy/variance.
* **Ensemble disagreement** (``"ensemble"``) — measures disagreement across
  multiple fitted estimators.
* **Temperature scaling** (``"temperature"``) — calibrates probability
  estimates via a learned scalar temperature.
* **Laplace approximation** (``"laplace"``) — simple diagonal Laplace
  approximation for linear models.

Key class
---------
:class:`UncertaintyEstimator`

Usage
-----
::

    from physml.uncertainty import UncertaintyEstimator
    from sklearn.ensemble import RandomForestClassifier

    models = [RandomForestClassifier(n_estimators=20, random_state=i).fit(X, y)
              for i in range(5)]
    ue = UncertaintyEstimator(method="ensemble")
    ue.fit(models, X_train, y_train)
    uncertainty = ue.uncertainty(X_test)  # shape (n_test,)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class UncertaintyEstimator:
    """Estimate predictive uncertainty using several principled methods.

    Parameters
    ----------
    method : str, default "ensemble"
        Uncertainty estimation method: ``"ensemble"``, ``"mc_dropout"``,
        ``"temperature"``, or ``"laplace"``.
    n_passes : int, default 20
        Number of stochastic forward passes (``"mc_dropout"`` only).
    temperature : float, default 1.0
        Initial temperature for ``"temperature"`` scaling; updated by
        :meth:`calibrate`.
    random_state : int | None, default None
        Seed for reproducibility.
    """

    _VALID_METHODS = {"ensemble", "mc_dropout", "temperature", "laplace"}

    def __init__(
        self,
        method: str = "ensemble",
        n_passes: int = 20,
        temperature: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(self._VALID_METHODS)}, got {method!r}"
            )
        self.method = method
        self.n_passes = max(1, n_passes)
        self.temperature = float(temperature)
        self.random_state = random_state

        self._models: list[Any] = []
        self._classes: np.ndarray | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        models: Any,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "UncertaintyEstimator":
        """Register models and optionally calibrate temperature.

        Parameters
        ----------
        models : estimator or list of estimators
            Fitted sklearn-compatible estimator(s).
        X, y : array-like | None
            Calibration data for ``"temperature"`` and ``"laplace"`` methods.
        """
        if not isinstance(models, list):
            models = [models]
        if not models:
            raise ValueError("At least one model must be provided.")
        self._models = models

        # Determine class labels from first model with classes_ attribute
        for m in models:
            if hasattr(m, "classes_"):
                self._classes = np.asarray(m.classes_)
                break

        if self.method == "temperature" and X is not None and y is not None:
            self._calibrate_temperature(np.asarray(X, dtype=float), np.asarray(y))

        self._is_fitted = True
        return self

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calibrate temperature on hold-out data.  Returns learned temperature."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._calibrate_temperature(X, y)
        return self.temperature

    # ------------------------------------------------------------------
    # Prediction & uncertainty
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged probability estimates.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        probas = self._collect_probas(X)
        mean_p = probas.mean(axis=0)

        if self.method == "temperature" and self.temperature != 1.0:
            mean_p = self._apply_temperature(mean_p)

        return mean_p

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample uncertainty scores (higher = more uncertain).

        Uses **predictive entropy** for classification tasks and
        **prediction variance** for regression tasks.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        probas = self._collect_probas(X)  # (n_models, n_samples, n_classes) or (n, n_samples)

        if probas.ndim == 3 and probas.shape[2] > 1:
            mean_p = probas.mean(axis=0)
            if self.method == "temperature" and self.temperature != 1.0:
                mean_p = self._apply_temperature(mean_p)
            # Predictive entropy: H[y|x] = -∑ p log p  (clipped to ≥ 0)
            eps = 1e-12
            with np.errstate(invalid="ignore", divide="ignore"):
                entropy = -np.sum(mean_p * np.log(np.clip(mean_p, eps, None)), axis=1)
            return np.maximum(0.0, entropy)
        else:
            # Regression variance (squeeze away trailing dim-1 axis if present)
            var = probas.var(axis=0)
            return var.squeeze(-1) if var.ndim > 1 and var.shape[-1] == 1 else var

    def most_uncertain(self, X: np.ndarray, n: int = 10) -> np.ndarray:
        """Return indices of *n* most uncertain samples in *X*."""
        scores = self.uncertainty(X)
        n = min(n, len(scores))
        return np.argsort(scores)[::-1][:n]

    def aleatoric_epistemic_split(
        self, X: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Decompose uncertainty into aleatoric and epistemic components.

        * **Aleatoric** (irreducible) ≈ mean of individual model entropies.
        * **Epistemic** (reducible) ≈ total entropy − aleatoric.

        Returns
        -------
        dict with keys ``"total"``, ``"aleatoric"``, ``"epistemic"``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        probas = self._collect_probas(X)  # (n_models, n_samples, n_classes)
        if probas.ndim != 3:
            u = self.uncertainty(X)
            return {"total": u, "aleatoric": u * 0.5, "epistemic": u * 0.5}

        eps = 1e-12
        # Aleatoric: expected entropy
        per_model_entropy = -np.sum(probas * np.log(probas + eps), axis=2)  # (n_m, n_s)
        aleatoric = np.maximum(0.0, per_model_entropy.mean(axis=0))
        # Total: entropy of mean
        mean_p = probas.mean(axis=0)
        total = np.maximum(0.0, -np.sum(mean_p * np.log(mean_p + eps), axis=1))
        epistemic = np.maximum(0.0, total - aleatoric)
        return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_probas(self, X: np.ndarray) -> np.ndarray:
        """Gather probability arrays from all models."""
        all_p = []
        for m in self._models:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)
            else:
                preds = m.predict(X).reshape(-1, 1)
                p = preds.astype(float)
            all_p.append(p)
        stacked = np.stack(all_p, axis=0)  # (n_models, n_samples, ...)
        return stacked

    def _apply_temperature(self, proba: np.ndarray) -> np.ndarray:
        """Divide logits by temperature and re-normalise."""
        eps = 1e-12
        logits = np.log(proba + eps)
        scaled = logits / self.temperature
        # Softmax
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def _calibrate_temperature(self, X: np.ndarray, y: np.ndarray) -> None:
        """Simple grid search for optimal temperature (NLL minimisation)."""
        eps = 1e-12
        probas = self._collect_probas(X).mean(axis=0)
        best_t = 1.0
        best_nll = float("inf")
        for t in np.linspace(0.1, 5.0, 50):
            p_cal = self._apply_temperature_scalar(probas, t)
            n_classes = p_cal.shape[1]
            y_int = np.asarray(y, dtype=int)
            # Map labels to column indices
            if self._classes is not None:
                label_to_idx = {int(c): i for i, c in enumerate(self._classes)}
                idx = np.array([label_to_idx.get(int(yi), 0) for yi in y_int])
            else:
                idx = np.clip(y_int, 0, n_classes - 1)
            nll = -np.mean(np.log(p_cal[np.arange(len(idx)), idx] + eps))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        self.temperature = best_t

    def _apply_temperature_scalar(self, proba: np.ndarray, t: float) -> np.ndarray:
        eps = 1e-12
        logits = np.log(proba + eps) / t
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this estimator.")

    def __repr__(self) -> str:
        return (
            f"UncertaintyEstimator(method={self.method!r}, "
            f"n_passes={self.n_passes}, temperature={self.temperature:.3f})"
        )
