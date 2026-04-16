"""Stage 13 — Confidence calibration for neural-backend predictors.

:func:`calibrate_temperature` fits a single temperature parameter *T* on a
held-out calibration set so that ``softmax(logits / T)`` produces
well-calibrated probabilities.  The calibration is applied transparently inside
:class:`~physml.mycelium_agent.MyceliumAgent` after the initial ``fit()`` and
requires no changes to the underlying predictor.

For regressors (or predictors without ``predict_proba``) calibration is a
no-op — the function returns ``T = 1.0`` immediately.

Algorithm
---------
Temperature scaling is a single-parameter post-hoc calibration method [Guo et
al., 2017].  We minimise the negative log-likelihood of the held-out labels
with respect to *T* using scipy's bounded scalar minimiser.  Because the MLP
logits are not directly accessible through scikit-learn's API, we work with the
*probabilities* and recover approximate logits via ``log(p + eps)`` before
scaling.

Usage (direct)
--------------
::

    from physml.calibration import calibrate_temperature, apply_temperature

    T = calibrate_temperature(predictor, X_cal, y_cal)
    calibrated_proba = apply_temperature(predictor, X_new, T)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_temperature(
    predictor: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    T_min: float = 0.05,
    T_max: float = 10.0,
) -> float:
    """Fit a temperature *T* that minimises NLL on the calibration set.

    Parameters
    ----------
    predictor : fitted classifier with ``predict_proba``
    X_cal : array of shape (n_cal, n_features)
    y_cal : array of shape (n_cal,)
    T_min, T_max : search bounds for T

    Returns
    -------
    float — optimal temperature (1.0 if calibration cannot be applied)
    """
    if not hasattr(predictor, "predict_proba"):
        return 1.0

    try:
        proba = predictor.predict_proba(X_cal)  # (n, n_classes)
    except Exception:
        return 1.0

    if proba.ndim != 2 or proba.shape[1] < 2:
        return 1.0

    # Encode labels as integer class indices
    try:
        classes = getattr(predictor, "classes_", None)
        if classes is None:
            classes = np.unique(y_cal)
        label_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([label_to_idx.get(yi, 0) for yi in y_cal], dtype=int)
    except Exception:
        return 1.0

    n_samples = len(y_idx)
    eps = 1e-9

    def _nll(log_T: float) -> float:
        T = math.exp(log_T)
        # Recover approximate log-logits, scale, re-softmax
        log_p = np.log(proba + eps)
        scaled = log_p / T
        # Numerically stable softmax
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        cal_proba = exp_s / exp_s.sum(axis=1, keepdims=True)
        correct_proba = cal_proba[np.arange(n_samples), y_idx]
        return -float(np.mean(np.log(correct_proba + eps)))

    try:
        from scipy.optimize import minimize_scalar  # type: ignore
        result = minimize_scalar(
            _nll,
            bounds=(math.log(T_min), math.log(T_max)),
            method="bounded",
        )
        T = float(math.exp(result.x))
    except Exception:
        return 1.0

    return max(T_min, min(T_max, T))


def apply_temperature(
    predictor: Any,
    X: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return calibrated probabilities for *X* using *temperature*.

    Parameters
    ----------
    predictor : fitted classifier with ``predict_proba``
    X : array of shape (n_samples, n_features)
    temperature : float returned by :func:`calibrate_temperature`

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes)
        Calibrated probabilities.  Returns raw probabilities when
        calibration cannot be applied (T ≈ 1.0 or no predict_proba).
    """
    if not hasattr(predictor, "predict_proba"):
        raise AttributeError("predictor has no predict_proba")
    proba = predictor.predict_proba(X)
    if abs(temperature - 1.0) < 1e-6:
        return proba
    eps = 1e-9
    log_p = np.log(proba + eps)
    scaled = log_p / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)
