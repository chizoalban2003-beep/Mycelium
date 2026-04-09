"""Statistical fingerprinting for signal particles.

Every signal that enters the ecosystem gets a full statistical fingerprint:
    - Normality (Shapiro-Wilk) — is this particle in a fluid or crystalline medium?
    - Entropy (Shannon) — how much information does this signal carry?
    - Autocorrelation — does this signal have temporal structure (periodicity)?
    - Stationarity (ADF-like) — is this signal stable or drifting?
    - Effect size — how large is this particle's impact on the field?

These measurements become the particle's intrinsic properties — they're not
analysis of the particle, they ARE the particle's physical characteristics.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def compute_fingerprint(
    values: list[float | int],
    timestamps: list[float] | None = None,
) -> dict[str, Any]:
    """Compute the full statistical fingerprint of a signal's value history.

    Parameters
    ----------
    values : list of numeric values (e.g., CPU readings, occurrence counts)
    timestamps : optional list of epoch seconds for temporal analysis

    Returns
    -------
    dict with: normality_p, normality_stat, entropy, autocorrelation,
               stationarity, mean, std, skewness, kurtosis, effect_size,
               n_observations, medium (fluid/crystalline/gaseous)
    """
    n = len(values)
    fp: dict[str, Any] = {
        "n_observations": n,
        "normality_p": None,
        "normality_stat": None,
        "entropy": 0.0,
        "autocorrelation": 0.0,
        "stationarity": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "effect_size": 0.0,
        "medium": "gaseous",
    }

    if n < 3:
        return fp

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 3:
        return fp

    fp["mean"] = round(float(np.mean(arr)), 6)
    fp["std"] = round(float(np.std(arr, ddof=1)), 6) if n > 1 else 0.0

    # --- Normality (Shapiro-Wilk) ---
    if _HAS_SCIPY and 3 <= n <= 5000:
        try:
            stat, p = sp_stats.shapiro(arr[:5000])
            fp["normality_stat"] = round(float(stat), 6)
            fp["normality_p"] = round(float(p), 8)
        except Exception:
            pass

    # --- Shannon entropy ---
    try:
        if n > 0:
            bins = min(20, max(2, n // 3))
            hist, _ = np.histogram(arr, bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            fp["entropy"] = round(float(-np.sum(probs * np.log2(probs))), 6)
    except Exception:
        pass

    # --- Autocorrelation (lag-1) ---
    if n > 2:
        try:
            mean = np.mean(arr)
            var = np.var(arr)
            if var > 1e-12:
                ac = np.sum((arr[:-1] - mean) * (arr[1:] - mean)) / ((n - 1) * var)
                fp["autocorrelation"] = round(float(np.clip(ac, -1, 1)), 6)
        except Exception:
            pass

    # --- Stationarity (mean + variance shift between first and second half) ---
    if n >= 6:
        try:
            mid = n // 2
            mean1 = float(np.mean(arr[:mid]))
            mean2 = float(np.mean(arr[mid:]))
            var1 = float(np.var(arr[:mid]))
            var2 = float(np.var(arr[mid:]))
            total_var = float(np.var(arr))

            # Variance shift
            var_shift = abs(var1 - var2) / (var1 + var2 + 1e-12)
            # Mean shift relative to overall spread
            mean_shift = abs(mean1 - mean2) / (math.sqrt(total_var) + 1e-12) if total_var > 1e-12 else abs(mean1 - mean2)
            mean_shift = min(1.0, mean_shift / 3.0)

            combined_shift = max(var_shift, mean_shift)
            fp["stationarity"] = round(max(0.0, 1.0 - combined_shift), 6)
        except Exception:
            fp["stationarity"] = 0.5

    # --- Skewness and Kurtosis ---
    if _HAS_SCIPY and n > 3:
        try:
            fp["skewness"] = round(float(sp_stats.skew(arr)), 6)
            fp["kurtosis"] = round(float(sp_stats.kurtosis(arr)), 6)
        except Exception:
            pass

    # --- Effect size (Cohen's d from mean vs 0) ---
    if fp["std"] > 1e-12:
        fp["effect_size"] = round(abs(fp["mean"]) / fp["std"], 6)

    # --- Medium classification ---
    if fp["std"] < 1e-6:
        fp["medium"] = "frozen"
    else:
        norm_p = fp["normality_p"]
        if norm_p is not None:
            if norm_p > 0.05:
                fp["medium"] = "fluid"
            elif norm_p > 0.001:
                fp["medium"] = "crystalline"
            else:
                fp["medium"] = "gaseous"
        else:
            fp["medium"] = "unknown"

    return fp


def fingerprint_to_particle_props(fp: dict[str, Any]) -> dict[str, float]:
    """Convert a statistical fingerprint into force field particle properties.

    Maps statistical measurements to physical properties:
        normality → ionization type (parametric vs nonparametric)
        entropy → viscosity contribution
        autocorrelation → spin (periodicity)
        stationarity → stability (low stationarity = drifting)
        effect_size → mass amplification
    """
    return {
        "mass_amplifier": min(3.0, 1.0 + _safe_float(fp.get("effect_size"), 0) * 0.3),
        "spin": _safe_float(fp.get("autocorrelation"), 0) * 0.5 + 0.5,
        "viscosity_contribution": _safe_float(fp.get("entropy"), 0) * 0.2,
        "stability": _safe_float(fp.get("stationarity"), 0.5),
        "ionization": "parametric" if fp.get("medium") == "fluid" else "nonparametric",
        "medium": str(fp.get("medium", "unknown")),
        "skewness": _safe_float(fp.get("skewness"), 0),
        "kurtosis": _safe_float(fp.get("kurtosis"), 0),
    }
