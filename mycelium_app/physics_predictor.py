from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import math

import numpy as np
import pandas as pd


TargetKind = Literal["numeric", "categorical", "datetime"]
FeatureKind = Literal["numeric", "categorical", "datetime", "bool"]


class PhysicsPlane(str, Enum):
    solid = "solid"
    liquid = "liquid"
    gas = "gas"


@dataclass(frozen=True)
class WeightInfo:
    feature: str
    weight: float
    method: str
    feature_kind: FeatureKind
    signed: bool


@dataclass(frozen=True)
class MigrationInfo:
    feature: str
    feature_kind: FeatureKind
    method: str
    charge: float
    entropy: float
    variance: float
    standard_error: float
    kl_divergence: float
    density: float
    viscosity: float
    terminal_velocity: float
    arrival_speed: float
    direction: Literal["pulled", "repelled", "neutral"]
    state: Literal["free", "dampened", "trapped"]


@dataclass(frozen=True)
class PredictionMetrics:
    target_kind: TargetKind
    n_rows: int
    n_train: int
    n_test: int
    train_fraction: float
    random_seed: int
    n_features_used: int
    mae: float | None = None
    rmse: float | None = None
    accuracy: float | None = None
    baseline_accuracy: float | None = None
    baseline_mae: float | None = None
    baseline_rmse: float | None = None
    best_cycle: int | None = None
    best_lift: float | None = None


@dataclass(frozen=True)
class BondInfo:
    feature_a: str
    feature_b: str
    affinity: float
    bonding_factor: float


@dataclass(frozen=True)
class IterationInfo:
    cycle: int
    test_accuracy: float | None = None
    lift_over_baseline: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None


@dataclass(frozen=True)
class EquilibriumZone:
    zone_id: int
    features: list[str]
    avg_pI: float
    avg_momentum: float
    strength: float


@dataclass(frozen=True)
class PredictionResult:
    target: str
    target_kind: TargetKind
    plane: PhysicsPlane
    weights: list[WeightInfo]
    migration_map: list[MigrationInfo]
    bonding_map: list[BondInfo]
    iteration_gains: list[IterationInfo]
    equilibrium_zones: list[EquilibriumZone]
    metrics: PredictionMetrics
    preview_rows: list[dict[str, Any]]


class PredictorError(ValueError):
    pass


def _plane_negative_multiplier(plane: PhysicsPlane) -> float:
    # How strongly negative correlation acts as a "stumbling block".
    # solid: harsher penalty, liquid: medium, gas: softer.
    return {
        PhysicsPlane.solid: 1.6,
        PhysicsPlane.liquid: 1.0,
        PhysicsPlane.gas: 0.6,
    }[plane]


def _plane_mobility(plane: PhysicsPlane) -> float:
    # Plane controls baseline mobility through the gel-like medium.
    return {
        PhysicsPlane.solid: 0.85,
        PhysicsPlane.liquid: 1.0,
        PhysicsPlane.gas: 1.15,
    }[plane]


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        # Try a cheap parse on a small sample.
        sample = series.dropna().astype("string").head(25)
        if sample.empty:
            return False
        # Heuristic: only attempt parsing if values look date-ish.
        # Prevents noisy warnings for arbitrary short strings like 'x'/'y'.
        has_digit = sample.str.contains(r"\d", regex=True).mean()
        if has_digit < 0.6:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        return parsed.notna().mean() >= 0.8
    return False


def infer_target_kind(series: pd.Series) -> TargetKind:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if _is_datetime_like(series):
        return "datetime"
    # Try numeric coercion for object/string targets.
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() >= 0.9:
            return "numeric"
    return "categorical"


def infer_feature_kind(series: pd.Series) -> FeatureKind:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if _is_datetime_like(series):
        return "datetime"
    return "categorical"


def _to_float_array(series: pd.Series, *, kind: FeatureKind | TargetKind) -> np.ndarray:
    if kind == "datetime":
        dt = pd.to_datetime(series, errors="coerce", utc=False)
        arr = dt.to_numpy(dtype="datetime64[ns]")
        out = arr.astype("int64").astype("float64")
        out[pd.isna(arr)] = np.nan
        return out
    if kind == "bool":
        # pandas bool can contain NA if using boolean dtype
        s = series.astype("float64")
        return s.to_numpy(dtype="float64", na_value=np.nan)
    # numeric or something coercible
    s = pd.to_numeric(series, errors="coerce")
    return s.to_numpy(dtype="float64", na_value=np.nan)


def _to_category_array(series: pd.Series) -> np.ndarray:
    return series.astype("string").fillna("__MISSING__").to_numpy()


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x0 = x[mask]
    y0 = y[mask]
    if np.nanstd(x0) == 0 or np.nanstd(y0) == 0:
        return 0.0
    c = float(np.corrcoef(x0, y0)[0, 1])
    if math.isnan(c) or math.isinf(c):
        return 0.0
    return max(-1.0, min(1.0, c))


def _correlation_ratio(categories: np.ndarray, measurements: np.ndarray) -> float:
    mask = np.isfinite(measurements)
    if mask.sum() < 3:
        return 0.0
    cats = categories[mask]
    vals = measurements[mask]
    grand_mean = float(np.mean(vals))
    ss_total = float(np.sum((vals - grand_mean) ** 2))
    if ss_total <= 0:
        return 0.0

    ss_between = 0.0
    for cat in pd.unique(cats):
        idx = cats == cat
        n = int(np.sum(idx))
        if n == 0:
            continue
        mean_cat = float(np.mean(vals[idx]))
        ss_between += n * (mean_cat - grand_mean) ** 2

    eta2 = ss_between / ss_total
    eta2 = max(0.0, min(1.0, eta2))
    return float(math.sqrt(eta2))


def _cramers_v(x_cat: np.ndarray, y_cat: np.ndarray) -> float:
    # Basic Cramer's V without bias correction.
    if len(x_cat) != len(y_cat):
        raise PredictorError("Length mismatch")

    df = pd.DataFrame({"x": x_cat, "y": y_cat})
    df = df.dropna()
    if df.shape[0] < 3:
        return 0.0

    ct = pd.crosstab(df["x"], df["y"])
    obs = ct.to_numpy(dtype="float64")
    n = float(obs.sum())
    if n <= 0:
        return 0.0

    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0))

    r, k = obs.shape
    denom = min(r - 1, k - 1)
    if denom <= 0:
        return 0.0

    v = math.sqrt((chi2 / n) / denom)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return float(max(0.0, min(1.0, v)))


def _is_binary_categorical(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    return s.nunique() == 2


def _safe_probabilities(arr: np.ndarray) -> np.ndarray:
    p = arr.astype("float64")
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0:
        return np.array([1.0], dtype="float64")
    return p / s


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p0 = _safe_probabilities(p)
    q0 = _safe_probabilities(q)
    m = max(len(p0), len(q0))
    if len(p0) < m:
        p0 = np.pad(p0, (0, m - len(p0)), mode="constant")
    if len(q0) < m:
        q0 = np.pad(q0, (0, m - len(q0)), mode="constant")

    eps = 1e-12
    p1 = np.clip(p0, eps, None)
    q1 = np.clip(q0, eps, None)
    p1 = p1 / p1.sum()
    q1 = q1 / q1.sum()
    return float(np.sum(p1 * np.log(p1 / q1)))


def _numeric_distribution(values: np.ndarray, bins: int = 20) -> np.ndarray:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return np.ones(1, dtype="float64")
    if np.nanstd(v) <= 1e-12:
        return np.array([1.0], dtype="float64")
    hist, _ = np.histogram(v, bins=max(4, int(bins)), density=False)
    return _safe_probabilities(hist.astype("float64"))


def _categorical_distribution(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("__MISSING__")
    counts = s.value_counts(dropna=False)
    total = float(counts.sum())
    if total <= 0:
        return pd.Series([1.0], index=["__MISSING__"], dtype="float64")
    return (counts / total).astype("float64")


def _shannon_entropy_from_probs(probabilities: np.ndarray) -> float:
    p = _safe_probabilities(probabilities)
    eps = 1e-12
    h = float(-np.sum(p * np.log(np.clip(p, eps, None))))
    h_max = float(np.log(len(p))) if len(p) > 1 else 1.0
    return float(np.clip(h / max(h_max, 1e-12), 0.0, 1.0))


def _numeric_entropy_and_variance(values: np.ndarray) -> tuple[float, float, float]:
    v = values[np.isfinite(values)]
    n = len(v)
    if n < 3:
        return 1.0, 1.0, 1.0

    probs = _numeric_distribution(v)
    entropy = _shannon_entropy_from_probs(probs)

    std = float(np.std(v))
    stderr = std / math.sqrt(max(1.0, float(n)))
    var_norm = float((std**2) / (1.0 + std**2))
    return entropy, var_norm, stderr


def _categorical_entropy_and_variance(series: pd.Series) -> tuple[float, float, float]:
    probs = _categorical_distribution(series).to_numpy(dtype="float64")
    entropy = _shannon_entropy_from_probs(probs)
    gini = float(1.0 - np.sum(probs**2))
    n = int(series.shape[0])
    p_mode = float(np.max(probs)) if len(probs) else 1.0
    stderr = math.sqrt(max(0.0, p_mode * (1.0 - p_mode)) / max(1.0, float(n)))
    return entropy, float(np.clip(gini, 0.0, 1.0)), stderr


def _global_numeric_reference(df: pd.DataFrame, feature_kinds: dict[str, FeatureKind]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for col, fk in feature_kinds.items():
        if fk not in ("numeric", "datetime", "bool"):
            continue
        arr = _to_float_array(df[col], kind=fk)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            arrays.append(arr)
    if not arrays:
        return np.ones(1, dtype="float64")
    stacked = np.concatenate(arrays)
    return _numeric_distribution(stacked)


def _global_categorical_reference(df: pd.DataFrame, feature_kinds: dict[str, FeatureKind]) -> pd.Series:
    frames: list[pd.Series] = []
    for col, fk in feature_kinds.items():
        if fk in ("numeric", "datetime"):
            continue
        frames.append(df[col].astype("string").fillna("__MISSING__"))
    if not frames:
        return pd.Series([1.0], index=["__MISSING__"], dtype="float64")
    pooled = pd.concat(frames, ignore_index=True)
    return _categorical_distribution(pooled)


def _kl_for_feature(
    feature: pd.Series,
    feature_kind: FeatureKind,
    global_numeric_ref: np.ndarray,
    global_categorical_ref: pd.Series,
) -> float:
    if feature_kind in ("numeric", "datetime", "bool"):
        vals = _to_float_array(feature, kind=feature_kind)
        return _kl_divergence(_numeric_distribution(vals), global_numeric_ref)

    p = _categorical_distribution(feature)
    idx = p.index.union(global_categorical_ref.index)
    p_arr = p.reindex(idx, fill_value=0.0).to_numpy(dtype="float64")
    q_arr = global_categorical_ref.reindex(idx, fill_value=0.0).to_numpy(dtype="float64")
    return _kl_divergence(p_arr, q_arr)


def _migration_state(terminal_velocity: float, viscosity: float) -> Literal["free", "dampened", "trapped"]:
    speed = abs(float(terminal_velocity))
    if speed <= 1e-10:
        return "trapped"
    ratio = speed / max(viscosity, 1e-6)
    if ratio < 0.2:
        return "trapped"
    if ratio < 0.8:
        return "dampened"
    return "free"


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -60, 60))
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return ex / denom


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _feature_affinity(a: pd.Series, a_kind: FeatureKind, b: pd.Series, b_kind: FeatureKind) -> float:
    if a_kind in ("numeric", "datetime", "bool") and b_kind in ("numeric", "datetime", "bool"):
        return abs(_pearson_corr(_to_float_array(a, kind=a_kind), _to_float_array(b, kind=b_kind)))

    if a_kind in ("numeric", "datetime", "bool") and b_kind == "categorical":
        return _correlation_ratio(_to_category_array(b), _to_float_array(a, kind=a_kind))

    if b_kind in ("numeric", "datetime", "bool") and a_kind == "categorical":
        return _correlation_ratio(_to_category_array(a), _to_float_array(b, kind=b_kind))

    return _cramers_v(_to_category_array(a), _to_category_array(b))


def _build_bonding_map(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_kinds: dict[str, FeatureKind],
    *,
    top_pairs: int = 20,
    min_affinity: float = 0.08,
) -> list[BondInfo]:
    bonds: list[BondInfo] = []
    for i in range(len(feature_cols)):
        a = feature_cols[i]
        for j in range(i + 1, len(feature_cols)):
            b = feature_cols[j]
            affinity = _feature_affinity(df[a], feature_kinds[a], df[b], feature_kinds[b])
            if not math.isfinite(affinity) or affinity < min_affinity:
                continue
            bonds.append(
                BondInfo(
                    feature_a=a,
                    feature_b=b,
                    affinity=float(affinity),
                    bonding_factor=float(1.0 + affinity),
                )
            )
    bonds = sorted(bonds, key=lambda x: x.affinity, reverse=True)
    return bonds[: max(0, int(top_pairs))]


def _bonding_factors(feature_cols: list[str], bonds: list[BondInfo]) -> dict[str, float]:
    factors = {c: 1.0 for c in feature_cols}
    if not bonds:
        return factors
    grouped: dict[str, list[float]] = {c: [] for c in feature_cols}
    for b in bonds:
        grouped[b.feature_a].append(b.affinity)
        grouped[b.feature_b].append(b.affinity)
    for c in feature_cols:
        if grouped[c]:
            factors[c] = 1.0 + float(np.mean(grouped[c]))
    return factors


def _isoelectric_point(
    feature: pd.Series,
    feature_kind: FeatureKind,
    target: pd.Series,
    target_kind: TargetKind,
) -> float:
    w, _, _ = _compute_association(feature, target, feature_kind, target_kind)
    w_norm = np.clip(float(w), -1.0, 1.0)
    rank_percentile = 0.5 + (w_norm / 2.0)
    pI = float(rank_percentile)
    return pI


def _spatial_gradient_viscosity(
    row_idx: int,
    target_logits: np.ndarray,
    base_viscosity: float,
    *,
    gradient_scale: float = 0.5,
) -> float:
    if len(target_logits) == 0 or row_idx >= len(target_logits):
        return base_viscosity
    row_logit = target_logits[row_idx]
    certainty = float(np.tanh(row_logit / 10.0))
    gradient_effect = 1.0 - gradient_scale * abs(float(certainty))
    return float(base_viscosity * max(0.3, gradient_effect))


def _focusing_constant(feature_velocity: float, *, baseline_k: float = 0.3) -> float:
    vel = abs(float(feature_velocity))
    return float(baseline_k * (1.0 + vel))


def _kinetic_momentum(
    cycle: int,
    velocity_magnitude: np.ndarray,
    *,
    momentum_decay: float = 0.85,
) -> np.ndarray:
    if cycle == 1:
        return np.zeros_like(velocity_magnitude)
    momentum = velocity_magnitude * momentum_decay
    return momentum


def _restoring_force(
    current_position: np.ndarray,
    pI: float,
    focusing_k: float,
) -> np.ndarray:
    displacement = current_position - float(pI)
    restoring = -focusing_k * displacement
    return restoring


def _discretize_into_zones(
    feature_positions: dict[str, float],
    n_zones: int = 5,
) -> dict[str, int]:
    if not feature_positions:
        return {}
    positions = np.array(list(feature_positions.values()))
    bin_edges = np.linspace(positions.min() - 0.01, positions.max() + 0.01, n_zones + 1)
    zone_map = {}
    for feat, pos in feature_positions.items():
        zone = int(np.digitize(float(pos), bin_edges)) - 1
        zone = max(0, min(n_zones - 1, zone))
        zone_map[feat] = zone
    return zone_map


def _compute_association(
    feature: pd.Series,
    target: pd.Series,
    feature_kind: FeatureKind,
    target_kind: TargetKind,
) -> tuple[float, str, bool]:
    """Returns (weight, method, signed).

    - Signed weights only make sense when the direction is interpretable.
    - Many category-based association measures are non-negative.
    """

    if target_kind in ("numeric", "datetime"):
        y = _to_float_array(target, kind=target_kind)
        if feature_kind in ("numeric", "datetime", "bool"):
            x = _to_float_array(feature, kind=feature_kind)
            return _pearson_corr(x, y), "pearson", True

        x_cat = _to_category_array(feature)
        return _correlation_ratio(x_cat, y), "eta", False

    # target is categorical
    y_cat = target.astype("string").fillna("__MISSING__")

    if feature_kind in ("numeric", "datetime", "bool"):
        x = _to_float_array(feature, kind=feature_kind)

        # If binary target, we can use point-biserial (pearson with 0/1 coding).
        if _is_binary_categorical(y_cat):
            labels = pd.unique(y_cat.dropna())
            positive = str(labels[0])
            y01 = (y_cat.astype("string") == positive).astype("float64").to_numpy(dtype="float64")
            return _pearson_corr(x, y01), "point_biserial", True

        # Multi-class: magnitude-only association.
        return _correlation_ratio(_to_category_array(y_cat), x), "eta", False

    # categorical feature vs categorical target
    x_cat = _to_category_array(feature)
    return _cramers_v(x_cat, _to_category_array(y_cat)), "cramers_v", False


def _zscore(a: np.ndarray) -> np.ndarray:
    out = a.astype("float64")
    m = float(np.nanmean(out)) if np.isfinite(np.nanmean(out)) else 0.0
    s = float(np.nanstd(out)) if np.isfinite(np.nanstd(out)) else 0.0
    if s <= 1e-12:
        s = 1.0
    out = np.where(np.isfinite(out), out, m)
    return (out - m) / s


def _zscore_with_train_stats(all_values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train_vals = all_values[train_mask]
    m = float(np.nanmean(train_vals)) if np.isfinite(np.nanmean(train_vals)) else 0.0
    s = float(np.nanstd(train_vals)) if np.isfinite(np.nanstd(train_vals)) else 0.0
    if s <= 1e-12:
        s = 1.0
    filled = np.where(np.isfinite(all_values), all_values, m)
    return (filled - m) / s


def _train_test_split_mask(n: int, train_fraction: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    tf = float(train_fraction)
    if tf >= 0.999:
        # "No split" mode: train == test == full dataset
        train_mask = np.ones(n, dtype=bool)
        test_mask = np.ones(n, dtype=bool)
        return train_mask, test_mask
    if not (0.05 <= tf <= 0.95):
        raise PredictorError("train_fraction must be between 0.05 and 0.95 (or 1.0 for no split)")
    if n < 3:
        raise PredictorError("Need at least 3 rows")

    rng = np.random.default_rng(int(random_seed))
    idx = rng.permutation(n)
    n_train = int(round(n * tf))
    n_train = max(1, min(n - 1, n_train))

    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def _encode_feature_numeric(
    feature: pd.Series,
    feature_kind: FeatureKind,
    target_numeric: np.ndarray,
    *,
    target_is_finite_mask: np.ndarray,
) -> np.ndarray:
    if feature_kind in ("numeric", "datetime", "bool"):
        return _to_float_array(feature, kind=feature_kind)

    # categorical -> target mean encoding
    x_cat = feature.astype("string").fillna("__MISSING__")
    df = pd.DataFrame({"x": x_cat, "y": target_numeric})
    df = df.loc[target_is_finite_mask]
    if df.empty:
        # fallback: all zeros
        return np.zeros(len(feature), dtype="float64")

    means = df.groupby("x")["y"].mean()
    overall = float(df["y"].mean())
    encoded = x_cat.map(means).fillna(overall)
    return encoded.to_numpy(dtype="float64")


def _select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    return [c for c in df.columns if c != target_col]


def _numeric_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return 0.0, 0.0
    err = y_pred[mask] - y_true[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(float(np.mean(err**2))))
    return mae, rmse


def run_physics_prediction(
    df: pd.DataFrame,
    *,
    target_col: str,
    plane: PhysicsPlane = PhysicsPlane.solid,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    top_k_weights: int = 30,
    max_preview_rows: int = 25,
    max_classes: int = 20,
    n_cycles: int = 30,
    cycle_learning_rate: float = 0.18,
    shear_alpha: float = 0.75,
    top_bond_pairs: int = 20,
    n_zones: int = 5,
    cascade_enabled: bool = True,
    competitive_inhibition: bool = True,
    thermal_noise: bool = False,
    thermal_noise_cycles: int = 3,
    thermal_noise_level: float = 0.10,
    stage2_cycles: int = 2,
    stage2_voltage_multiplier: float = 2.0,
    inhibition_strength: float = 0.7,
    scavenger_cycles: int = 1,
) -> PredictionResult:
    if target_col not in df.columns:
        raise PredictorError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    if df.shape[0] < 3:
        raise PredictorError("Need at least 3 rows")

    train_mask, test_mask = _train_test_split_mask(int(df.shape[0]), train_fraction, random_seed)

    target_series = df[target_col]
    target_kind = infer_target_kind(target_series)

    feature_cols = _select_feature_columns(df, target_col)
    if not feature_cols:
        raise PredictorError("No features available (dataset only contains the target column)")

    feature_kinds: dict[str, FeatureKind] = {c: infer_feature_kind(df[c]) for c in feature_cols}
    bonds = _build_bonding_map(df[feature_cols][train_mask], feature_cols, feature_kinds, top_pairs=top_bond_pairs)
    bond_factors = _bonding_factors(feature_cols, bonds)

    pI_map: dict[str, float] = {}
    for col in feature_cols:
        pI_map[col] = _isoelectric_point(df[col], feature_kinds[col], target_series, target_kind)

    global_numeric_ref = _global_numeric_reference(df[feature_cols], feature_kinds)
    global_categorical_ref = _global_categorical_reference(df[feature_cols], feature_kinds)

    # Compute association weights on TRAIN only (for explanation + feature selection)
    weights: list[WeightInfo] = []
    migration_map: list[MigrationInfo] = []
    plane_mobility = _plane_mobility(plane)
    neg_mult = _plane_negative_multiplier(plane)
    for col in feature_cols:
        feat = df[col]
        fk = feature_kinds[col]
        w, method, signed = _compute_association(feat[train_mask], target_series[train_mask], fk, target_kind)
        if not math.isfinite(w):
            w = 0.0
        weights.append(WeightInfo(feature=col, weight=float(w), method=method, feature_kind=fk, signed=signed))

        if fk in ("numeric", "datetime", "bool"):
            entropy, variance, stderr = _numeric_entropy_and_variance(_to_float_array(feat[train_mask], kind=fk))
        else:
            entropy, variance, stderr = _categorical_entropy_and_variance(feat[train_mask])

        kl = _kl_for_feature(feat[train_mask], fk, global_numeric_ref, global_categorical_ref)
        certainty = 1.0 / (1.0 + max(0.0, float(stderr)))
        density = (1.0 + max(0.0, kl)) * certainty
        viscosity = max(1e-6, entropy + variance)
        bond_factor = bond_factors.get(col, 1.0)

        charge = float(w)
        if charge < 0:
            charge *= neg_mult
        terminal_velocity = plane_mobility * (charge * density * bond_factor) / viscosity

        if terminal_velocity > 1e-10:
            direction: Literal["pulled", "repelled", "neutral"] = "pulled"
        elif terminal_velocity < -1e-10:
            direction = "repelled"
        else:
            direction = "neutral"

        migration_map.append(
            MigrationInfo(
                feature=col,
                feature_kind=fk,
                method=method,
                charge=float(w),
                entropy=float(entropy),
                variance=float(variance),
                standard_error=float(stderr),
                kl_divergence=float(kl),
                density=float(density),
                viscosity=float(viscosity),
                terminal_velocity=float(terminal_velocity),
                arrival_speed=float(abs(terminal_velocity)),
                direction=direction,
                state=_migration_state(terminal_velocity, viscosity),
            )
        )

    weights_sorted = sorted(weights, key=lambda wi: abs(wi.weight), reverse=True)
    weights_used = [w for w in weights_sorted if abs(w.weight) > 1e-8]

    # If everything is ~0, keep a few anyway so the UI can show something.
    if not weights_used:
        weights_used = weights_sorted[: min(10, len(weights_sorted))]

    # Keep only top-k for prediction.
    weights_used = weights_used[: max(1, min(top_k_weights, len(weights_used)))]

    migration_by_feature = {m.feature: m for m in migration_map}

    if target_kind in ("numeric", "datetime"):
        # Stage 1 zones for numeric targets too.
        feature_cols_used = [w.feature for w in weights_used]
        weights_by_feature: dict[str, WeightInfo] = {w.feature: w for w in weights_used}
        feature_positions_stage1: dict[str, float] = {col: pI_map.get(col, 0.5) for col in feature_cols_used}
        zone_assignment_stage1 = _discretize_into_zones(feature_positions_stage1, n_zones=n_zones)
        zone_bins_stage1: dict[int, list[str]] = {i: [] for i in range(n_zones)}
        for feat, zid in zone_assignment_stage1.items():
            zone_bins_stage1[int(zid)].append(feat)
        zone1_features = list(max(zone_bins_stage1.values(), key=lambda xs: len(xs), default=[]))

        dominant_global = max(feature_cols_used, key=lambda f: abs(migration_by_feature[f].terminal_velocity))
        dominant_global_kind = feature_kinds[dominant_global]
        affinity_to_global_dominant: dict[str, float] = {}
        for f in feature_cols_used:
            if f == dominant_global:
                affinity_to_global_dominant[f] = 0.0
            else:
                affinity_to_global_dominant[f] = float(
                    _feature_affinity(
                        df[f][train_mask],
                        feature_kinds[f],
                        df[dominant_global][train_mask],
                        dominant_global_kind,
                    )
                )

        if len(zone1_features) < 3 and len(feature_cols_used) >= 3:
            neighbors: list[tuple[float, str]] = []
            for b in bonds:
                if b.feature_a == dominant_global:
                    neighbors.append((float(b.affinity), b.feature_b))
                elif b.feature_b == dominant_global:
                    neighbors.append((float(b.affinity), b.feature_a))
            neighbors.sort(key=lambda t: t[0], reverse=True)
            zone1_features = [dominant_global] + [n for _, n in neighbors if n != dominant_global]
            if len(zone1_features) < 3:
                for f in feature_cols_used:
                    if f not in zone1_features:
                        zone1_features.append(f)
                    if len(zone1_features) >= 3:
                        break
            zone1_features = zone1_features[: min(len(zone1_features), len(feature_cols_used))]

        y = _to_float_array(target_series, kind=target_kind)
        y_train = y[train_mask]
        y_train_mask = np.isfinite(y_train)
        y_mean = float(np.nanmean(y_train)) if y_train_mask.any() else 0.0

        # Baseline predictor: train mean.
        pred = np.full(df.shape[0], y_mean, dtype="float64")
        baseline_mae, baseline_rmse = _numeric_metrics(y[test_mask], pred[test_mask])
        if target_kind == "datetime":
            baseline_mae /= 1e9
            baseline_rmse /= 1e9

        # Precompute feature z-scores (train stats) for stability.
        z_by_feature: dict[str, np.ndarray] = {}
        for wi in weights_used:
            feat = df[wi.feature]
            x_raw = _encode_feature_numeric(
                feat,
                wi.feature_kind,
                y,
                target_is_finite_mask=np.isfinite(y) & train_mask,
            )
            z_by_feature[wi.feature] = _zscore_with_train_stats(x_raw, train_mask)

        iteration_gains: list[IterationInfo] = []
        n_cycles_eff = max(1, int(n_cycles))
        lr = float(cycle_learning_rate)
        grad1 = 0.35
        rng_stage1 = np.random.default_rng(int(random_seed) + 31007)

        for cycle in range(1, n_cycles_eff + 1):
            residual = y - pred
            residual_train = residual[train_mask]
            residual_std = float(np.nanstd(residual_train))
            if not math.isfinite(residual_std) or residual_std <= 1e-12:
                residual_std = 1.0

            update_score = np.zeros(df.shape[0], dtype="float64")
            denom = 0.0

            for wi in weights_used:
                col = wi.feature
                z = z_by_feature[col]
                charge = _pearson_corr(z[train_mask], residual_train)
                if not math.isfinite(charge) or abs(charge) < 1e-8:
                    continue

                medium = migration_by_feature[col]
                bond_factor = bond_factors.get(col, 1.0)
                certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                density = (1.0 + max(0.0, medium.kl_divergence)) * certainty
                eta_base = max(1e-6, medium.entropy + medium.variance)
                eta_dynamic = max(1e-6, eta_base / (1.0 + shear_alpha * abs(charge) * bond_factor))

                inhibition = 0.0
                if competitive_inhibition and col != dominant_global:
                    inhibition = float(inhibition_strength) * abs(float(affinity_to_global_dominant.get(col, 0.0)))

                thermal_term = 0.0
                if thermal_noise and cycle <= int(thermal_noise_cycles):
                    eta_dynamic = max(
                        1e-6,
                        eta_dynamic
                        * (1.0 + rng_stage1.uniform(-float(thermal_noise_level), float(thermal_noise_level))),
                    )
                    thermal_term = float(rng_stage1.normal(0.0, float(thermal_noise_level))) * abs(float(charge))

                eff_charge = float(charge)
                if eff_charge < 0:
                    eff_charge *= neg_mult
                q = eff_charge * density * bond_factor
                x_pos = float(pI_map.get(col, 0.5))
                field = float(plane_mobility) - float(grad1) * x_pos
                v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                denom += abs(v)
                update_score += v * z

            if denom <= 1e-12:
                denom = 1.0
            pred = pred + lr * (update_score / denom) * residual_std

            mae, rmse = _numeric_metrics(y[test_mask], pred[test_mask])
            if target_kind == "datetime":
                mae /= 1e9
                rmse /= 1e9
            iteration_gains.append(
                IterationInfo(
                    cycle=cycle,
                    test_mae=float(mae),
                    test_rmse=float(rmse),
                    lift_over_baseline=float(baseline_rmse - rmse),
                )
            )

        # Stage 2 cascade (fractionate Zone 1)
        shattered_zones: list[EquilibriumZone] = []
        if cascade_enabled and len(zone1_features) >= 3:
            global_var = float(np.mean([m.variance for m in migration_map])) if migration_map else 1.0
            cluster_var = float(np.mean([migration_by_feature[f].variance for f in zone1_features]))
            grad2 = float(np.clip(0.55 * (cluster_var / (global_var + 1e-9)), 0.05, 0.9))
            E2 = float(stage2_voltage_multiplier) * float(plane_mobility)
            lr2 = float(cycle_learning_rate) * 0.6
            rng_stage2 = np.random.default_rng(int(random_seed) + 41011)

            dominant = max(zone1_features, key=lambda f: abs(migration_by_feature[f].terminal_velocity))
            dominant_kind = feature_kinds[dominant]
            affinity_to_dominant: dict[str, float] = {}
            for f in zone1_features:
                if f == dominant:
                    affinity_to_dominant[f] = 0.0
                else:
                    affinity_to_dominant[f] = float(
                        _feature_affinity(
                            df[f][train_mask],
                            feature_kinds[f],
                            df[dominant][train_mask],
                            dominant_kind,
                        )
                    )

            for stage_cycle in range(1, max(1, int(stage2_cycles)) + 1):
                residual = y - pred
                residual_train = residual[train_mask]
                residual_std = float(np.nanstd(residual_train))
                if not math.isfinite(residual_std) or residual_std <= 1e-12:
                    residual_std = 1.0

                update_score = np.zeros(df.shape[0], dtype="float64")
                denom = 0.0

                for col in zone1_features:
                    z = z_by_feature[col]
                    charge = _pearson_corr(z[train_mask], residual_train)
                    if not math.isfinite(charge) or abs(charge) < 1e-8:
                        continue

                    medium = migration_by_feature[col]
                    bond_factor = bond_factors.get(col, 1.0)
                    certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                    density = (1.0 + max(0.0, medium.kl_divergence)) * certainty
                    eta_base = max(1e-6, medium.entropy + medium.variance)
                    eta_dynamic = max(1e-6, eta_base / (1.0 + shear_alpha * abs(charge) * bond_factor))

                    inhibition = 0.0
                    if competitive_inhibition and col != dominant:
                        inhibition = float(inhibition_strength) * abs(float(affinity_to_dominant.get(col, 0.0)))

                    thermal_term = 0.0
                    if thermal_noise and stage_cycle <= int(thermal_noise_cycles):
                        eta_dynamic = max(
                            1e-6,
                            eta_dynamic
                            * (1.0 + rng_stage2.uniform(-float(thermal_noise_level), float(thermal_noise_level))),
                        )
                        thermal_term = float(rng_stage2.normal(0.0, float(thermal_noise_level))) * abs(float(charge))

                    eff_charge = float(charge)
                    if eff_charge < 0:
                        eff_charge *= neg_mult
                    q = eff_charge * density * bond_factor
                    x_pos = float(pI_map.get(col, 0.5))
                    field = E2 - grad2 * x_pos
                    v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                    denom += abs(v)
                    update_score += v * z

                if denom <= 1e-12:
                    denom = 1.0
                pred = pred + lr2 * (update_score / denom) * residual_std

                mae, rmse = _numeric_metrics(y[test_mask], pred[test_mask])
                if target_kind == "datetime":
                    mae /= 1e9
                    rmse /= 1e9
                iteration_gains.append(
                    IterationInfo(
                        cycle=n_cycles_eff + stage_cycle,
                        test_mae=float(mae),
                        test_rmse=float(rmse),
                        lift_over_baseline=float(baseline_rmse - rmse),
                    )
                )

            # Cluster shattering: split Zone 1 into 3 sub-zones for output.
            zone1_sorted = sorted(zone1_features, key=lambda f: float(pI_map.get(f, 0.5)))
            subzones = [list(chunk) for chunk in np.array_split(np.array(zone1_sorted, dtype=object), 3) if len(chunk) > 0]
            for k, feats in enumerate(subzones[:3]):
                feats_list = [str(x) for x in feats]
                shattered_zones.append(
                    EquilibriumZone(
                        zone_id=100 + k,
                        features=feats_list,
                        avg_pI=float(np.mean([pI_map.get(f, 0.5) for f in feats_list])),
                        avg_momentum=0.0,
                        strength=float(len(feats_list) / max(1, len(zone1_features))),
                    )
                )

            # Scavenger pass: recycle the weakest third of Zone 1.
            waste_features = sorted(zone1_features, key=lambda f: abs(float(weights_by_feature[f].weight)))
            waste_features = waste_features[: max(1, len(waste_features) // 3)]
            if int(scavenger_cycles) > 0 and waste_features:
                lr3 = float(cycle_learning_rate) * 0.25
                rng_scav = np.random.default_rng(int(random_seed) + 51017)
                for sc_cycle in range(1, int(scavenger_cycles) + 1):
                    residual = y - pred
                    residual_train = residual[train_mask]
                    residual_std = float(np.nanstd(residual_train))
                    if not math.isfinite(residual_std) or residual_std <= 1e-12:
                        residual_std = 1.0

                    update_score = np.zeros(df.shape[0], dtype="float64")
                    denom = 0.0
                    for col in waste_features:
                        z = z_by_feature[col]
                        charge = _pearson_corr(z[train_mask], residual_train)
                        if not math.isfinite(charge) or abs(charge) < 1e-8:
                            continue

                        medium = migration_by_feature[col]
                        bond_factor = bond_factors.get(col, 1.0)
                        certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                        density = (1.0 + max(0.0, medium.kl_divergence)) * certainty
                        eta_base = max(1e-6, medium.entropy + medium.variance)
                        eta_dynamic = max(1e-6, eta_base / (1.0 + shear_alpha * abs(charge) * bond_factor))
                        if thermal_noise and sc_cycle <= int(thermal_noise_cycles):
                            eta_dynamic = max(
                                1e-6,
                                eta_dynamic
                                * (1.0 + rng_scav.uniform(-float(thermal_noise_level), float(thermal_noise_level))),
                            )

                        eff_charge = float(charge)
                        if eff_charge < 0:
                            eff_charge *= neg_mult
                        q = eff_charge * density * bond_factor
                        x_pos = float(pI_map.get(col, 0.5))
                        field = float(plane_mobility) - grad2 * x_pos
                        v = q * field / eta_dynamic

                        denom += abs(v)
                        update_score += v * z

                    if denom <= 1e-12:
                        denom = 1.0
                    pred = pred + lr3 * (update_score / denom) * residual_std

                    mae, rmse = _numeric_metrics(y[test_mask], pred[test_mask])
                    if target_kind == "datetime":
                        mae /= 1e9
                        rmse /= 1e9
                    iteration_gains.append(
                        IterationInfo(
                            cycle=n_cycles_eff + max(1, int(stage2_cycles)) + sc_cycle,
                            test_mae=float(mae),
                            test_rmse=float(rmse),
                            lift_over_baseline=float(baseline_rmse - rmse),
                        )
                    )

        mae, rmse = _numeric_metrics(y[test_mask], pred[test_mask])
        if target_kind == "datetime":
            mae /= 1e9
            rmse /= 1e9

        best_iter = min(iteration_gains, key=lambda it: float(it.test_rmse) if it.test_rmse is not None else 1e99) if iteration_gains else None
        best_cycle = None if best_iter is None else int(best_iter.cycle)
        best_lift = None if best_iter is None else float(best_iter.lift_over_baseline or 0.0)

        preview = []
        test_indices = np.flatnonzero(test_mask)
        for idx in test_indices[: min(max_preview_rows, len(test_indices))]:
            i = int(idx)
            actual = y[i]
            predicted = pred[i]
            if target_kind == "datetime":
                actual_disp = None if not math.isfinite(actual) else pd.to_datetime(int(actual)).isoformat()
                pred_disp = None if not math.isfinite(predicted) else pd.to_datetime(int(predicted)).isoformat()
            else:
                actual_disp = None if not math.isfinite(actual) else float(actual)
                pred_disp = None if not math.isfinite(predicted) else float(predicted)
            preview.append({"row": i, "actual": actual_disp, "predicted": pred_disp})

        metrics = PredictionMetrics(
            target_kind=target_kind,
            n_rows=int(df.shape[0]),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            train_fraction=float(train_fraction),
            random_seed=int(random_seed),
            n_features_used=len(weights_used),
            mae=float(mae),
            rmse=float(rmse),
            baseline_mae=float(baseline_mae),
            baseline_rmse=float(baseline_rmse),
            best_cycle=best_cycle,
            best_lift=best_lift,
        )

        equilibrium_zones: list[EquilibriumZone] = []
        for zone_id in range(n_zones):
            feats = zone_bins_stage1.get(zone_id, [])
            if feats:
                avg_pI = float(np.mean([pI_map.get(f, 0.5) for f in feats]))
                strength = float(len(feats) / len(feature_cols_used)) if feature_cols_used else 0.0
                equilibrium_zones.append(
                    EquilibriumZone(
                        zone_id=zone_id,
                        features=feats,
                        avg_pI=avg_pI,
                        avg_momentum=0.0,
                        strength=strength,
                    )
                )
        if shattered_zones:
            equilibrium_zones.extend(shattered_zones)

        return PredictionResult(
            target=target_col,
            target_kind=target_kind,
            plane=plane,
            weights=weights_used,
            migration_map=sorted(migration_map, key=lambda m: m.arrival_speed, reverse=True),
            bonding_map=bonds,
            equilibrium_zones=equilibrium_zones,
            iteration_gains=iteration_gains,
            metrics=metrics,
            preview_rows=preview,
        )

    # Categorical target: one-vs-rest scoring for up to max_classes classes.
    y_cat = target_series.astype("string").fillna("__MISSING__")
    # Determine classes from TRAIN only to avoid leaking rare/unseen labels.
    classes = list(pd.Series(y_cat[train_mask]).value_counts().index)
    if len(classes) > max_classes:
        raise PredictorError(
            f"Too many classes in target ({len(classes)}). Limit is {max_classes}. "
            "Consider filtering or choosing a different target."
        )

    # Use weights based on overall association for display; but build per-class weights for scoring.
    feature_info = {w.feature: w for w in weights_used}
    feature_cols_used = [w.feature for w in weights_used]

    # Stage 1 (Primary Sorting): compute stable zones based on pI positions.
    feature_positions_stage1: dict[str, float] = {col: pI_map.get(col, 0.5) for col in feature_cols_used}
    zone_assignment_stage1 = _discretize_into_zones(feature_positions_stage1, n_zones=n_zones)
    zone_bins_stage1: dict[int, list[str]] = {i: [] for i in range(n_zones)}
    for feat, zid in zone_assignment_stage1.items():
        zone_bins_stage1[int(zid)].append(feat)
    # Zone 1 complex = highest interaction bin (largest by feature count).
    zone1_features = list(max(zone_bins_stage1.values(), key=lambda xs: len(xs), default=[]))

    dominant_global = max(feature_cols_used, key=lambda f: abs(migration_by_feature[f].terminal_velocity))
    dominant_global_kind = feature_kinds[dominant_global]
    affinity_to_global_dominant: dict[str, float] = {}
    for f in feature_cols_used:
        if f == dominant_global:
            affinity_to_global_dominant[f] = 0.0
        else:
            affinity_to_global_dominant[f] = float(
                _feature_affinity(
                    df[f][train_mask],
                    feature_kinds[f],
                    df[dominant_global][train_mask],
                    dominant_global_kind,
                )
            )

    # If the largest bin is too small to fractionate, define Zone 1 as the dominant feature
    # plus its strongest bonded neighbors (high-interaction complex).
    if len(zone1_features) < 3 and len(feature_cols_used) >= 3:
        neighbors: list[tuple[float, str]] = []
        for b in bonds:
            if b.feature_a == dominant_global:
                neighbors.append((float(b.affinity), b.feature_b))
            elif b.feature_b == dominant_global:
                neighbors.append((float(b.affinity), b.feature_a))
        neighbors.sort(key=lambda t: t[0], reverse=True)
        zone1_features = [dominant_global] + [n for _, n in neighbors if n != dominant_global]
        # Pad by weight if needed.
        if len(zone1_features) < 3:
            for f in feature_cols_used:
                if f not in zone1_features:
                    zone1_features.append(f)
                if len(zone1_features) >= 3:
                    break
        zone1_features = zone1_features[: min(len(zone1_features), len(feature_cols_used))]

    # Precompute feature numeric encodings for binary targets.
    x_encoded_by_feature: dict[str, np.ndarray] = {}
    for col in feature_cols_used:
        fk = feature_info[col].feature_kind
        if fk in ("numeric", "datetime", "bool"):
            x_encoded_by_feature[col] = _to_float_array(df[col], kind=fk)
        else:
            # placeholder; will encode per-class
            x_encoded_by_feature[col] = np.zeros(df.shape[0], dtype="float64")

    priors: list[float] = []
    logits = np.zeros((df.shape[0], len(classes)), dtype="float64")
    for j, cls in enumerate(classes):
        y01 = (y_cat == str(cls)).astype("float64").to_numpy(dtype="float64")
        prior = float(np.clip(y01[train_mask].mean(), 1e-9, 1 - 1e-9))
        priors.append(prior)
        logits[:, j] = math.log(prior)

    baseline_label = str(pd.Series(y_cat[train_mask]).value_counts().index[0])
    baseline_accuracy = float(np.mean((y_cat[test_mask] == baseline_label).astype("float64")))
    iteration_gains: list[IterationInfo] = []

    n_cycles_eff = max(1, int(n_cycles))
    lr = float(cycle_learning_rate)
    shear = max(0.0, float(shear_alpha))
    grad1 = 0.35
    rng_stage1 = np.random.default_rng(int(random_seed) + 30011)

    for cycle in range(1, n_cycles_eff + 1):
        cycle_update = np.zeros_like(logits)
        for j, cls in enumerate(classes):
            y01 = (y_cat == str(cls)).astype("float64").to_numpy(dtype="float64")
            p = _sigmoid(logits[:, j])
            residual = y01 - p
            residual_train = residual[train_mask]

            class_score = np.zeros(df.shape[0], dtype="float64")
            denom = 0.0

            for col in feature_cols_used:
                fk = feature_info[col].feature_kind
                if fk in ("numeric", "datetime", "bool"):
                    x_raw = x_encoded_by_feature[col]
                else:
                    x_cat = df[col].astype("string").fillna("__MISSING__")
                    tmp = pd.DataFrame({"x": x_cat[train_mask], "r": residual_train})
                    rates = tmp.groupby("x")["r"].mean()
                    x_raw = x_cat.map(rates).fillna(0.0).to_numpy(dtype="float64")

                z = _zscore_with_train_stats(x_raw, train_mask)
                charge = _pearson_corr(z[train_mask], residual_train)
                if not math.isfinite(charge) or abs(charge) < 1e-8:
                    continue

                medium = migration_by_feature[col]
                bond_factor = bond_factors.get(col, 1.0)
                certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                density = (1.0 + max(0.0, medium.kl_divergence)) * certainty
                eta_base = max(1e-6, medium.entropy + medium.variance)
                eta_dynamic = max(1e-6, eta_base / (1.0 + shear * abs(charge) * bond_factor))

                inhibition = 0.0
                if competitive_inhibition and col != dominant_global:
                    inhibition = float(inhibition_strength) * abs(float(affinity_to_global_dominant.get(col, 0.0)))

                thermal_term = 0.0
                if thermal_noise and cycle <= int(thermal_noise_cycles):
                    eta_dynamic = max(
                        1e-6,
                        eta_dynamic
                        * (1.0 + rng_stage1.uniform(-float(thermal_noise_level), float(thermal_noise_level))),
                    )
                    thermal_term = float(rng_stage1.normal(0.0, float(thermal_noise_level))) * abs(float(charge))

                eff_charge = float(charge)
                if eff_charge < 0:
                    eff_charge *= neg_mult
                q = eff_charge * density * bond_factor
                x_pos = float(pI_map.get(col, 0.5))
                field = float(plane_mobility) - float(grad1) * x_pos
                v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                denom += abs(v)
                class_score += v * z

            if denom <= 1e-12:
                denom = 1.0
            cycle_update[:, j] = class_score / denom

        logits += lr * cycle_update
        probs = _softmax(logits)
        pred_idx_cycle = np.argmax(probs, axis=1)
        pred_cycle = np.array([classes[int(i)] for i in pred_idx_cycle], dtype="object")
        test_acc = float(np.mean((y_cat[test_mask].to_numpy(dtype="object") == pred_cycle[test_mask]).astype("float64")))
        iteration_gains.append(
            IterationInfo(
                cycle=cycle,
                test_accuracy=test_acc,
                lift_over_baseline=test_acc - baseline_accuracy,
            )
        )

    # Stage 2 (Fractional Distillation): automatically re-focus the strongest cluster (Zone 1)
    # with higher field strength + narrower gradient, plus inhibition + thermal noise.
    if cascade_enabled and len(zone1_features) >= 3:
        # Narrowed gradient: use variance within the cluster relative to global feature variance.
        global_var = float(np.mean([m.variance for m in migration_map])) if migration_map else 1.0
        cluster_var = float(np.mean([migration_by_feature[f].variance for f in zone1_features]))
        grad = float(np.clip(0.55 * (cluster_var / (global_var + 1e-9)), 0.05, 0.9))
        E = float(stage2_voltage_multiplier) * float(plane_mobility)

        rng = np.random.default_rng(int(random_seed) + 40007)
        lr2 = float(cycle_learning_rate) * 0.75

        dominant = max(zone1_features, key=lambda f: abs(migration_by_feature[f].terminal_velocity))
        dominant_kind = feature_kinds[dominant]
        affinity_to_dominant: dict[str, float] = {}
        for f in zone1_features:
            if f == dominant:
                affinity_to_dominant[f] = 0.0
            else:
                affinity_to_dominant[f] = float(
                    _feature_affinity(
                        df[f][train_mask],
                        feature_kinds[f],
                        df[dominant][train_mask],
                        dominant_kind,
                    )
                )

        n2 = max(1, int(stage2_cycles))
        for stage_cycle in range(1, n2 + 1):
            cycle_update = np.zeros_like(logits)
            for j, cls in enumerate(classes):
                y01 = (y_cat == str(cls)).astype("float64").to_numpy(dtype="float64")
                p = _sigmoid(logits[:, j])
                residual = y01 - p
                residual_train = residual[train_mask]

                class_score = np.zeros(df.shape[0], dtype="float64")
                denom = 0.0

                for col in zone1_features:
                    fk = feature_info[col].feature_kind
                    if fk in ("numeric", "datetime", "bool"):
                        x_raw = x_encoded_by_feature[col]
                    else:
                        x_cat = df[col].astype("string").fillna("__MISSING__")
                        tmp = pd.DataFrame({"x": x_cat[train_mask], "r": residual_train})
                        rates = tmp.groupby("x")["r"].mean()
                        x_raw = x_cat.map(rates).fillna(0.0).to_numpy(dtype="float64")

                    z = _zscore_with_train_stats(x_raw, train_mask)
                    charge = _pearson_corr(z[train_mask], residual_train)
                    if not math.isfinite(charge) or abs(charge) < 1e-8:
                        continue

                    medium = migration_by_feature[col]
                    bond_factor = bond_factors.get(col, 1.0)
                    certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                    density = (1.0 + max(0.0, medium.kl_divergence)) * certainty

                    eta_base = max(1e-6, medium.entropy + medium.variance)
                    eta_dynamic = max(1e-6, eta_base / (1.0 + shear * abs(charge) * bond_factor))

                    inhibition = 0.0
                    if competitive_inhibition and col != dominant:
                        inhibition = float(inhibition_strength) * abs(float(affinity_to_dominant.get(col, 0.0)))

                    thermal_term = 0.0
                    if thermal_noise and stage_cycle <= int(thermal_noise_cycles):
                        eta_dynamic = max(
                            1e-6,
                            eta_dynamic * (1.0 + rng.uniform(-float(thermal_noise_level), float(thermal_noise_level))),
                        )
                        thermal_term = float(rng.normal(0.0, float(thermal_noise_level))) * abs(float(charge))

                    eff_charge = float(charge)
                    if eff_charge < 0:
                        eff_charge *= neg_mult
                    q = eff_charge * density * bond_factor
                    x_pos = float(pI_map.get(col, 0.5))

                    # v4 focusing: v = [q(E - ∇pH * x) + T_noise] / [η + I_competitive]
                    field = E - grad * x_pos
                    v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                    denom += abs(v)
                    class_score += v * z

                if denom <= 1e-12:
                    denom = 1.0
                cycle_update[:, j] = class_score / denom

            logits += lr2 * cycle_update
            probs = _softmax(logits)
            pred_idx_cycle = np.argmax(probs, axis=1)
            pred_cycle = np.array([classes[int(i)] for i in pred_idx_cycle], dtype="object")
            test_acc = float(
                np.mean((y_cat[test_mask].to_numpy(dtype="object") == pred_cycle[test_mask]).astype("float64"))
            )
            iteration_gains.append(
                IterationInfo(
                    cycle=n_cycles_eff + stage_cycle,
                    test_accuracy=test_acc,
                    lift_over_baseline=test_acc - baseline_accuracy,
                )
            )

        # Cluster shattering: split Zone 1 into at least 3 sub-zones by pI quantiles.
        zone1_sorted = sorted(zone1_features, key=lambda f: float(pI_map.get(f, 0.5)))
        subzones = [list(chunk) for chunk in np.array_split(np.array(zone1_sorted, dtype=object), 3) if len(chunk) > 0]
        shattered_zones: list[EquilibriumZone] = []
        for k, feats in enumerate(subzones[:3]):
            feats_list = [str(x) for x in feats]
            shattered_zones.append(
                EquilibriumZone(
                    zone_id=100 + k,
                    features=feats_list,
                    avg_pI=float(np.mean([pI_map.get(f, 0.5) for f in feats_list])),
                    avg_momentum=float(0.0),
                    strength=float(len(feats_list) / max(1, len(zone1_features))),
                )
            )

        # Residual recycling: send the "waste" (lowest-signal third) back for a scavenger pass.
        waste_features = sorted(zone1_features, key=lambda f: abs(float(feature_info[f].weight)))
        waste_features = waste_features[: max(1, len(waste_features) // 3)]
        if scavenger_cycles > 0 and waste_features:
            rng2 = np.random.default_rng(int(random_seed) + 50021)
            lr3 = float(cycle_learning_rate) * 0.25
            for sc_cycle in range(1, int(scavenger_cycles) + 1):
                cycle_update = np.zeros_like(logits)
                for j, cls in enumerate(classes):
                    y01 = (y_cat == str(cls)).astype("float64").to_numpy(dtype="float64")
                    p = _sigmoid(logits[:, j])
                    residual = y01 - p
                    residual_train = residual[train_mask]

                    class_score = np.zeros(df.shape[0], dtype="float64")
                    denom = 0.0
                    for col in waste_features:
                        fk = feature_info[col].feature_kind
                        if fk in ("numeric", "datetime", "bool"):
                            x_raw = x_encoded_by_feature[col]
                        else:
                            x_cat = df[col].astype("string").fillna("__MISSING__")
                            tmp = pd.DataFrame({"x": x_cat[train_mask], "r": residual_train})
                            rates = tmp.groupby("x")["r"].mean()
                            x_raw = x_cat.map(rates).fillna(0.0).to_numpy(dtype="float64")

                        z = _zscore_with_train_stats(x_raw, train_mask)
                        charge = _pearson_corr(z[train_mask], residual_train)
                        if not math.isfinite(charge) or abs(charge) < 1e-8:
                            continue

                        medium = migration_by_feature[col]
                        bond_factor = bond_factors.get(col, 1.0)
                        certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                        density = (1.0 + max(0.0, medium.kl_divergence)) * certainty

                        eta_base = max(1e-6, medium.entropy + medium.variance)
                        eta_dynamic = max(1e-6, eta_base / (1.0 + shear * abs(charge) * bond_factor))
                        if thermal_noise and sc_cycle <= int(thermal_noise_cycles):
                            eta_dynamic = max(
                                1e-6,
                                eta_dynamic
                                * (1.0 + rng2.uniform(-float(thermal_noise_level), float(thermal_noise_level))),
                            )

                        eff_charge = float(charge)
                        if eff_charge < 0:
                            eff_charge *= neg_mult
                        q = eff_charge * density * bond_factor
                        x_pos = float(pI_map.get(col, 0.5))
                        field = float(plane_mobility) - grad * x_pos
                        v = (q * field) / eta_dynamic

                        denom += abs(v)
                        class_score += v * z

                    if denom <= 1e-12:
                        denom = 1.0
                    cycle_update[:, j] = class_score / denom

                logits += lr3 * cycle_update
                probs = _softmax(logits)
                pred_idx_cycle = np.argmax(probs, axis=1)
                pred_cycle = np.array([classes[int(i)] for i in pred_idx_cycle], dtype="object")
                test_acc = float(
                    np.mean((y_cat[test_mask].to_numpy(dtype="object") == pred_cycle[test_mask]).astype("float64"))
                )
                iteration_gains.append(
                    IterationInfo(
                        cycle=n_cycles_eff + n2 + sc_cycle,
                        test_accuracy=test_acc,
                        lift_over_baseline=test_acc - baseline_accuracy,
                    )
                )

        # Persist shattered zones for output (added after Stage 1 zones at the end).
        _cascade_shattered_zones = shattered_zones
    else:
        _cascade_shattered_zones = []

    probs = _softmax(logits)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [classes[int(i)] for i in pred_idx]

    y_all = y_cat.to_numpy(dtype="object")
    pred_all = np.array(pred_labels, dtype="object")
    accuracy = float(np.mean((y_all[test_mask] == pred_all[test_mask]).astype("float64")))
    best_iter = max(iteration_gains, key=lambda it: it.test_accuracy) if iteration_gains else None

    preview = []
    test_indices = np.flatnonzero(test_mask)
    for idx in test_indices[: min(max_preview_rows, len(test_indices))]:
        i = int(idx)
        preview.append({"row": i, "actual": str(y_cat.iloc[i]), "predicted": str(pred_labels[i])})

    metrics = PredictionMetrics(
        target_kind=target_kind,
        n_rows=int(df.shape[0]),
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        train_fraction=float(train_fraction),
        random_seed=int(random_seed),
        n_features_used=len(weights_used),
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        best_cycle=None if best_iter is None else int(best_iter.cycle),
        best_lift=None if best_iter is None else float(best_iter.lift_over_baseline),
    )

    # Output equilibrium zones = Stage 1 zones + (optional) Stage 2 shattered zones.
    equilibrium_zones: list[EquilibriumZone] = []
    for zone_id in range(n_zones):
        features_in_zone = zone_bins_stage1.get(zone_id, [])
        if features_in_zone:
            avg_pI = float(np.mean([pI_map.get(f, 0.5) for f in features_in_zone]))
            avg_momentum = float(np.mean([0.85 ** int(best_iter.cycle) for _ in features_in_zone])) if best_iter else 0.0
            strength = float(len(features_in_zone) / len(feature_cols_used)) if feature_cols_used else 0.0
            equilibrium_zones.append(
                EquilibriumZone(
                    zone_id=zone_id,
                    features=features_in_zone,
                    avg_pI=avg_pI,
                    avg_momentum=avg_momentum,
                    strength=strength,
                )
            )

    # Attach shattered sub-zones if cascade ran.
    if _cascade_shattered_zones:
        equilibrium_zones.extend(_cascade_shattered_zones)

    return PredictionResult(
        target=target_col,
        target_kind=target_kind,
        plane=plane,
        weights=weights_used,
        migration_map=sorted(migration_map, key=lambda m: m.arrival_speed, reverse=True),
        bonding_map=bonds,
        equilibrium_zones=equilibrium_zones,
        iteration_gains=iteration_gains,
        metrics=metrics,
        preview_rows=preview,
    )
