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


@dataclass(frozen=True)
class PredictionResult:
    target: str
    target_kind: TargetKind
    plane: PhysicsPlane
    weights: list[WeightInfo]
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

    # Compute association weights on TRAIN only (for explanation + feature selection)
    weights: list[WeightInfo] = []
    for col in feature_cols:
        feat = df[col]
        fk = infer_feature_kind(feat)
        w, method, signed = _compute_association(feat[train_mask], target_series[train_mask], fk, target_kind)
        if not math.isfinite(w):
            w = 0.0
        weights.append(WeightInfo(feature=col, weight=float(w), method=method, feature_kind=fk, signed=signed))

    weights_sorted = sorted(weights, key=lambda wi: abs(wi.weight), reverse=True)
    weights_used = [w for w in weights_sorted if abs(w.weight) > 1e-8]

    # If everything is ~0, keep a few anyway so the UI can show something.
    if not weights_used:
        weights_used = weights_sorted[: min(10, len(weights_sorted))]

    # Keep only top-k for prediction.
    weights_used = weights_used[: max(1, min(top_k_weights, len(weights_used)))]

    if target_kind in ("numeric", "datetime"):
        y = _to_float_array(target_series, kind=target_kind)
        y_train = y[train_mask]
        y_train_mask = np.isfinite(y_train)
        y_mean = float(np.nanmean(y_train)) if y_train_mask.any() else 0.0
        y_std = float(np.nanstd(y_train)) if y_train_mask.any() else 0.0
        if y_std <= 1e-12:
            y_std = 1.0

        neg_mult = _plane_negative_multiplier(plane)
        denom = float(sum(abs(w.weight) for w in weights_used))
        if denom <= 1e-12:
            denom = 1.0

        score = np.zeros(df.shape[0], dtype="float64")
        for wi in weights_used:
            feat = df[wi.feature]
            x_raw = _encode_feature_numeric(
                feat,
                wi.feature_kind,
                y,
                target_is_finite_mask=np.isfinite(y) & train_mask,
            )
            z = _zscore_with_train_stats(x_raw, train_mask)

            w = float(wi.weight)
            if w < 0:
                w *= neg_mult
            score += w * z

        pred = y_mean + (score / denom) * y_std

        mae, rmse = _numeric_metrics(y[test_mask], pred[test_mask])

        if target_kind == "datetime":
            # Interpret error in seconds (since values are nanoseconds).
            mae = mae / 1e9
            rmse = rmse / 1e9

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
            mae=mae,
            rmse=rmse,
        )
        return PredictionResult(
            target=target_col,
            target_kind=target_kind,
            plane=plane,
            weights=weights_used,
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

    # Precompute feature numeric encodings for binary targets.
    x_encoded_by_feature: dict[str, np.ndarray] = {}
    for col in feature_cols_used:
        fk = feature_info[col].feature_kind
        if fk in ("numeric", "datetime", "bool"):
            x_encoded_by_feature[col] = _to_float_array(df[col], kind=fk)
        else:
            # placeholder; will encode per-class
            x_encoded_by_feature[col] = np.zeros(df.shape[0], dtype="float64")

    scores = np.zeros((df.shape[0], len(classes)), dtype="float64")

    for j, cls in enumerate(classes):
        y01 = (y_cat == str(cls)).astype("float64").to_numpy(dtype="float64")
        y01_train = y01[train_mask]
        prior = float(np.clip(y01_train.mean(), 1e-9, 1 - 1e-9))
        class_score = np.zeros(df.shape[0], dtype="float64")
        denom = 0.0

        for col in feature_cols_used:
            fk = feature_info[col].feature_kind

            if fk in ("numeric", "datetime", "bool"):
                x = x_encoded_by_feature[col]
                z = _zscore_with_train_stats(x, train_mask)
                w = _pearson_corr(z[train_mask], y01_train)
            else:
                # categorical: encode by P(y=1|cat) for this class
                x_cat = df[col].astype("string").fillna("__MISSING__")
                tmp = pd.DataFrame({"x": x_cat[train_mask], "y": y01_train})
                rates = tmp.groupby("x")["y"].mean()
                encoded = x_cat.map(rates).fillna(prior).to_numpy(dtype="float64")
                z = _zscore_with_train_stats(encoded, train_mask)
                w = _pearson_corr(z[train_mask], y01_train)

            if not math.isfinite(w) or abs(w) < 1e-8:
                continue

            denom += abs(w)
            w_eff = w
            if w_eff < 0:
                w_eff *= _plane_negative_multiplier(plane)
            class_score += w_eff * z

        if denom <= 1e-12:
            denom = 1.0
        # Log prior nudges the decision toward common classes.
        scores[:, j] = (class_score / denom) + math.log(prior)

    pred_idx = np.argmax(scores, axis=1)
    pred_labels = [classes[int(i)] for i in pred_idx]

    y_all = y_cat.to_numpy(dtype="object")
    pred_all = np.array(pred_labels, dtype="object")
    accuracy = float(np.mean((y_all[test_mask] == pred_all[test_mask]).astype("float64")))

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
    )

    return PredictionResult(
        target=target_col,
        target_kind=target_kind,
        plane=plane,
        weights=weights_used,
        metrics=metrics,
        preview_rows=preview,
    )
