from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Literal

import math

import numpy as np
import pandas as pd

try:
    from scipy import stats as _sp_stats  # type: ignore
except Exception:  # pragma: no cover
    _sp_stats = None


TargetKind = Literal["numeric", "categorical", "datetime"]
FeatureKind = Literal["numeric", "categorical", "datetime", "bool"]


class PhysicsPlane(str, Enum):
    solid = "solid"
    liquid = "liquid"
    gas = "gas"


@dataclass(frozen=True, slots=True)
class WeightInfo:
    feature: str
    weight: float
    method: str
    feature_kind: FeatureKind
    signed: bool


@dataclass(frozen=True, slots=True)
class MigrationInfo:
    feature: str
    feature_kind: FeatureKind
    method: str
    charge: float
    ionization: Literal["parametric", "nonparametric"]
    normality_p: float | None
    p_value: float | None
    mass: float
    stable: bool
    complex_id: int | None
    complex_size: int | None
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


@dataclass(frozen=True, slots=True)
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
    buffer_ionization: Literal["parametric", "nonparametric"] | None = None
    buffer_normality_p: float | None = None
    gel_band_sharpness: float | None = None
    gel_smearing: float | None = None
    gel_ghost_band_rate: float | None = None
    gel_confidence_mean: float | None = None
    gel_confidence_std: float | None = None
    # Selective prediction / abstention metrics (categorical targets only when enabled).
    abstain_rate: float | None = None
    coverage: float | None = None
    selective_accuracy: float | None = None


@dataclass(frozen=True, slots=True)
class BondInfo:
    feature_a: str
    feature_b: str
    affinity: float
    bonding_factor: float
    bond_type: str = "affinity"


@dataclass(frozen=True, slots=True)
class IterationInfo:
    cycle: int
    test_accuracy: float | None = None
    lift_over_baseline: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None


@dataclass(frozen=True, slots=True)
class EquilibriumZone:
    zone_id: int
    features: list[str]
    avg_pI: float
    avg_momentum: float
    strength: float


@dataclass(frozen=True, slots=True)
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
    diagnostics: dict[str, Any] | None = None
    test_row_indices: list[int] | None = None
    test_actual: list[Any] | None = None
    test_predicted: list[Any] | None = None


@dataclass(slots=True)
class PredictorRuntimeState:
    version: int = 1
    cycle_index: int = 0
    adaptive_gain: float = 1.0
    homeostasis_score: float = 0.5
    preferred_plane: PhysicsPlane | None = None
    last_metrics: dict[str, Any] = field(default_factory=dict)
    feature_last_seen: dict[str, int] = field(default_factory=dict)
    complex_last_seen: dict[str, int] = field(default_factory=dict)
    dream_buffer: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PredictorError(ValueError):
    pass


def _homeostasis_score_from_metrics(metrics: PredictionMetrics) -> float:
    """Derive a compact stability score from recent metrics."""

    smearing = float(metrics.gel_smearing) if metrics.gel_smearing is not None and math.isfinite(float(metrics.gel_smearing)) else 0.5
    confidence_std = float(metrics.gel_confidence_std) if metrics.gel_confidence_std is not None and math.isfinite(float(metrics.gel_confidence_std)) else 0.0
    selective_accuracy = (
        float(metrics.selective_accuracy)
        if metrics.selective_accuracy is not None and math.isfinite(float(metrics.selective_accuracy))
        else None
    )
    abstain_rate = (
        float(metrics.abstain_rate)
        if metrics.abstain_rate is not None and math.isfinite(float(metrics.abstain_rate))
        else None
    )

    stability = 1.0
    stability -= 0.45 * float(np.clip(smearing, 0.0, 1.0))
    stability -= 0.20 * float(np.clip(confidence_std, 0.0, 1.0))
    if selective_accuracy is not None:
        stability += 0.35 * float(np.clip(selective_accuracy, 0.0, 1.0))
    if abstain_rate is not None:
        stability -= 0.20 * float(np.clip(abstain_rate, 0.0, 1.0))
    return float(np.clip(stability, 0.0, 1.0))


def _adaptive_gain_from_state(base_gain: float, state: PredictorRuntimeState | None) -> float:
    """Apply a simple homeostatic adjustment to the PCR gain."""

    gain = float(base_gain)
    if not math.isfinite(gain) or gain <= 0.0:
        gain = 0.55
    if state is None or not state.last_metrics:
        return float(gain)

    try:
        metrics = PredictionMetrics(**state.last_metrics)
    except Exception:
        return float(gain)

    stability = _homeostasis_score_from_metrics(metrics)
    # Stable systems should amplify less; unsettled systems can explore a bit more.
    multiplier = 1.22 - 0.95 * stability
    multiplier = float(np.clip(multiplier, 0.35, 1.25))
    return float(gain * multiplier)


def serialize_predictor_state(state: PredictorRuntimeState) -> dict[str, Any]:
    """Convert runtime state into a JSON-safe dictionary."""

    payload = asdict(state)
    if payload.get("preferred_plane") is not None:
        payload["preferred_plane"] = str(payload["preferred_plane"].value)
    return payload


def deserialize_predictor_state(payload: dict[str, Any]) -> PredictorRuntimeState:
    """Load runtime state from a dictionary."""

    data = dict(payload or {})
    preferred_plane = data.get("preferred_plane")
    if preferred_plane is not None:
        try:
            data["preferred_plane"] = PhysicsPlane(str(preferred_plane))
        except Exception:
            data["preferred_plane"] = None
    return PredictorRuntimeState(
        version=int(data.get("version", 1)),
        cycle_index=int(data.get("cycle_index", 0)),
        adaptive_gain=float(data.get("adaptive_gain", 1.0)),
        homeostasis_score=float(data.get("homeostasis_score", 0.5)),
        preferred_plane=data.get("preferred_plane"),
        last_metrics=dict(data.get("last_metrics", {}) or {}),
        feature_last_seen={str(k): int(v) for k, v in dict(data.get("feature_last_seen", {}) or {}).items()},
        complex_last_seen={str(k): int(v) for k, v in dict(data.get("complex_last_seen", {}) or {}).items()},
        dream_buffer=list(data.get("dream_buffer", []) or []),
        metadata=dict(data.get("metadata", {}) or {}),
    )


def save_predictor_state(state: PredictorRuntimeState, file_path: str | Path) -> None:
    """Persist runtime state to disk as JSON."""

    path = Path(file_path)
    path.write_text(json.dumps(serialize_predictor_state(state), indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_predictor_state(file_path: str | Path) -> PredictorRuntimeState:
    """Load runtime state from a JSON file."""

    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PredictorError("Predictor state file must contain a JSON object")
    return deserialize_predictor_state(payload)


def prune_predictor_state(
    state: PredictorRuntimeState,
    *,
    max_age_cycles: int = 1000,
    min_terminal_velocity: float = 1e-4,
) -> dict[str, int]:
    """Remove stale features/complexes from runtime memory."""

    max_age = int(max(0, max_age_cycles))
    pruned_features = 0
    pruned_complexes = 0
    current_cycle = int(state.cycle_index)

    feature_keep: dict[str, int] = {}
    for name, last_seen in state.feature_last_seen.items():
        age = current_cycle - int(last_seen)
        if age <= max_age:
            feature_keep[name] = int(last_seen)
        else:
            pruned_features += 1
    state.feature_last_seen = feature_keep

    complex_keep: dict[str, int] = {}
    for key, last_seen in state.complex_last_seen.items():
        age = current_cycle - int(last_seen)
        if age <= max_age:
            complex_keep[key] = int(last_seen)
        else:
            pruned_complexes += 1
    state.complex_last_seen = complex_keep

    state.metadata["ghost_prune_max_age_cycles"] = max_age
    state.metadata["ghost_prune_min_terminal_velocity"] = float(min_terminal_velocity)
    state.metadata["ghost_pruned_features"] = int(pruned_features)
    state.metadata["ghost_pruned_complexes"] = int(pruned_complexes)
    return {"features": int(pruned_features), "complexes": int(pruned_complexes)}


def update_predictor_state_from_result(
    state: PredictorRuntimeState,
    result: PredictionResult,
    *,
    low_confidence_rows: list[dict[str, Any]] | None = None,
    low_confidence_limit: int = 32,
) -> PredictorRuntimeState:
    """Update homeostasis, ghost tracking, and dreaming buffer from a finished run."""

    state.cycle_index += 1
    state.last_metrics = serialize_metrics(result.metrics)
    state.homeostasis_score = _homeostasis_score_from_metrics(result.metrics)
    state.adaptive_gain = float(np.clip(1.22 - 0.95 * state.homeostasis_score, 0.35, 1.25))
    if state.homeostasis_score >= 0.65:
        state.preferred_plane = PhysicsPlane.solid
    elif state.homeostasis_score >= 0.40:
        state.preferred_plane = PhysicsPlane.liquid
    else:
        state.preferred_plane = PhysicsPlane.gas

    for weight in result.weights:
        state.feature_last_seen[str(weight.feature)] = int(state.cycle_index)

    complex_velocities: dict[str, list[float]] = {}
    for migration in result.migration_map:
        if migration.complex_id is None:
            continue
        key = f"complex:{int(migration.complex_id)}"
        complex_velocities.setdefault(key, []).append(abs(float(migration.terminal_velocity)))

    for key, velocities in complex_velocities.items():
        if velocities and float(np.mean(velocities)) >= float(state.metadata.get("ghost_min_terminal_velocity", 1e-4)):
            state.complex_last_seen[key] = int(state.cycle_index)

    prune_predictor_state(
        state,
        max_age_cycles=int(state.metadata.get("ghost_prune_max_age_cycles", 1000)),
        min_terminal_velocity=float(state.metadata.get("ghost_prune_min_terminal_velocity", 1e-4)),
    )

    if low_confidence_rows:
        buffered = list(low_confidence_rows[: max(0, int(low_confidence_limit))])
        if buffered:
            state.dream_buffer.extend(buffered)
            state.dream_buffer = state.dream_buffer[-max(1, int(low_confidence_limit)) :]

    state.metadata["last_target"] = result.target
    state.metadata["last_plane"] = result.plane.value
    state.metadata["last_run_rows"] = int(result.metrics.n_rows)
    return state


def serialize_metrics(metrics: PredictionMetrics) -> dict[str, Any]:
    """Serialize prediction metrics to a JSON-safe dict."""

    return asdict(metrics)


def _missing_like_mask(series: pd.Series) -> np.ndarray:
    """Return a boolean mask of values that should be treated as missing.

    Applies a pragmatic definition that catches NaN/NA and common string sentinels.
    """

    s = series
    base = s.isna().to_numpy(dtype=bool)
    try:
        s_str = s.astype("string")
        lowered = s_str.str.strip().str.lower()
        sent = lowered.isin(["", "nan", "none", "null", "nat", "<na>"])
        base = base | sent.fillna(False).to_numpy(dtype=bool)
    except Exception:
        # If string casting fails, keep NaN-only behavior.
        pass
    return base


def _infer_cadence_hz_from_series(series: pd.Series) -> float | None:
    """Infer sample cadence in Hz from a datetime-like series."""

    try:
        dt = pd.to_datetime(series, errors="coerce", utc=False)
    except Exception:
        return None

    dt_series = pd.Series(dt).dropna()
    if dt_series.shape[0] < 3:
        return None

    deltas = dt_series.diff().dt.total_seconds()
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if deltas.empty:
        return None

    median_delta = float(np.median(deltas.to_numpy(dtype="float64")))
    if not math.isfinite(median_delta) or median_delta <= 0.0:
        return None
    return float(1.0 / median_delta)


def _infer_cadence_hz_from_dataframe(df: pd.DataFrame, *, target_col: str) -> tuple[float | None, str | None]:
    """Infer cadence from the first datetime-like column or DatetimeIndex."""

    if isinstance(df.index, pd.DatetimeIndex):
        cadence = _infer_cadence_hz_from_series(pd.Series(df.index))
        if cadence is not None:
            return cadence, "index"

    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or _is_datetime_like(s):
            cadence = _infer_cadence_hz_from_series(s)
            if cadence is not None:
                return cadence, str(col)

    return None, None


def _rolling_window_sedimentation(
    series: pd.Series,
    *,
    window: int,
    fold: float = 3.0,
    cluster_min_size: int = 3,
) -> tuple[pd.Series, dict[str, Any]]:
    """Clip isolated spikes while preserving clustered transients."""

    s = pd.to_numeric(series, errors="coerce").astype("float64")
    n = int(s.shape[0])
    w = int(window)
    if n <= 0 or w < 3:
        return s, {"window": int(max(0, w)), "fold": float(fold), "cluster_min_size": int(cluster_min_size), "clipped": 0, "preserved_clusters": 0}

    if w > n:
        w = n if n % 2 == 1 else max(3, n - 1)
    if w % 2 == 0:
        w = max(3, w - 1)

    fold0 = float(fold)
    if not math.isfinite(fold0) or fold0 <= 0.0:
        fold0 = 3.0

    cluster_min = int(cluster_min_size)
    if cluster_min < 1:
        cluster_min = 1

    min_periods = max(3, w // 3)
    median = s.rolling(window=w, center=False, min_periods=min_periods).median()
    median = median.bfill().ffill()
    if median.isna().all():
        finite_vals = s.to_numpy(dtype="float64")
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        median_fallback = float(np.median(finite_vals)) if finite_vals.size else 0.0
        if not math.isfinite(median_fallback):
            median_fallback = 0.0
        median = pd.Series(np.full(n, median_fallback), index=s.index)
    abs_dev = (s - median).abs()
    mad = abs_dev.rolling(window=w, center=False, min_periods=min_periods).median()
    mad = mad.bfill().ffill()
    if mad.isna().all():
        finite_abs_dev = abs_dev.to_numpy(dtype="float64")
        finite_abs_dev = finite_abs_dev[np.isfinite(finite_abs_dev)]
        mad_fallback = float(np.median(finite_abs_dev)) if finite_abs_dev.size else 0.0
        if not math.isfinite(mad_fallback) or mad_fallback < 0.0:
            mad_fallback = 0.0
        mad = pd.Series(np.full(n, mad_fallback), index=s.index)
    scale = 1.4826 * mad
    threshold = fold0 * scale

    deviation = abs_dev > threshold
    if cluster_min > 1:
        cluster_counts = deviation.astype("int64").rolling(window=w, center=False, min_periods=1).sum().fillna(0.0)
        preserve_mask = deviation & (cluster_counts >= float(cluster_min))
    else:
        preserve_mask = deviation.copy()

    clip_mask = deviation & ~preserve_mask
    lower = (median - threshold).to_numpy(dtype="float64")
    upper = (median + threshold).to_numpy(dtype="float64")
    values = s.to_numpy(dtype="float64")
    clipped = values.copy()
    if int(clip_mask.sum()) > 0:
        clipped[clip_mask.to_numpy(dtype=bool)] = np.clip(
            clipped[clip_mask.to_numpy(dtype=bool)],
            lower[clip_mask.to_numpy(dtype=bool)],
            upper[clip_mask.to_numpy(dtype=bool)],
        )

    diagnostics = {
        "window": int(w),
        "fold": float(fold0),
        "cluster_min_size": int(cluster_min),
        "clipped": int(clip_mask.sum()),
        "preserved_clusters": int(preserve_mask.sum()),
        "warmup_strategy": "backfill_first_valid",
    }
    return pd.Series(clipped, index=s.index), diagnostics


def _clean_dataframe_for_prediction(
    df: pd.DataFrame,
    *,
    target_col: str,
    train_mask: np.ndarray | None = None,
    drop_duplicates: bool = True,
    drop_missing_target: bool = True,
    impute_missing: bool = True,
    clip_numeric_outliers: bool = True,
    outlier_strategy: Literal["winsorize", "iqr", "gaussian", "mad", "arbitrary", "feature_engine", "rolling", "none"] = "winsorize",
    outlier_fold: float = 1.5,
    outlier_q_low: float = 0.005,
    outlier_q_high: float = 0.995,
    arbitrary_min: float | None = None,
    arbitrary_max: float | None = None,
    rolling_window: int | None = None,
    rolling_window_cadence_hz: float | None = None,
    rolling_window_seconds: float = 60.0,
    rolling_mad_fold: float = 3.0,
    rolling_cluster_min_size: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diag: dict[str, Any] = {
        "drop_duplicates": bool(drop_duplicates),
        "drop_missing_target": bool(drop_missing_target),
        "impute_missing": bool(impute_missing),
        "clip_numeric_outliers": bool(clip_numeric_outliers),
        "outlier_strategy": str(outlier_strategy),
        "outlier_fold": float(outlier_fold),
        "outlier_q_low": float(outlier_q_low),
        "outlier_q_high": float(outlier_q_high),
        "arbitrary_min": arbitrary_min,
        "arbitrary_max": arbitrary_max,
        "rolling_window": rolling_window,
        "rolling_window_cadence_hz": rolling_window_cadence_hz,
        "rolling_window_seconds": float(rolling_window_seconds),
        "rolling_mad_fold": float(rolling_mad_fold),
        "rolling_cluster_min_size": int(rolling_cluster_min_size),
    }

    n0 = int(df.shape[0])
    diag["n_rows_in"] = n0

    out = df
    if drop_duplicates:
        n_before = int(out.shape[0])
        out = out.drop_duplicates()
        diag["dropped_duplicates"] = int(n_before - int(out.shape[0]))

    if drop_missing_target:
        if target_col in out.columns:
            n_before = int(out.shape[0])
            missing_tgt = _missing_like_mask(out[target_col])
            if missing_tgt.any():
                out = out.loc[~missing_tgt]
            diag["dropped_missing_target"] = int(n_before - int(out.shape[0]))
        else:
            diag["dropped_missing_target"] = 0

    # If we changed row count, ensure dense row indices for downstream preview/test indexing.
    if int(out.shape[0]) != n0:
        out = out.reset_index(drop=True)

    # Feature-level imputation/outlier clipping using TRAIN stats only.
    n_imputed_total = 0
    n_clipped_total = 0
    clipped_cols: list[str] = []

    strategy = str(outlier_strategy or "winsorize").strip().lower()
    if strategy not in ("winsorize", "iqr", "gaussian", "mad", "arbitrary", "feature_engine", "rolling", "none"):
        strategy = "winsorize"

    rolling_window_size: int | None = None
    cadence_inferred_hz: float | None = None
    cadence_inferred_source: str | None = None
    if rolling_window is not None:
        try:
            rolling_window_size = int(rolling_window)
        except Exception:
            rolling_window_size = None
    else:
        cadence_candidate: float | None = None
        if rolling_window_cadence_hz is not None and math.isfinite(float(rolling_window_cadence_hz)):
            cadence_candidate = float(rolling_window_cadence_hz)
            cadence_inferred_source = "argument"
        elif strategy == "rolling":
            cadence_candidate, cadence_inferred_source = _infer_cadence_hz_from_dataframe(out, target_col=target_col)
        if cadence_candidate is not None and cadence_candidate > 0.0:
            cadence_inferred_hz = float(cadence_candidate)
            if cadence_inferred_hz > 10.0:
                seconds = 0.33
            else:
                seconds = float(rolling_window_seconds)
            if not math.isfinite(seconds) or seconds <= 0.0:
                seconds = 60.0
            rolling_window_size = int(round(cadence_inferred_hz * seconds))
    if rolling_window_size is not None and rolling_window_size < 3:
        rolling_window_size = 3

    if train_mask is None:
        train_mask = np.ones(int(out.shape[0]), dtype=bool)

    use_feature_engine_backend = bool(clip_numeric_outliers) and strategy == "feature_engine"
    use_rolling_window_backend = bool(clip_numeric_outliers) and (
        strategy == "rolling" or rolling_window_size is not None
    )
    if strategy == "rolling" and rolling_window_size is None:
        fallback_window = max(3, min(int(out.shape[0]) if int(out.shape[0]) > 0 else 3, 61))
        if fallback_window % 2 == 0:
            fallback_window = max(3, fallback_window - 1)
        rolling_window_size = fallback_window

    diag["cadence_inferred_hz"] = cadence_inferred_hz
    diag["cadence_inferred_source"] = cadence_inferred_source
    diag["rolling_window_mode"] = (
        "manual" if rolling_window is not None else ("inferred" if cadence_inferred_hz is not None else ("fallback" if strategy == "rolling" else "disabled"))
    )
    diag["rolling_window_effective"] = int(rolling_window_size) if rolling_window_size is not None else None

    effective_train_mask = train_mask

    feature_cols = [c for c in out.columns if c != target_col]
    for col in feature_cols:
        s = out[col]

        # Bool columns: impute missing with train mode, keep as bool-ish.
        if pd.api.types.is_bool_dtype(s):
            if not impute_missing:
                continue
            s0 = s.astype("object")
            miss = s0.isna().to_numpy(dtype=bool)
            if int(miss.sum()) == 0:
                continue
            train_vals = s0.to_numpy(copy=False)[effective_train_mask]
            train_vals = train_vals[pd.notna(train_vals)]
            fill = bool(train_vals[0]) if train_vals.size else False
            n_imputed_total += int(miss.sum())
            out[col] = s0.fillna(fill).astype(bool)
            continue

        # Numeric columns: coerce to float, impute median, optionally clip outliers.
        if pd.api.types.is_numeric_dtype(s):
            x_all = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
            x_train = x_all[effective_train_mask]
            finite_train = x_train[np.isfinite(x_train)]
            fill = float(np.nanmedian(finite_train)) if finite_train.size else 0.0
            if not math.isfinite(fill):
                fill = 0.0

            missing_mask = ~np.isfinite(x_all)
            if impute_missing and int(missing_mask.sum()) > 0:
                n_imputed_total += int(missing_mask.sum())
                x_all = np.where(missing_mask, fill, x_all)

            if use_rolling_window_backend and rolling_window_size is not None:
                rolled, roll_diag = _rolling_window_sedimentation(
                    pd.Series(x_all, index=out.index),
                    window=int(rolling_window_size),
                    fold=float(rolling_mad_fold),
                    cluster_min_size=int(rolling_cluster_min_size),
                )
                x_all = rolled.to_numpy(dtype="float64")
                n_clipped_total += int(roll_diag.get("clipped", 0))
                if int(roll_diag.get("clipped", 0)) > 0:
                    clipped_cols.append(str(col))
                diag.setdefault("rolling_window_details", {})[str(col)] = roll_diag

            if clip_numeric_outliers and strategy not in ("none", "rolling") and not use_feature_engine_backend and not use_rolling_window_backend:
                # Need a minimum number of finite train points to define robust caps.
                if finite_train.size >= 16:
                    lo: float | None = None
                    hi: float | None = None

                    if strategy == "winsorize":
                        ql = float(max(0.0, min(0.499, float(outlier_q_low))))
                        qh = float(max(0.501, min(1.0, float(outlier_q_high))))
                        lo = float(np.quantile(finite_train, ql))
                        hi = float(np.quantile(finite_train, qh))
                    elif strategy == "iqr":
                        q1 = float(np.quantile(finite_train, 0.25))
                        q3 = float(np.quantile(finite_train, 0.75))
                        iqr = float(q3 - q1)
                        f = float(outlier_fold)
                        if not math.isfinite(f) or f <= 0:
                            f = 1.5
                        lo = q1 - f * iqr
                        hi = q3 + f * iqr
                    elif strategy == "gaussian":
                        mu = float(np.mean(finite_train))
                        sd = float(np.std(finite_train))
                        f = float(outlier_fold)
                        if not math.isfinite(f) or f <= 0:
                            f = 3.0
                        lo = mu - f * sd
                        hi = mu + f * sd
                    elif strategy == "mad":
                        med = float(np.median(finite_train))
                        mad = float(np.median(np.abs(finite_train - med)))
                        # Consistent MAD estimate of sigma.
                        mad_sigma = 1.4826 * mad
                        f = float(outlier_fold)
                        if not math.isfinite(f) or f <= 0:
                            f = 3.5
                        lo = med - f * mad_sigma
                        hi = med + f * mad_sigma
                    elif strategy == "arbitrary":
                        lo = None if arbitrary_min is None else float(arbitrary_min)
                        hi = None if arbitrary_max is None else float(arbitrary_max)

                    if lo is not None and not math.isfinite(float(lo)):
                        lo = None
                    if hi is not None and not math.isfinite(float(hi)):
                        hi = None
                    if lo is None and hi is None:
                        lo = None
                        hi = None

                    if lo is not None and hi is not None and float(hi) < float(lo):
                        lo, hi = hi, lo

                    if lo is not None or hi is not None:
                        # np.clip requires both; emulate with where if one side missing.
                        before = x_all
                        if lo is not None and hi is not None:
                            x_all = np.clip(x_all, float(lo), float(hi))
                        elif lo is not None:
                            x_all = np.where(x_all < float(lo), float(lo), x_all)
                        elif hi is not None:
                            x_all = np.where(x_all > float(hi), float(hi), x_all)
                        clipped_mask = np.isfinite(before) & np.isfinite(x_all) & (before != x_all)
                        if int(clipped_mask.sum()) > 0:
                            n_clipped_total += int(clipped_mask.sum())
                            clipped_cols.append(str(col))

            out[col] = x_all
            continue

        # Datetime columns: impute missing with train median timestamp.
        if pd.api.types.is_datetime64_any_dtype(s):
            if impute_missing:
                x_all = s.astype("datetime64[ns]").astype("int64").to_numpy(dtype="int64", copy=True)
                # pandas uses NaT == min int
                nat = np.iinfo(np.int64).min
                x_train = x_all[effective_train_mask]
                finite_train = x_train[x_train != nat]
                fill = int(np.median(finite_train)) if finite_train.size else 0
                missing = x_all == nat
                if int(missing.sum()) > 0:
                    n_imputed_total += int(missing.sum())
                    x_all[missing] = fill
                    out[col] = pd.to_datetime(x_all)
            continue

        # Everything else: treat as categorical-like; impute missing tokens.
        if impute_missing:
            miss = _missing_like_mask(s)
            if int(miss.sum()) > 0:
                n_imputed_total += int(miss.sum())
                s_str = s.astype("string").fillna("__MISSING__")
                try:
                    s_str = s_str.mask(miss, "__MISSING__")
                except Exception:
                    pass
                out[col] = s_str
            else:
                # Normalize to string for consistent downstream handling.
                try:
                    out[col] = s.astype("string")
                except Exception:
                    pass

    diag["n_rows_out"] = int(out.shape[0])
    diag["imputed_values"] = int(n_imputed_total)
    diag["clipped_outliers"] = int(n_clipped_total)
    diag["clipped_columns"] = sorted(set(clipped_cols))[:50]

    if train_mask is None:
        diag["n_rows_out"] = int(out.shape[0])
        return out, diag

    if use_feature_engine_backend:
        try:
            from feature_engine.outliers import Winsorizer  # type: ignore

            numeric_cols = [
                c
                for c in feature_cols
                if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c])
            ]
            if numeric_cols:
                fe_train = out.loc[train_mask, numeric_cols].copy()
                fe_all = out[numeric_cols].copy()
                wins = Winsorizer(
                    capping_method="iqr",
                    tail="both",
                    fold=float(outlier_fold),
                    variables=numeric_cols,
                )
                wins.fit(fe_train)
                fe_all2 = wins.transform(fe_all)
                for c in numeric_cols:
                    before = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype="float64")
                    after = pd.to_numeric(fe_all2[c], errors="coerce").to_numpy(dtype="float64")
                    changed = np.isfinite(before) & np.isfinite(after) & (before != after)
                    if int(changed.sum()) > 0:
                        n_clipped_total += int(changed.sum())
                        clipped_cols.append(str(c))
                    out[c] = after
                diag["feature_engine"] = {"ok": True, "method": "winsorizer_iqr"}
                diag["clipped_outliers"] = int(n_clipped_total)
                diag["clipped_columns"] = sorted(set(clipped_cols))[:50]
            else:
                diag["feature_engine"] = {"ok": True, "method": "winsorizer_iqr", "note": "no numeric columns"}
        except Exception as e:
            diag["feature_engine"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return out, diag


def clean_tabular_dataframe(
    df: pd.DataFrame,
    *,
    target_col: str,
    train_mask: np.ndarray | None = None,
    drop_duplicates: bool = True,
    drop_missing_target: bool = True,
    impute_missing: bool = True,
    clip_numeric_outliers: bool = True,
    outlier_strategy: Literal["winsorize", "iqr", "gaussian", "mad", "arbitrary", "feature_engine", "none"] = "winsorize",
    outlier_fold: float = 1.5,
    outlier_q_low: float = 0.005,
    outlier_q_high: float = 0.995,
    arbitrary_min: float | None = None,
    arbitrary_max: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Public wrapper for the predictor cleaning pass."""

    return _clean_dataframe_for_prediction(
        df,
        target_col=target_col,
        train_mask=train_mask,
        drop_duplicates=drop_duplicates,
        drop_missing_target=drop_missing_target,
        impute_missing=impute_missing,
        clip_numeric_outliers=clip_numeric_outliers,
        outlier_strategy=outlier_strategy,
        outlier_fold=outlier_fold,
        outlier_q_low=outlier_q_low,
        outlier_q_high=outlier_q_high,
        arbitrary_min=arbitrary_min,
        arbitrary_max=arbitrary_max,
    )


def _primer_strength_from_p(
    p_value: float | None,
    *,
    p_threshold: float,
    tau: float,
    strength_cap: float,
) -> float:
    """Convert a p-value into a bounded primer "binding strength".

    strength = 0 when p is missing or p >= threshold.
    Otherwise strength grows with -log10(p) and is scaled by `tau`.
    """

    if p_value is None:
        return 0.0
    try:
        pv = float(p_value)
    except Exception:
        return 0.0
    if not math.isfinite(pv):
        return 0.0
    if pv <= 0.0:
        pv = 1e-300
    if pv >= float(p_threshold):
        return 0.0

    t = float(tau)
    if not math.isfinite(t) or t <= 1e-12:
        t = 1.0
    smax = float(strength_cap)
    if not math.isfinite(smax) or smax <= 0.0:
        smax = 1.0

    s = (-math.log10(pv)) / t
    return float(np.clip(s, 0.0, smax))


def _pcr_amplification_factor(
    *,
    p_value: float | None,
    stable: bool,
    enabled: bool,
    cycles: int,
    p_threshold: float,
    tau: float,
    gain: float,
    strength_cap: float,
    amp_cap: float,
    require_stable: bool,
) -> tuple[float, float]:
    """Return (amp, primer_strength) for a single feature."""

    if not bool(enabled):
        return 1.0, 0.0
    if int(cycles) <= 0:
        return 1.0, 0.0
    if bool(require_stable) and not bool(stable):
        return 1.0, 0.0

    s = _primer_strength_from_p(
        p_value,
        p_threshold=float(p_threshold),
        tau=float(tau),
        strength_cap=float(strength_cap),
    )
    if s <= 0.0:
        return 1.0, 0.0

    g = float(gain)
    if not math.isfinite(g) or g <= 0.0:
        g = 0.0

    amp = 1.0 + float(cycles) * g * s
    cap = float(amp_cap)
    if not math.isfinite(cap) or cap <= 1.0:
        cap = 1.0
    amp = float(np.clip(amp, 1.0, cap))
    return amp, float(s)


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


def _plane_base_viscosity(plane: PhysicsPlane) -> float:
    # Baseline resistance of the medium (solid > liquid > gas).
    return {
        PhysicsPlane.solid: 1.25,
        PhysicsPlane.liquid: 1.0,
        PhysicsPlane.gas: 0.85,
    }[plane]


def _calculate_viscosity_field(
    *,
    plane: PhysicsPlane,
    entropy: float,
    variance: float,
    correlation_strength: float,
    mass: float,
    ionization: Literal["parametric", "nonparametric"],
    unstable: bool,
    complex_drag: float = 1.0,
) -> float:
    """Target-induced viscosity field.

    - High correlation + significant p (high mass) -> thinning (lower viscosity).
    - Low correlation or low significance -> thermal turbulence (higher viscosity).
    - Nonparametric compounds see additional viscous drag.
    - Complex anchoring can drag a whole complex into high-entropy zones.
    """

    plane_eta = _plane_base_viscosity(plane)
    entropy0 = float(np.clip(float(entropy), 0.0, 1.0))
    variance0 = float(np.clip(float(variance), 0.0, 1.0))

    # Strength should live in [0, 1] for all association methods.
    strength = float(np.clip(abs(float(correlation_strength)), 0.0, 1.0))
    mass_norm = float(np.clip(float(mass) / 6.0, 0.0, 1.0))

    # Base viscosity of the medium (plane) + local entropy/variance contribution.
    base_eta = plane_eta * max(1e-6, 0.25 + (entropy0 + variance0))

    # Thermal turbulence: low strength and/or low significance increases viscosity.
    turbulence = float(np.clip(0.70 * (1.0 - strength) + 0.30 * (1.0 - mass_norm), 0.0, 1.0))

    # Statistical thinning: high strength with high mass reduces viscosity.
    thinning = float(np.clip(strength * mass_norm, 0.0, 1.0))

    # Translate factors into viscosity space; clamp to keep physics stable.
    eta = base_eta * (1.0 + 0.60 * entropy0) * (1.0 + 0.90 * turbulence)
    eta = eta * float(max(0.25, float(complex_drag)))
    eta = eta - (base_eta * 0.85 * thinning)
    eta = max(1e-6, float(eta))

    if ionization == "nonparametric":
        eta *= 1.35
    # Unstable compounds (p > alpha) should see high inert viscosity so random noise
    # cannot migrate toward the target and pollute the final velocity field.
    if unstable:
        eta *= 2.25
    else:
        # Express lanes: high-affinity + high-mass compounds move through a thinned medium.
        # This is intentionally non-linear so mid-strength features aren't over-boosted.
        lane = float(np.clip(thinning, 0.0, 1.0))
        if lane >= 0.40:
            eta *= float(np.clip(1.0 - 0.20 * (lane**0.85), 0.70, 1.0))

    return float(max(1e-6, eta))


def _vibrational_viscosity_multiplier(
    cycle: int,
    *,
    enabled: bool,
    period: int,
    amplitude: float,
    waveform: Literal["sine", "square"] = "square",
    phase: float = 0.0,
) -> float:
    """Cycle-wise viscosity multiplier.

    Intended as a gentle, opt-in oscillation of the effective viscosity during training.
    Multiplies η (so lower values => stronger updates).
    """

    if not bool(enabled):
        return 1.0

    p = int(period)
    if p <= 0:
        p = 1
    a = float(np.clip(float(amplitude), 0.0, 0.95))
    if a <= 0.0:
        return 1.0

    w = str(waveform).lower().strip()
    c = int(max(1, cycle))
    ph = float(phase) if math.isfinite(float(phase)) else 0.0

    if w == "sine":
        theta = 2.0 * math.pi * (float((c - 1) % p) / float(p)) + ph
        mult = 1.0 + a * math.sin(theta)
    else:
        # Square wave: alternate low/high viscosity in blocks of `period` cycles.
        phase_offset = 0 if math.sin(ph) >= 0.0 else 1
        wave_phase = (((c - 1) // p) + phase_offset) % 2
        mult = (1.0 - a) if wave_phase == 0 else (1.0 + a)

    return float(np.clip(mult, 0.05, 20.0))


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
        finite = pd.to_numeric(series, errors="coerce")
        non_null = finite.dropna()
        if not non_null.empty:
            n_unique = int(non_null.nunique(dropna=True))
            if n_unique <= max(20, int(math.sqrt(max(1, len(non_null))))) and n_unique <= 25:
                return "categorical"
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


def _safe_feature_slug(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))
    slug = slug.strip("_")
    return slug or "feature"


def _isotope_feature_bundle(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_kinds: dict[str, FeatureKind],
    *,
    target_series: pd.Series,
    target_kind: TargetKind,
    train_mask: np.ndarray,
    max_numeric_features: int = 6,
    max_category_levels: int = 8,
    max_total_isotopes: int = 48,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Create bounded numeric × categorical cross-features for high-cardinality structure.

    The generated features behave like "isotopes": a numeric carrier signal that only
    exists when a categorical state is present. This keeps the feature explosion bounded
    while still capturing interactions such as device power at night.
    """

    working = df.copy()
    if not feature_cols or max_numeric_features <= 0 or max_total_isotopes <= 0:
        return working, [], {"enabled": False, "generated": 0, "pairs": []}

    numeric_like = [c for c in feature_cols if feature_kinds.get(c) in ("numeric", "datetime", "bool")]
    categorical_like = [c for c in feature_cols if feature_kinds.get(c) == "categorical"]
    if not numeric_like or not categorical_like:
        return working, [], {"enabled": True, "generated": 0, "pairs": []}

    target_rank = _to_float_array(target_series, kind=target_kind) if target_kind in ("numeric", "datetime") else None
    ranked_numeric: list[tuple[float, str]] = []
    for col in numeric_like:
        x = _to_float_array(df[col], kind=feature_kinds[col])
        if target_rank is not None:
            score = abs(_pearson_corr(x[train_mask], target_rank[train_mask]))
        else:
            finite = x[train_mask]
            finite = finite[np.isfinite(finite)]
            score = float(np.nanstd(finite)) if finite.size else 0.0
        ranked_numeric.append((float(score), str(col)))
    ranked_numeric.sort(reverse=True)
    selected_numeric = [col for _, col in ranked_numeric[: max(1, int(max_numeric_features))]]

    generated_cols: list[str] = []
    bundle_pairs: list[dict[str, Any]] = []
    generated_count = 0
    for num_col in selected_numeric:
        x = _to_float_array(df[num_col], kind=feature_kinds[num_col])
        x = np.where(np.isfinite(x), x, 0.0)
        x_center = x - float(np.nanmean(x[train_mask])) if np.any(train_mask) else x
        for cat_col in categorical_like:
            if generated_count >= int(max_total_isotopes):
                break
            cat_series = df[cat_col].astype("string").fillna("__MISSING__")
            levels = list(pd.Series(cat_series[train_mask]).value_counts().index)
            if not levels:
                continue
            levels = levels[: max(1, int(max_category_levels))]
            for level in levels:
                if generated_count >= int(max_total_isotopes):
                    break
                level_mask = (cat_series == str(level)).to_numpy(dtype="float64")
                if not np.any(level_mask > 0.0):
                    continue
                col_name = f"__iso__{_safe_feature_slug(num_col)}__x__{_safe_feature_slug(cat_col)}__{_safe_feature_slug(level)}"
                if col_name in working.columns:
                    continue
                working[col_name] = x_center * level_mask
                generated_cols.append(col_name)
                bundle_pairs.append({"numeric": str(num_col), "categorical": str(cat_col), "level": str(level), "column": col_name})
                generated_count += 1

    diagnostics = {
        "enabled": True,
        "generated": int(generated_count),
        "numeric_selected": selected_numeric,
        "categorical_selected": categorical_like[:],
        "pairs": bundle_pairs[:24],
    }
    return working, generated_cols, diagnostics


def should_abstain_from_prediction(
    metrics: PredictionMetrics,
    *,
    confidence_std_threshold: float = 0.25,
    smearing_threshold: float = 0.65,
    min_selective_accuracy: float = 0.25,
) -> bool:
    """Return True when the run looks too uncertain to trust."""

    conf_std = metrics.gel_confidence_std
    smear = metrics.gel_smearing
    selective_accuracy = metrics.selective_accuracy

    if conf_std is not None and math.isfinite(float(conf_std)) and float(conf_std) > float(confidence_std_threshold):
        return True
    if smear is not None and math.isfinite(float(smear)) and float(smear) > float(smearing_threshold):
        return True
    if selective_accuracy is not None and math.isfinite(float(selective_accuracy)) and float(selective_accuracy) < float(min_selective_accuracy):
        return True
    return False


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


def _require_scipy() -> None:
    if _sp_stats is None:
        raise PredictorError(
            "SciPy is required for the bio-stochastic electrophoresis engine (Shapiro-Wilk, Kruskal, chi-square, etc). "
            "Install it with: pip install -r requirements/base.txt"
        )


def _shapiro_p(values: np.ndarray, *, max_n: int = 5000, seed: int = 0) -> float | None:
    if _sp_stats is None:
        return None
    v = values[np.isfinite(values)].astype("float64")
    if v.size < 8:
        return None
    if v.size > max_n:
        rng = np.random.default_rng(int(seed))
        v = rng.choice(v, size=int(max_n), replace=False)
    # SciPy shapiro returns (stat, p)
    try:
        return float(_sp_stats.shapiro(v).pvalue)
    except Exception:
        return None


def _ionization_from_p(normality_p: float | None, *, alpha: float = 0.05) -> Literal["parametric", "nonparametric"]:
    if normality_p is None:
        return "nonparametric"
    return "parametric" if float(normality_p) > float(alpha) else "nonparametric"


def _mass_from_p(p_value: float | None) -> tuple[float, bool]:
    """Convert p-value to a stable 'mass' scalar.

    Larger mass => more stable/impactful compound.
    """

    if p_value is None or not math.isfinite(float(p_value)):
        return 0.0, False
    p = float(max(1e-300, min(1.0, float(p_value))))
    mass = float(np.clip(-math.log10(p), 0.0, 12.0))
    stable = p <= 0.05
    return mass, stable


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    if _sp_stats is None:
        return _pearson_corr(x, y), None
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0, None
    try:
        res = _sp_stats.pearsonr(x[mask], y[mask])
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return _pearson_corr(x, y), None


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    if _sp_stats is None:
        # fallback to Pearson-style corr of ranks (cheap approximation)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            return 0.0, None
        xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype="float64")
        yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype="float64")
        return _pearson_corr(xr, yr), None

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0, None
    try:
        res = _sp_stats.spearmanr(x[mask], y[mask])
        # SciPy returns statistic + pvalue, but can sometimes return NaN.
        stat = 0.0 if not math.isfinite(float(res.statistic)) else float(res.statistic)
        pval = None if res.pvalue is None or (not math.isfinite(float(res.pvalue))) else float(res.pvalue)
        return float(np.clip(stat, -1.0, 1.0)), pval
    except Exception:
        return 0.0, None


def _safe_ttest_ind(a: np.ndarray, b: np.ndarray) -> float | None:
    if _sp_stats is None:
        return None
    a0 = a[np.isfinite(a)]
    b0 = b[np.isfinite(b)]
    if a0.size < 3 or b0.size < 3:
        return None
    try:
        return float(_sp_stats.ttest_ind(a0, b0, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        return None


def _safe_mannwhitneyu(a: np.ndarray, b: np.ndarray) -> float | None:
    if _sp_stats is None:
        return None
    a0 = a[np.isfinite(a)]
    b0 = b[np.isfinite(b)]
    if a0.size < 3 or b0.size < 3:
        return None
    try:
        return float(_sp_stats.mannwhitneyu(a0, b0, alternative="two-sided").pvalue)
    except Exception:
        return None


def _safe_kruskal(groups: list[np.ndarray]) -> float | None:
    if _sp_stats is None:
        return None
    cleaned: list[np.ndarray] = []
    for g in groups:
        g0 = g[np.isfinite(g)]
        if g0.size >= 3:
            cleaned.append(g0)
    if len(cleaned) < 2:
        return None
    try:
        return float(_sp_stats.kruskal(*cleaned).pvalue)
    except Exception:
        return None


def _safe_anova(groups: list[np.ndarray]) -> float | None:
    if _sp_stats is None:
        return None
    cleaned: list[np.ndarray] = []
    for g in groups:
        g0 = g[np.isfinite(g)]
        if g0.size >= 3:
            cleaned.append(g0)
    if len(cleaned) < 2:
        return None
    try:
        return float(_sp_stats.f_oneway(*cleaned).pvalue)
    except Exception:
        return None


def _safe_chi2_p(x_cat: np.ndarray, y_cat: np.ndarray) -> float | None:
    if _sp_stats is None:
        return None
    try:
        df = pd.DataFrame({"x": x_cat, "y": y_cat}).dropna()
        if df.shape[0] < 5:
            return None
        ct = pd.crosstab(df["x"], df["y"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return None
        _, p, _, _ = _sp_stats.chi2_contingency(ct.to_numpy(dtype="float64"), correction=False)
        return float(p)
    except Exception:
        return None


def _rank_biserial_from_u(u: float, n1: int, n2: int, *, direction: float) -> float:
    # r_rb in [-1, 1]; direction should be +/-1 based on median difference.
    denom = max(1.0, float(n1) * float(n2))
    r = 1.0 - (2.0 * float(u) / denom)
    r = float(np.clip(r, -1.0, 1.0))
    return float(r * float(np.sign(direction) if direction != 0 else 1.0))


def _compute_compound_association(
    feature: pd.Series,
    target: pd.Series,
    feature_kind: FeatureKind,
    target_kind: TargetKind,
    *,
    train_mask: np.ndarray,
    random_seed: int,
    buffer_ionization: Literal["parametric", "nonparametric"] | None = None,
) -> tuple[float, str, bool, Literal["parametric", "nonparametric"], float | None, float | None, float, bool]:
    """Bio-stochastic association.

    Returns: (charge, method, signed, ionization, normality_p, p_value, mass, stable)
    """

    # Normality is defined for numeric-like features (compound ionization).
    normality_p: float | None = None
    if feature_kind in ("numeric", "datetime", "bool"):
        normality_p = _shapiro_p(_to_float_array(feature[train_mask], kind=feature_kind), seed=random_seed)
    ionization = _ionization_from_p(normality_p)

    # Default: use existing association as charge.
    charge, method, signed = _compute_association(feature[train_mask], target[train_mask], feature_kind, target_kind)
    p_value: float | None = None

    if target_kind in ("numeric", "datetime"):
        y = _to_float_array(target, kind=target_kind)
        if feature_kind in ("numeric", "datetime", "bool"):
            x = _to_float_array(feature, kind=feature_kind)
            # Dual-gate: if either the feature or the target buffer is nonparametric,
            # prefer rank-based association to reduce sensitivity to heavy tails/noise.
            tgt_ion = buffer_ionization
            use_parametric = (ionization == "parametric") and (tgt_ion in (None, "parametric"))
            if use_parametric:
                r, p = _safe_pearsonr(x[train_mask], y[train_mask])
                charge, method, signed = float(r), "pearson", True
                p_value = p
            else:
                r, p = _safe_spearmanr(x[train_mask], y[train_mask])
                charge, method, signed = float(r), "spearman", True
                p_value = p
        else:
            # Categorical feature vs numeric target: ANOVA (parametric) or Kruskal-Wallis (nonparam).
            x_cat = feature.astype("string").fillna("__MISSING__")
            groups = []
            for cat in pd.unique(x_cat[train_mask]):
                idx = (x_cat == cat).to_numpy() & train_mask
                groups.append(y[idx])
            gate = buffer_ionization if buffer_ionization is not None else ionization
            p_value = _safe_anova(groups) if gate == "parametric" else _safe_kruskal(groups)
            # Direction isn't interpretable; keep eta.
            charge, method, signed = _correlation_ratio(_to_category_array(x_cat[train_mask]), y[train_mask]), "eta", False

    else:
        # Target is categorical.
        y_cat = target.astype("string").fillna("__MISSING__")
        if feature_kind in ("numeric", "datetime", "bool"):
            x = _to_float_array(feature, kind=feature_kind)
            if _is_binary_categorical(y_cat[train_mask]):
                labels = pd.unique(y_cat[train_mask].dropna())
                positive = str(labels[0])
                pos_mask = (y_cat.astype("string") == positive).to_numpy() & train_mask
                neg_mask = (~pos_mask) & train_mask

                x_pos = x[pos_mask]
                x_neg = x[neg_mask]

                if ionization == "parametric":
                    p_value = _safe_ttest_ind(x_pos, x_neg)
                    y01 = (y_cat.astype("string") == positive).astype("float64").to_numpy(dtype="float64")
                    r, p = _safe_pearsonr(x, y01)
                    charge, method, signed = float(r), "point_biserial", True
                    p_value = p_value if p_value is not None else p
                else:
                    if _sp_stats is not None:
                        try:
                            res = _sp_stats.mannwhitneyu(
                                x_pos[np.isfinite(x_pos)],
                                x_neg[np.isfinite(x_neg)],
                                alternative="two-sided",
                            )
                            p_value = float(res.pvalue)
                            direction = float(np.nanmedian(x_pos) - np.nanmedian(x_neg))
                            charge = _rank_biserial_from_u(float(res.statistic), int(np.isfinite(x_pos).sum()), int(np.isfinite(x_neg).sum()), direction=direction)
                            method, signed = "mwu_rank_biserial", True
                        except Exception:
                            p_value = _safe_mannwhitneyu(x_pos, x_neg)
                            charge, method, signed = _pearson_corr(x, (y_cat.astype("string") == positive).astype("float64").to_numpy(dtype="float64")), "point_biserial", True
                    else:
                        p_value = _safe_mannwhitneyu(x_pos, x_neg)
                        charge, method, signed = _pearson_corr(x, (y_cat.astype("string") == positive).astype("float64").to_numpy(dtype="float64")), "point_biserial", True
            else:
                # Multi-class target: ANOVA or Kruskal across target classes.
                groups = []
                for cls in pd.unique(y_cat[train_mask]):
                    idx = (y_cat == cls).to_numpy() & train_mask
                    groups.append(x[idx])
                p_value = _safe_anova(groups) if ionization == "parametric" else _safe_kruskal(groups)
                charge, method, signed = _correlation_ratio(_to_category_array(y_cat[train_mask]), x[train_mask]), "eta", False
        else:
            # Categorical feature vs categorical target: chi-square p-value, cramers-v charge.
            x_cat = _to_category_array(feature[train_mask])
            y_arr = _to_category_array(y_cat[train_mask])
            p_value = _safe_chi2_p(x_cat, y_arr)
            charge, method, signed = _cramers_v(x_cat, y_arr), "cramers_v", False

    mass, stable = _mass_from_p(p_value)
    return float(charge), str(method), bool(signed), ionization, normality_p, p_value, float(mass), bool(stable)


def _collinearity_complexes(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_kinds: dict[str, FeatureKind],
    *,
    train_mask: np.ndarray,
    threshold: float = 0.9,
) -> tuple[dict[str, int], dict[int, list[str]], list[BondInfo]]:
    """Detect multicollinearity complexes among numeric-like features."""

    numeric_feats = [c for c in feature_cols if feature_kinds.get(c) in ("numeric", "datetime", "bool")]
    if len(numeric_feats) < 2:
        return {}, {}, []

    # Build adjacency based on absolute Pearson correlation on train.
    arrays: dict[str, np.ndarray] = {c: _to_float_array(df[c], kind=feature_kinds[c]) for c in numeric_feats}
    edges: list[tuple[str, str, float]] = []
    for i in range(len(numeric_feats)):
        a = numeric_feats[i]
        for j in range(i + 1, len(numeric_feats)):
            b = numeric_feats[j]
            corr = abs(_pearson_corr(arrays[a][train_mask], arrays[b][train_mask]))
            if math.isfinite(corr) and corr >= float(threshold):
                edges.append((a, b, float(corr)))

    if not edges:
        return {}, {}, []

    graph: dict[str, set[str]] = {c: set() for c in numeric_feats}
    for a, b, _ in edges:
        graph[a].add(b)
        graph[b].add(a)

    complex_by_feature: dict[str, int] = {}
    members_by_complex: dict[int, list[str]] = {}
    complex_id = 1
    for node in numeric_feats:
        if node in complex_by_feature:
            continue
        if not graph[node]:
            continue
        stack = [node]
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in complex_by_feature:
                continue
            complex_by_feature[cur] = complex_id
            comp.append(cur)
            for nxt in graph.get(cur, set()):
                if nxt not in complex_by_feature:
                    stack.append(nxt)
        if len(comp) >= 2:
            members_by_complex[complex_id] = sorted(comp)
            complex_id += 1
        else:
            # singletons not complexes
            complex_by_feature.pop(node, None)

    col_bonds: list[BondInfo] = []
    for a, b, corr in edges:
        col_bonds.append(
            BondInfo(
                feature_a=a,
                feature_b=b,
                affinity=float(corr),
                bonding_factor=float(1.0 + corr),
                bond_type="collinearity",
            )
        )
    return complex_by_feature, members_by_complex, col_bonds


def _fractionate_kw_zones(
    df: pd.DataFrame,
    *,
    feature_cols_used: list[str],
    feature_kinds: dict[str, FeatureKind],
    pI_map: dict[str, float],
    target_series: pd.Series,
    target_kind: TargetKind,
    train_mask: np.ndarray,
    start_zone_id: int = 100,
    max_classes: int = 6,
    max_levels: int = 8,
) -> list[EquilibriumZone]:
    """Fractionation sub-zoning using Kruskal-Wallis.

    - For categorical targets (>=3 classes): numeric-like features are shattered into per-class sub-zones.
    - For numeric targets: categorical features (>=3 levels) are shattered into per-level sub-zones.
    """

    zones: list[EquilibriumZone] = []
    zid = int(start_zone_id)
    try:
        if target_kind == "categorical":
            y_cat = target_series.astype("string").fillna("__MISSING__")
            classes = list(pd.Series(y_cat[train_mask]).value_counts().index)
            if len(classes) < 3:
                return []
            classes = classes[: max_classes]
            for col in feature_cols_used:
                fk = feature_kinds.get(col)
                if fk not in ("numeric", "datetime", "bool"):
                    continue
                x = _to_float_array(df[col], kind=fk)
                groups = []
                for cls in classes:
                    idx = (y_cat == str(cls)).to_numpy() & train_mask
                    groups.append(x[idx])
                p_kw = _safe_kruskal(groups)
                if p_kw is None or float(p_kw) > 0.05:
                    continue
                feats = [f"{col}::{cls}" for cls in classes]
                zones.append(
                    EquilibriumZone(
                        zone_id=int(zid),
                        features=feats,
                        avg_pI=float(pI_map.get(col, 0.5)),
                        avg_momentum=0.0,
                        strength=float(min(1.0, (-math.log10(max(1e-300, float(p_kw)))) / 6.0)),
                    )
                )
                zid += 1
        else:
            y_num = _to_float_array(target_series, kind=target_kind)
            for col in feature_cols_used:
                fk = feature_kinds.get(col)
                if fk != "categorical":
                    continue
                x_cat = df[col].astype("string").fillna("__MISSING__")
                levels = list(pd.Series(x_cat[train_mask]).value_counts().index)
                if len(levels) < 3:
                    continue
                levels = levels[: max_levels]
                groups = []
                for lv in levels:
                    idx = (x_cat == str(lv)).to_numpy() & train_mask
                    groups.append(y_num[idx])
                p_kw = _safe_kruskal(groups)
                if p_kw is None or float(p_kw) > 0.05:
                    continue
                feats = [f"{col}::{lv}" for lv in levels]
                zones.append(
                    EquilibriumZone(
                        zone_id=int(zid),
                        features=feats,
                        avg_pI=float(pI_map.get(col, 0.5)),
                        avg_momentum=0.0,
                        strength=float(min(1.0, (-math.log10(max(1e-300, float(p_kw)))) / 6.0)),
                    )
                )
                zid += 1
    except Exception:
        return []

    return zones


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


def _normalized_entropy(probs: np.ndarray) -> np.ndarray:
    if probs.size == 0:
        return np.zeros((0,), dtype="float64")
    k = int(probs.shape[1]) if probs.ndim == 2 else 1
    if k <= 1:
        return np.zeros((int(probs.shape[0]),), dtype="float64")
    p = np.clip(probs.astype("float64"), 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return (h / max(1e-12, float(np.log(float(k))))).astype("float64")


def _gel_health_regression(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None, float | None]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 8:
        return None, None, None
    yt = y_true[mask]
    yp = y_pred[mask]
    resid = yp - yt
    rmse = float(math.sqrt(float(np.mean(resid**2))))
    scale = float(np.nanstd(yt))
    rmse_norm = rmse / max(1e-9, scale)
    band_sharpness = float(np.clip(1.0 / (1.0 + rmse_norm), 0.0, 1.0))
    smearing = float(np.clip(rmse_norm, 0.0, 1.0))

    med = float(np.nanmedian(resid))
    mad = float(np.nanmedian(np.abs(resid - med)))
    thr = 3.5 * max(1e-9, mad)
    ghost_rate = float(np.mean((np.abs(resid - med) > thr).astype("float64")))
    return band_sharpness, smearing, float(np.clip(ghost_rate, 0.0, 1.0))


def _gel_health_classification(
    probs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    high_conf_threshold: float = 0.75,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if probs.size == 0 or probs.ndim != 2:
        return None, None, None, None, None
    if y_true.size == 0 or y_pred.size == 0:
        return None, None, None, None, None
    n = int(min(len(probs), len(y_true), len(y_pred)))
    if n < 8:
        return None, None, None, None, None
    p0 = probs[:n]
    yt = y_true[:n]
    yp = y_pred[:n]
    conf = np.max(p0, axis=1)
    conf_mean = float(np.mean(conf))
    conf_std = float(np.std(conf))
    entropy_mean = float(np.mean(_normalized_entropy(p0)))

    band_sharpness = float(np.clip(conf_mean - conf_std, 0.0, 1.0))
    smearing = float(np.clip(entropy_mean, 0.0, 1.0))
    wrong = (yt != yp)
    ghost_rate = float(np.mean(((conf >= float(high_conf_threshold)) & wrong).astype("float64")))
    return band_sharpness, smearing, float(np.clip(ghost_rate, 0.0, 1.0)), conf_mean, conf_std


def run_physics_prediction(
    df: pd.DataFrame,
    *,
    target_col: str,
    plane: PhysicsPlane = PhysicsPlane.solid,
    runtime_state: PredictorRuntimeState | None = None,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    top_k_weights: int = 30,
    max_preview_rows: int = 25,
    max_classes: int = 20,
    n_cycles: int = 30,
    cycle_learning_rate: float = 0.18,
    cycle_learning_rate_schedule: Literal["constant", "linear_decay", "cosine_decay"] = "constant",
    cycle_learning_rate_min_multiplier: float = 0.25,
    cycle_learning_rate_exp_decay: float = 1.0,
    shear_alpha: float = 0.75,
    shear_alpha_schedule: Literal["constant", "linear_decay", "cosine_decay"] = "constant",
    shear_alpha_min_multiplier: float = 0.25,
    top_bond_pairs: int = 20,
    n_zones: int = 5,
    cascade_enabled: bool = True,
    competitive_inhibition: bool = True,
    thermal_noise: bool = False,
    thermal_noise_cycles: int = 3,
    thermal_noise_level: float = 0.10,
    stage2_cycles: int = 2,
    stage2_trigger_cycle: int = 50,
    stage2_voltage_multiplier: float = 2.0,
    inhibition_strength: float = 0.7,
    scavenger_cycles: int = 1,
    stage2_shatter_complexes: bool = False,
    # Shatter-Reload (v4.4 experiment): after a chosen global cycle, "shatter" remaining
    # multicollinearity complexes (undo complex drag + remove coupling bond factor), then
    # optionally force a very low learning rate for the remaining cycles.
    shatter_reload_cycle: int = 0,
    shatter_reload_learning_rate: float = 0.0,
    shatter_reload_mode: Literal["drag_and_bond", "drag_only", "bond_only"] = "drag_and_bond",
    # Field-Effect coupling (v4.5 experiment, numeric targets): after a chosen global
    # cycle, apply a covariance-weighted pull across active features so updates are
    # influenced by correlated neighbors (Ridge-like global negotiation).
    field_effect_enabled: bool = False,
    field_effect_alpha: float = 0.0,
    field_effect_start_cycle: int = 0,
    field_effect_use_abs_corr: bool = True,
    field_effect_coupling: Literal["linear", "r_squared"] = "linear",
    field_effect_alpha_exp_decay: float = 1.0,
    # Optional convergence control: stop early if the predicted distribution stabilizes.
    early_stop_patience: int = 0,
    early_stop_tol: float = 1e-4,
    # Optional: oscillate viscosity during training (helps shake loose tangled dynamics).
    vibrational_viscosity_enabled: bool = False,
    vibrational_viscosity_period: int = 5,
    vibrational_viscosity_amplitude: float = 0.12,
    vibrational_viscosity_waveform: Literal["sine", "square"] = "square",
    # Target-induced viscosity scaling (buffer shift experiment): adapt viscosity based on
    # how far the model is from the target (uses normalized residual magnitude).
    target_induced_viscosity_enabled: bool = False,
    target_induced_viscosity_gain: float = 0.0,
    target_induced_viscosity_min_multiplier: float = 0.50,
    target_induced_viscosity_max_multiplier: float = 1.00,
    # Multi-Buffer / zone-specific chemistry (v4.6 experiment, numeric targets):
    # compute low/mid/high zones from current predictions using train-quantile thresholds,
    # then scale viscosity and Field-Effect coupling by zone.
    multibuffer_enabled: bool = False,
    multibuffer_q_low: float = 0.33,
    multibuffer_q_high: float = 0.67,
    multibuffer_low_viscosity_multiplier: float = 1.10,
    multibuffer_mid_viscosity_multiplier: float = 1.00,
    multibuffer_high_viscosity_multiplier: float = 1.00,
    multibuffer_low_field_alpha_multiplier: float = 1.00,
    multibuffer_mid_field_alpha_multiplier: float = 1.00,
    multibuffer_high_field_alpha_multiplier: float = 1.15,
    multibuffer_transition_frac: float = 0.0,
    low_confidence_mode: Literal["none", "flag", "abstain"] = "none",
    low_confidence_threshold: float = 0.0,
    low_confidence_entropy_threshold: float = 0.0,
    low_confidence_smear_metric: str = "entropy",
    low_confidence_combine_rule: Literal["or", "and"] = "or",
    low_confidence_auto_conf_quantile: float = 0.20,
    low_confidence_auto_smear_quantile: float = 0.80,
    low_confidence_safeguard_max_abstain: float = 0.95,
    # Confirmatory-band override (coverage expansion): keep mid-confidence rows when
    # many strong/stable features converge into the same migration zone.
    low_confidence_confirmatory_enabled: bool = False,
    low_confidence_confirmatory_conf_min: float = 0.50,
    low_confidence_confirmatory_conf_max: float = 0.90,
    low_confidence_confirmatory_consensus_threshold: float = 0.60,
    low_confidence_confirmatory_min_ion_hits: int = 0,
    # Target re-ionization: for rows marked low-confidence, run a small sub-cycle that only
    # updates those rows with a different "buffer" (effectively higher shear / lower inhibition).
    low_confidence_reionization_cycles: int = 0,
    low_confidence_reionization_shear_multiplier: float = 1.25,
    low_confidence_reionization_inhibition_multiplier: float = 0.75,
    # Secondary ionization (cascade expansion): optionally take the rows that would be
    # flagged/abstained and run a second, row-restricted refinement pass using a
    # lower effective viscosity and (optionally) rank-based charges for nonparametric
    # features. Includes a cluster-based promotion rule.
    low_confidence_secondary_enabled: bool = False,
    low_confidence_secondary_cycles: int = 0,
    low_confidence_secondary_viscosity_multiplier: float = 0.75,
    low_confidence_secondary_viscosity_anneal: bool = False,
    low_confidence_secondary_viscosity_multiplier_start: float | None = None,
    low_confidence_secondary_inhibition_multiplier: float = 0.85,
    low_confidence_secondary_shear_multiplier: float = 1.10,
    low_confidence_secondary_relax_ionization_gate: bool = True,
    low_confidence_secondary_ionization_z_min: float = 0.10,
    low_confidence_secondary_relaxed_ion_conf_min: float = 0.55,
    low_confidence_secondary_use_spearman: bool = True,
    low_confidence_secondary_spearman_min_abs: float = 0.015,
    low_confidence_secondary_spearman_margin: float = 0.010,
    low_confidence_secondary_promote_min_zone_votes: int = 3,
    low_confidence_secondary_promote_z_min: float = 0.50,
    low_confidence_secondary_promote_conf_min: float = 0.42,
    # Reciprocating Sieve (v4.2): attempt to shake loose tangled, low-confidence rows
    # that have high instability but near-zero net update during secondary ionization.
    low_confidence_secondary_sieve_enabled: bool = False,
    low_confidence_secondary_sieve_cycles: int = 2,
    low_confidence_secondary_sieve_reverse_multiplier: float = 0.75,
    low_confidence_secondary_sieve_noise_std: float = 0.04,
    low_confidence_secondary_sieve_instability_min: float = 0.65,
    low_confidence_secondary_sieve_conf_delta_max: float = 0.002,
    low_confidence_secondary_sieve_update_norm_max: float = 0.003,
    # Ionization/viscosity-aware abstention controls.
    low_confidence_require_ionized: bool = False,
    low_confidence_ionization_pvalue: float = 0.05,
    low_confidence_ionization_z_min: float = 0.25,
    low_confidence_viscosity_override: bool = False,
    low_confidence_viscosity_override_threshold: float = 1.0,
    low_confidence_label: str = "__LOW_CONFIDENCE__",
    abstain_if_uncertain: bool = False,
    abstain_confidence_std_threshold: float = 0.25,
    abstain_smearing_threshold: float = 0.65,
    abstain_min_selective_accuracy: float = 0.25,
    enable_isotopes: bool = True,
    isotope_max_numeric_features: int = 6,
    isotope_max_category_levels: int = 8,
    isotope_max_total_features: int = 48,
    # Primary-stage micro-shakes (v4.3 experiment): apply scheduled sieve pulses during
    # the main electrophoresis cycles to prevent tangles from solidifying.
    low_confidence_primary_sieve_enabled: bool = False,
    low_confidence_primary_sieve_cycle_a: int = 30,
    low_confidence_primary_sieve_cycle_b: int = 45,
    low_confidence_primary_sieve_shake_cycles: int = 2,
    low_confidence_primary_sieve_reverse_multiplier: float = 1.0,
    low_confidence_primary_sieve_noise_std: float = 0.08,
    low_confidence_primary_sieve_instability_min: float = 0.50,
    low_confidence_primary_sieve_conf_delta_max: float = 0.003,
    # PCR-style feature amplification (v5.0 sprout): boost statistically significant
    # features (primer binds when p < threshold) by increasing their contribution
    # to the update numerator (without expanding X).
    pcr_enabled: bool = False,
    pcr_cycles: int = 0,
    pcr_pvalue_threshold: float = 0.05,
    pcr_tau: float = 4.0,
    pcr_gain: float = 0.55,
    pcr_strength_cap: float = 2.5,
    pcr_amp_cap: float = 3.5,
    pcr_require_stable: bool = True,
    cleaning_enabled: bool = True,
    cleaning_drop_duplicates: bool = True,
    cleaning_drop_missing_target: bool = True,
    cleaning_impute_missing: bool = True,
    cleaning_clip_numeric_outliers: bool = True,
    cleaning_outlier_strategy: Literal["winsorize", "iqr", "gaussian", "mad", "arbitrary", "feature_engine", "none"] = "winsorize",
    cleaning_outlier_fold: float = 1.5,
    cleaning_outlier_q_low: float = 0.005,
    cleaning_outlier_q_high: float = 0.995,
    cleaning_arbitrary_min: float | None = None,
    cleaning_arbitrary_max: float | None = None,
    cleaning_rolling_window: int | None = None,
    cleaning_rolling_window_cadence_hz: float | None = None,
    cleaning_rolling_window_seconds: float = 60.0,
    cleaning_rolling_mad_fold: float = 3.0,
    cleaning_rolling_cluster_min_size: int = 3,
    return_predictions: bool = False,
) -> PredictionResult | None:
    if target_col not in df.columns:
        raise PredictorError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    if runtime_state is not None and runtime_state.preferred_plane is not None:
        plane = runtime_state.preferred_plane

    # Basic cleaning (dedupe + drop missing target) BEFORE split, so masks align with cleaned df.
    cleaning_diag: dict[str, Any] | None = None
    if bool(cleaning_enabled):
        df, cleaning_diag = _clean_dataframe_for_prediction(
            df,
            target_col=target_col,
            train_mask=None,
            drop_duplicates=bool(cleaning_drop_duplicates),
            drop_missing_target=bool(cleaning_drop_missing_target),
            impute_missing=False,
            clip_numeric_outliers=False,
        )

    if df.shape[0] < 3:
        raise PredictorError("Need at least 3 rows")

    train_mask, test_mask = _train_test_split_mask(int(df.shape[0]), train_fraction, random_seed)

    isotope_diag: dict[str, Any] | None = None
    isotope_cols: list[str] = []
    if bool(enable_isotopes):
        feature_cols_pre = _select_feature_columns(df, target_col)
        feature_kinds_pre = {c: infer_feature_kind(df[c]) for c in feature_cols_pre}
        df, isotope_cols, isotope_diag = _isotope_feature_bundle(
            df,
            feature_cols_pre,
            feature_kinds_pre,
            target_series=df[target_col],
            target_kind=infer_target_kind(df[target_col]),
            train_mask=train_mask,
            max_numeric_features=int(isotope_max_numeric_features),
            max_category_levels=int(isotope_max_category_levels),
            max_total_isotopes=int(isotope_max_total_features),
        )
        if isotope_cols:
            train_mask, test_mask = _train_test_split_mask(int(df.shape[0]), train_fraction, random_seed)

    # Train-stat cleaning (impute + outlier clip) AFTER split, using TRAIN stats only.
    if bool(cleaning_enabled):
        df, cleaning_diag2 = _clean_dataframe_for_prediction(
            df,
            target_col=target_col,
            train_mask=train_mask,
            drop_duplicates=False,
            drop_missing_target=False,
            impute_missing=bool(cleaning_impute_missing),
            clip_numeric_outliers=bool(cleaning_clip_numeric_outliers),
            outlier_strategy=str(cleaning_outlier_strategy),
            outlier_fold=float(cleaning_outlier_fold),
            outlier_q_low=float(cleaning_outlier_q_low),
            outlier_q_high=float(cleaning_outlier_q_high),
            arbitrary_min=cleaning_arbitrary_min,
            arbitrary_max=cleaning_arbitrary_max,
            rolling_window=cleaning_rolling_window,
            rolling_window_cadence_hz=cleaning_rolling_window_cadence_hz,
            rolling_window_seconds=float(cleaning_rolling_window_seconds),
            rolling_mad_fold=float(cleaning_rolling_mad_fold),
            rolling_cluster_min_size=int(cleaning_rolling_cluster_min_size),
        )
        # Merge per-phase diagnostics (prefer post-split flags/stats).
        if cleaning_diag is None:
            cleaning_diag = cleaning_diag2
        else:
            cleaning_diag = {**cleaning_diag, **cleaning_diag2}

    _require_scipy()

    vib_enabled = bool(vibrational_viscosity_enabled)
    vib_period = int(vibrational_viscosity_period)
    vib_amp = float(vibrational_viscosity_amplitude)
    vib_wave = str(vibrational_viscosity_waveform).lower().strip()
    if vib_wave not in ("sine", "square"):
        vib_wave = "square"

    homeostasis_gain = _adaptive_gain_from_state(1.0, runtime_state)

    lr_schedule = str(cycle_learning_rate_schedule).lower().strip()
    if lr_schedule not in ("constant", "linear_decay", "cosine_decay", "exp_decay"):
        lr_schedule = "constant"
    lr_min_mult = float(cycle_learning_rate_min_multiplier)
    if not math.isfinite(lr_min_mult):
        lr_min_mult = 0.25
    lr_min_mult = float(np.clip(lr_min_mult, 0.0, 1.0))

    lr_exp = float(cycle_learning_rate_exp_decay)
    if not math.isfinite(lr_exp):
        lr_exp = 1.0
    lr_exp = float(np.clip(lr_exp, 0.0, 1.0))

    shear_schedule = str(shear_alpha_schedule).lower().strip()
    if shear_schedule not in ("constant", "linear_decay", "cosine_decay"):
        shear_schedule = "constant"
    shear_min_mult = float(shear_alpha_min_multiplier)
    if not math.isfinite(shear_min_mult):
        shear_min_mult = 0.25
    shear_min_mult = float(np.clip(shear_min_mult, 0.0, 1.0))

    def _schedule_multiplier(
        cycle_1based: int,
        total_cycles: int,
        *,
        kind: str,
        min_multiplier: float,
    ) -> float:
        if kind == "constant" or int(total_cycles) <= 1:
            return 1.0
        c = int(max(1, cycle_1based))
        t = float(c - 1) / float(max(1, int(total_cycles) - 1))
        t = float(np.clip(t, 0.0, 1.0))
        m = float(np.clip(float(min_multiplier), 0.0, 1.0))
        if kind == "linear_decay":
            return (1.0 - t) + m * t
        # cosine_decay
        return m + 0.5 * (1.0 - m) * (1.0 + math.cos(math.pi * t))

    def _lr_at(cycle_1based: int, total_cycles: int) -> float:
        base = float(cycle_learning_rate) * float(homeostasis_gain)
        if not math.isfinite(base) or base <= 0.0:
            base = 0.18
        if lr_schedule == "exp_decay":
            c = int(max(1, cycle_1based))
            if lr_exp <= 0.0:
                m = lr_min_mult
            else:
                m = float(lr_exp) ** float(c - 1)
            m = float(np.clip(m, lr_min_mult, 1.0))
            return base * m

        return base * _schedule_multiplier(cycle_1based, total_cycles, kind=lr_schedule, min_multiplier=lr_min_mult)

    def _shear_at(cycle_1based: int, total_cycles: int) -> float:
        base = float(shear_alpha)
        if not math.isfinite(base) or base < 0.0:
            base = 0.0
        return base * _schedule_multiplier(cycle_1based, total_cycles, kind=shear_schedule, min_multiplier=shear_min_mult)

    shatter_cycle = int(shatter_reload_cycle)
    if shatter_cycle < 0:
        shatter_cycle = 0
    shatter_lr = float(shatter_reload_learning_rate)
    if not math.isfinite(shatter_lr) or shatter_lr <= 0.0:
        shatter_lr = 0.0

    shatter_mode = str(shatter_reload_mode).lower().strip()
    if shatter_mode not in ("drag_and_bond", "drag_only", "bond_only"):
        shatter_mode = "drag_and_bond"

    def _lr_effective(global_cycle_1based: int, total_cycles: int) -> float:
        lr0 = float(_lr_at(global_cycle_1based, total_cycles))
        if shatter_cycle > 0 and shatter_lr > 0.0 and int(global_cycle_1based) > shatter_cycle:
            return float(shatter_lr)
        return lr0

    def _complex_shattered(global_cycle_1based: int) -> bool:
        return bool(shatter_cycle > 0 and int(global_cycle_1based) > shatter_cycle)

    def _apply_shatter_to_feature(
        *,
        global_cycle_1based: int,
        medium: MigrationInfo,
        col: str,
        bond_factor: float,
        eta_base: float,
    ) -> tuple[float, float]:
        if not _complex_shattered(global_cycle_1based):
            return bond_factor, eta_base
        if medium.complex_size is None or int(medium.complex_size) < 2:
            return bond_factor, eta_base

        bf = float(bond_factor)
        eb = float(eta_base)
        if shatter_mode in ("drag_and_bond", "bond_only"):
            bf = 1.0
        if shatter_mode in ("drag_and_bond", "drag_only"):
            eb = max(1e-6, eb / float(complex_drag_by_feature.get(col, 1.0)))
        return bf, eb

    field_enabled = bool(field_effect_enabled)
    field_alpha = float(field_effect_alpha)
    if not math.isfinite(field_alpha) or field_alpha <= 0.0:
        field_alpha = 0.0
        field_enabled = False
    field_start = int(field_effect_start_cycle)
    if field_start < 0:
        field_start = 0
    field_abs = bool(field_effect_use_abs_corr)

    field_coupling = str(field_effect_coupling).lower().strip()
    if field_coupling not in ("linear", "r_squared"):
        field_coupling = "linear"

    field_alpha_decay = float(field_effect_alpha_exp_decay)
    if not math.isfinite(field_alpha_decay) or field_alpha_decay <= 0.0:
        field_alpha_decay = 1.0

    def _field_active(global_cycle_1based: int) -> bool:
        return bool(field_enabled and field_alpha > 0.0 and field_start > 0 and int(global_cycle_1based) >= field_start)

    def _field_alpha_at(global_cycle_1based: int) -> float:
        if not _field_active(global_cycle_1based):
            return 0.0
        # Effective alpha can slowly grow (>1) or decay (<1) after activation.
        k = max(0, int(global_cycle_1based) - int(field_start))
        try:
            a_eff = float(field_alpha) * float(field_alpha_decay) ** float(k)
        except Exception:
            a_eff = float(field_alpha)
        if not math.isfinite(a_eff) or a_eff <= 0.0:
            return 0.0
        return float(a_eff)

    tiv_enabled = bool(target_induced_viscosity_enabled)
    tiv_gain = float(target_induced_viscosity_gain)
    if not math.isfinite(tiv_gain) or tiv_gain <= 0.0:
        tiv_enabled = False
        tiv_gain = 0.0
    tiv_min = float(target_induced_viscosity_min_multiplier)
    tiv_max = float(target_induced_viscosity_max_multiplier)
    if not math.isfinite(tiv_min):
        tiv_min = 0.50
    if not math.isfinite(tiv_max):
        tiv_max = 1.00
    tiv_min = float(np.clip(tiv_min, 0.05, 1.0))
    tiv_max = float(np.clip(tiv_max, tiv_min, 1.0))

    def _target_induced_viscosity_multiplier(residual_train: np.ndarray, residual_std: float) -> float:
        if not tiv_enabled:
            return 1.0
        rs = float(residual_std)
        if not math.isfinite(rs) or rs <= 1e-12:
            rs = 1.0
        lvl = float(np.nanmean(np.abs(residual_train))) / (rs + 1e-9)
        if not math.isfinite(lvl) or lvl < 0.0:
            lvl = 0.0
        mult = 1.0 / (1.0 + tiv_gain * lvl)
        if not math.isfinite(mult):
            mult = 1.0
        return float(np.clip(mult, tiv_min, tiv_max))

    mb_enabled = bool(multibuffer_enabled)
    mb_q_low = float(multibuffer_q_low)
    mb_q_high = float(multibuffer_q_high)
    if not math.isfinite(mb_q_low):
        mb_q_low = 0.33
    if not math.isfinite(mb_q_high):
        mb_q_high = 0.67
    mb_q_low = float(np.clip(mb_q_low, 0.01, 0.99))
    mb_q_high = float(np.clip(mb_q_high, 0.01, 0.99))
    if mb_q_low >= mb_q_high:
        mb_q_low, mb_q_high = 0.33, 0.67

    mb_visc_low = float(multibuffer_low_viscosity_multiplier)
    mb_visc_mid = float(multibuffer_mid_viscosity_multiplier)
    mb_visc_high = float(multibuffer_high_viscosity_multiplier)
    if not math.isfinite(mb_visc_low) or mb_visc_low <= 0.0:
        mb_visc_low = 1.10
    if not math.isfinite(mb_visc_mid) or mb_visc_mid <= 0.0:
        mb_visc_mid = 1.00
    if not math.isfinite(mb_visc_high) or mb_visc_high <= 0.0:
        mb_visc_high = 1.00

    mb_alpha_low = float(multibuffer_low_field_alpha_multiplier)
    mb_alpha_mid = float(multibuffer_mid_field_alpha_multiplier)
    mb_alpha_high = float(multibuffer_high_field_alpha_multiplier)
    if not math.isfinite(mb_alpha_low) or mb_alpha_low <= 0.0:
        mb_alpha_low = 1.00
    if not math.isfinite(mb_alpha_mid) or mb_alpha_mid <= 0.0:
        mb_alpha_mid = 1.00
    if not math.isfinite(mb_alpha_high) or mb_alpha_high <= 0.0:
        mb_alpha_high = 1.15

    mb_transition_frac = float(multibuffer_transition_frac)
    if not math.isfinite(mb_transition_frac) or mb_transition_frac <= 0.0:
        mb_transition_frac = 0.0
    mb_transition_frac = float(np.clip(mb_transition_frac, 0.0, 0.50))

    target_series = df[target_col]
    target_kind = infer_target_kind(target_series)

    # Active Buffer ionization (numeric-like targets only).
    buffer_normality_p: float | None = None
    buffer_ionization: Literal["parametric", "nonparametric"] | None = None
    if target_kind in ("numeric", "datetime"):
        buffer_normality_p = _shapiro_p(
            _to_float_array(target_series[train_mask], kind=target_kind),
            seed=int(random_seed) + 9101,
        )
        buffer_ionization = _ionization_from_p(buffer_normality_p)

    feature_cols = _select_feature_columns(df, target_col)
    if not feature_cols:
        raise PredictorError("No features available (dataset only contains the target column)")

    feature_kinds: dict[str, FeatureKind] = {c: infer_feature_kind(df[c]) for c in feature_cols}
    bonds = _build_bonding_map(df[feature_cols][train_mask], feature_cols, feature_kinds, top_pairs=top_bond_pairs)

    # Coupling (collinearity) bonds (added on top of general affinity bonds).
    complex_by_feature, members_by_complex, col_bonds = _collinearity_complexes(
        df,
        feature_cols,
        feature_kinds,
        train_mask=train_mask,
        threshold=0.9,
    )
    if col_bonds:
        bonds = list(bonds) + list(col_bonds)

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
    unstable_features: set[str] = set()

    # Pass 1: compute per-feature bio-stochastic stats on TRAIN (so complexes can interact).
    compound_stats: dict[str, dict[str, Any]] = {}
    for col in feature_cols:
        feat = df[col]
        fk = feature_kinds[col]
        (
            w,
            method,
            signed,
            ionization,
            normality_p,
            p_value,
            mass,
            stable,
        ) = _compute_compound_association(
            feat,
            target_series,
            fk,
            target_kind,
            train_mask=train_mask,
            random_seed=int(random_seed) + 7,
            buffer_ionization=buffer_ionization,
        )
        if not math.isfinite(w):
            w = 0.0

        if fk in ("numeric", "datetime", "bool"):
            entropy, variance, stderr = _numeric_entropy_and_variance(_to_float_array(feat[train_mask], kind=fk))
        else:
            entropy, variance, stderr = _categorical_entropy_and_variance(feat[train_mask])

        kl = _kl_for_feature(feat[train_mask], fk, global_numeric_ref, global_categorical_ref)
        certainty = 1.0 / (1.0 + max(0.0, float(stderr)))
        density_base = (1.0 + max(0.0, float(kl))) * float(certainty)

        if not bool(stable):
            unstable_features.add(col)

        # Correlation/effect-size strength drives the target-gradient field.
        strength = float(np.clip(abs(float(w)), 0.0, 1.0))

        cid = complex_by_feature.get(col)
        csize: int | None = None
        if cid is not None:
            csize = len(members_by_complex.get(int(cid), []))

        compound_stats[col] = {
            "w": float(w),
            "method": str(method),
            "signed": bool(signed),
            "ionization": ionization,
            "normality_p": normality_p,
            "p_value": p_value,
            "mass": float(mass),
            "stable": bool(stable),
            "strength": float(strength),
            "entropy": float(entropy),
            "variance": float(variance),
            "stderr": float(stderr),
            "kl": float(kl),
            "density_base": float(density_base),
            "complex_id": None if cid is None else int(cid),
            "complex_size": None if csize is None else int(csize),
        }

    # PCR primer binding: compute per-feature amplification factors from TRAIN stats.
    pcr_enabled0 = bool(pcr_enabled)
    pcr_cycles0 = int(pcr_cycles)
    pcr_pthr0 = float(pcr_pvalue_threshold)
    pcr_tau0 = float(pcr_tau)
    pcr_gain0 = float(pcr_gain)
    pcr_strength_cap0 = float(pcr_strength_cap)
    pcr_amp_cap0 = float(pcr_amp_cap)
    pcr_require_stable0 = bool(pcr_require_stable)

    pcr_amp_by_feature: dict[str, float] = {}
    pcr_strength_by_feature: dict[str, float] = {}
    if pcr_enabled0 and pcr_cycles0 > 0:
        for col in feature_cols:
            s = compound_stats.get(col) or {}
            amp, strength = _pcr_amplification_factor(
                p_value=s.get("p_value"),
                stable=bool(s.get("stable", False)),
                enabled=pcr_enabled0,
                cycles=pcr_cycles0,
                p_threshold=pcr_pthr0,
                tau=pcr_tau0,
                gain=pcr_gain0,
                strength_cap=pcr_strength_cap0,
                amp_cap=pcr_amp_cap0,
                require_stable=pcr_require_stable0,
            )
            if amp > 1.0:
                pcr_amp_by_feature[str(col)] = float(amp)
                pcr_strength_by_feature[str(col)] = float(strength)

    # Complex anchoring + dissociation: anchors should primarily trap themselves,
    # not drag stable, high-signal neighbors into syrupy (high-viscosity) zones.
    complex_drag_by_feature: dict[str, float] = {}
    for cid, members in members_by_complex.items():
        if not members:
            continue
        strengths = [float(compound_stats[m]["strength"]) for m in members if m in compound_stats]
        if not strengths:
            continue
        avg_strength = float(np.mean(strengths))
        unstable_ratio = float(
            np.mean([1.0 if (m in unstable_features) else 0.0 for m in members if m in compound_stats])
        )
        has_anchor = any((compound_stats[m]["strength"] < 0.20) or (m in unstable_features) for m in members if m in compound_stats)
        if not has_anchor:
            for m in members:
                complex_drag_by_feature[str(m)] = 1.0
        else:
            # Global complex turbulence baseline.
            complex_drag = 1.0 + 0.45 * float(max(0.0, 1.0 - avg_strength)) + 0.30 * float(np.clip(unstable_ratio, 0.0, 1.0))
            complex_drag = float(np.clip(complex_drag, 1.0, 2.0))
            for m in members:
                m_strength = float(compound_stats[m]["strength"])
                is_unstable = m in unstable_features
                # Anchors get most of the drag; stable members are mostly dissociated.
                if is_unstable or m_strength < 0.20:
                    complex_drag_by_feature[str(m)] = float(np.clip(complex_drag * 1.15, 1.0, 2.25))
                else:
                    complex_drag_by_feature[str(m)] = 1.0

    # Pass 2: compute migration fields with target-gradient + complex interaction.
    for col in feature_cols:
        fk = feature_kinds[col]
        s = compound_stats[col]
        w = float(s["w"])
        method = str(s["method"])
        signed = bool(s["signed"])
        ionization = s["ionization"]
        normality_p = s["normality_p"]
        p_value = s["p_value"]
        mass_raw = float(s["mass"])
        stable = bool(s["stable"])
        strength = float(s["strength"])
        entropy = float(s["entropy"])
        variance = float(s["variance"])
        stderr = float(s["stderr"])
        kl = float(s["kl"])
        density = float(s["density_base"])
        cid = s["complex_id"]
        csize: int | None = s["complex_size"]

        # Complex-aware molecular weight: internal coupling and dimensionality add inertia.
        complex_scale = 1.0
        if csize is not None and int(csize) >= 2:
            complex_scale = 1.0 + 0.18 * float(min(6, int(csize)) - 1)
        mass = float(mass_raw * complex_scale)

        # Density flux: significant compounds (high mass) carry more certainty weight.
        mass_norm = float(np.clip(float(mass) / 6.0, 0.0, 1.0))
        density *= 1.0 + 0.55 * mass_norm

        # Coupling gives a mild density lift (transport coupling).
        if csize is not None and int(csize) >= 2:
            density *= 1.0 + 0.10 * float(min(6, int(csize)) - 1)

        bond_factor = bond_factors.get(col, 1.0)

        complex_drag = float(complex_drag_by_feature.get(col, 1.0))

        viscosity = _calculate_viscosity_field(
            plane=plane,
            entropy=entropy,
            variance=variance,
            correlation_strength=strength,
            mass=mass,
            ionization=ionization,
            unstable=(col in unstable_features),
            complex_drag=complex_drag,
        )

        # F = m a analog: heavier complexes accelerate less.
        inertia = float(1.0 + 0.90 * mass_norm)

        charge = float(w)
        if charge < 0:
            charge *= neg_mult
        terminal_velocity = plane_mobility * (charge * density * bond_factor) / (viscosity * inertia)

        if terminal_velocity > 1e-10:
            direction: Literal["pulled", "repelled", "neutral"] = "pulled"
        elif terminal_velocity < -1e-10:
            direction = "repelled"
        else:
            direction = "neutral"

        state = _migration_state(terminal_velocity, viscosity)
        if col in unstable_features and state == "free":
            state = "dampened"

        weights.append(WeightInfo(feature=col, weight=float(w), method=method, feature_kind=fk, signed=bool(signed)))
        migration_map.append(
            MigrationInfo(
                feature=col,
                feature_kind=fk,
                method=method,
                charge=float(charge),
                ionization=ionization,
                normality_p=normality_p,
                p_value=p_value,
                mass=float(mass),
                stable=bool(stable),
                complex_id=None if cid is None else int(cid),
                complex_size=None if csize is None else int(csize),
                entropy=float(entropy),
                variance=float(variance),
                standard_error=float(stderr),
                kl_divergence=float(kl),
                density=float(density),
                viscosity=float(viscosity),
                terminal_velocity=float(terminal_velocity),
                arrival_speed=float(abs(terminal_velocity)),
                direction=direction,
                state=state,
            )
        )

    def _selection_score(wi: WeightInfo) -> float:
        s0 = compound_stats.get(wi.feature)
        mass0 = 0.0 if not s0 else float(s0.get("mass", 0.0))
        mass_norm0 = float(np.clip(mass0 / 6.0, 0.0, 1.0))
        # Favor statistically stable compounds (high mass) to reduce random/noise lift.
        return float(abs(wi.weight) * (0.35 + 0.65 * mass_norm0))

    weights_sorted = sorted(weights, key=_selection_score, reverse=True)
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

        # Multi-Buffer thresholds are derived from TRAIN target distribution only.
        # Zones are assigned per-row using the model's current predictions (avoids leakage).
        mb_t_low: float | None = None
        mb_t_high: float | None = None
        mb_transition_width: float = 0.0
        multibuffer_diag: dict[str, Any] | None = None
        if mb_enabled:
            try:
                y_train_f = y_train[np.isfinite(y_train)]
                if y_train_f.size >= 16:
                    mb_t_low = float(np.nanquantile(y_train_f, mb_q_low))
                    mb_t_high = float(np.nanquantile(y_train_f, mb_q_high))
                    if not (math.isfinite(mb_t_low) and math.isfinite(mb_t_high) and mb_t_low < mb_t_high):
                        mb_t_low = None
                        mb_t_high = None
                else:
                    mb_t_low = None
                    mb_t_high = None
            except Exception:
                mb_t_low = None
                mb_t_high = None

            if mb_t_low is None or mb_t_high is None:
                mb_enabled = False
            else:
                mb_transition_width = float(mb_transition_frac) * float(mb_t_high - mb_t_low)
                if not math.isfinite(mb_transition_width) or mb_transition_width <= 0.0:
                    mb_transition_width = 0.0
                multibuffer_diag = {
                    "enabled": True,
                    "q_low": float(mb_q_low),
                    "q_high": float(mb_q_high),
                    "t_low": float(mb_t_low),
                    "t_high": float(mb_t_high),
                    "transition_frac": float(mb_transition_frac),
                    "transition_width": float(mb_transition_width),
                    "viscosity_multipliers": {
                        "low": float(mb_visc_low),
                        "mid": float(mb_visc_mid),
                        "high": float(mb_visc_high),
                    },
                    "field_alpha_multipliers": {
                        "low": float(mb_alpha_low),
                        "mid": float(mb_alpha_mid),
                        "high": float(mb_alpha_high),
                    },
                }

        def _multibuffer_multipliers(pred_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # Returns (viscosity_multiplier_vec, field_alpha_multiplier_vec).
            # Default behavior is neutral (all ones).
            n = int(pred_now.shape[0])
            if not mb_enabled or mb_t_low is None or mb_t_high is None:
                ones = np.ones(n, dtype="float64")
                return ones, ones

            finite = np.isfinite(pred_now)
            if mb_transition_width <= 0.0:
                zone = np.full(n, 1, dtype="int8")  # default mid
                if bool(np.any(finite)):
                    p = pred_now
                    zone[(finite) & (p < float(mb_t_low))] = 0
                    zone[(finite) & (p >= float(mb_t_high))] = 2

                visc = np.ones(n, dtype="float64")
                alpha = np.ones(n, dtype="float64")
                visc[zone == 0] = float(mb_visc_low)
                visc[zone == 1] = float(mb_visc_mid)
                visc[zone == 2] = float(mb_visc_high)
                alpha[zone == 0] = float(mb_alpha_low)
                alpha[zone == 1] = float(mb_alpha_mid)
                alpha[zone == 2] = float(mb_alpha_high)
                return visc, alpha

            # Soft (sigmoidal) transitions to reduce boundary oscillation.
            p = pred_now.astype("float64", copy=False)
            p = np.where(finite, p, 0.5 * (float(mb_t_low) + float(mb_t_high)))

            w = float(mb_transition_width)
            z_low = (p - float(mb_t_low)) / w
            z_high = (p - float(mb_t_high)) / w
            z_low = np.clip(z_low, -60.0, 60.0)
            z_high = np.clip(z_high, -60.0, 60.0)
            s_low = 1.0 / (1.0 + np.exp(-z_low))
            s_high = 1.0 / (1.0 + np.exp(-z_high))

            w_low = 1.0 - s_low
            w_high = s_high
            w_mid = np.clip(s_low - s_high, 0.0, 1.0)

            w_sum = w_low + w_mid + w_high
            w_sum = np.where(w_sum > 1e-12, w_sum, 1.0)
            w_low = w_low / w_sum
            w_mid = w_mid / w_sum
            w_high = w_high / w_sum

            visc = w_low * float(mb_visc_low) + w_mid * float(mb_visc_mid) + w_high * float(mb_visc_high)
            alpha = w_low * float(mb_alpha_low) + w_mid * float(mb_alpha_mid) + w_high * float(mb_alpha_high)
            visc = np.clip(visc, 1e-6, 1e9)
            alpha = np.clip(alpha, 1e-6, 1e9)
            return visc.astype("float64", copy=False), alpha.astype("float64", copy=False)
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

        # Field-Effect coupling matrix (train-derived z-correlation among active features).
        field_features = [w.feature for w in weights_used]
        field_pos = {f: i for i, f in enumerate(field_features)}
        field_Z: np.ndarray | None = None
        field_C: np.ndarray | None = None
        if field_enabled and len(field_features) >= 2:
            try:
                field_Z = np.column_stack([z_by_feature[f] for f in field_features]).astype("float64", copy=False)
                Zt = field_Z[train_mask]
                C = np.corrcoef(Zt, rowvar=False)
                if field_abs:
                    C = np.abs(C)
                if field_coupling == "r_squared":
                    # Weight strong correlations disproportionately; keep sign if abs_corr is disabled.
                    C = np.sign(C) * (C**2)
                C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(C, 0.0)
                field_C = C.astype("float64", copy=False)
            except Exception:
                field_Z = None
                field_C = None

        # Keep PCR amps only for the used features.
        if pcr_amp_by_feature:
            pcr_amp_by_feature = {k: v for k, v in pcr_amp_by_feature.items() if k in z_by_feature}

        iteration_gains: list[IterationInfo] = []
        n_cycles_eff = max(1, int(n_cycles))
        grad1 = 0.35
        rng_stage1 = np.random.default_rng(int(random_seed) + 31007)

        total_schedule_cycles = int(n_cycles_eff)
        if cascade_enabled and len(zone1_features) >= 3:
            total_schedule_cycles += int(max(1, stage2_cycles))
            if int(scavenger_cycles) > 0:
                total_schedule_cycles += int(scavenger_cycles)
        total_schedule_cycles = int(max(1, total_schedule_cycles))

        for cycle in range(1, n_cycles_eff + 1):
            lr = float(_lr_effective(cycle, total_schedule_cycles))
            shear_eff = float(_shear_at(cycle, total_schedule_cycles))
            residual = y - pred
            residual_train = residual[train_mask]
            residual_std = float(np.nanstd(residual_train))
            if not math.isfinite(residual_std) or residual_std <= 1e-12:
                residual_std = 1.0
            eta_scale = _target_induced_viscosity_multiplier(residual_train, residual_std)

            mb_visc_vec, mb_alpha_vec = _multibuffer_multipliers(pred)
            eta_scale_vec = np.asarray(mb_visc_vec, dtype="float64") * float(eta_scale)
            eta_scale_vec = np.clip(eta_scale_vec, 1e-6, 1e9)

            update_score = np.zeros(df.shape[0], dtype="float64")
            denom = 0.0

            # Collect per-feature velocities to optionally apply a Field-Effect coupling.
            a = np.zeros(len(weights_used), dtype="float64")

            for i_w, wi in enumerate(weights_used):
                col = wi.feature
                z = z_by_feature[col]
                charge = _pearson_corr(z[train_mask], residual_train)
                if not math.isfinite(charge) or abs(charge) < 1e-8:
                    continue

                medium = migration_by_feature[col]
                bond_factor = bond_factors.get(col, 1.0)
                certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                density = (1.0 + max(0.0, medium.kl_divergence)) * certainty
                eta_base = max(1e-6, medium.viscosity)
                bond_factor, eta_base = _apply_shatter_to_feature(
                    global_cycle_1based=cycle,
                    medium=medium,
                    col=col,
                    bond_factor=bond_factor,
                    eta_base=eta_base,
                )
                eta_dynamic = max(1e-6, eta_base / (1.0 + shear_eff * abs(charge) * bond_factor))

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
                if vib_enabled:
                    eta_dynamic = max(
                        1e-6,
                        eta_dynamic
                        * _vibrational_viscosity_multiplier(
                            cycle,
                            enabled=vib_enabled,
                            period=vib_period,
                            amplitude=vib_amp,
                            waveform=vib_wave,
                            phase=2.0 * math.pi * float(x_pos),
                        ),
                    )
                v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                denom += abs(v)
                amp = float(pcr_amp_by_feature.get(col, 1.0))
                a[i_w] = float(amp * v)

            if field_Z is not None and field_C is not None and _field_active(cycle):
                a_eff_alpha = _field_alpha_at(cycle)
                if a_eff_alpha > 0.0:
                    try:
                        base_score = field_Z @ a
                        coupled_score = field_Z @ (field_C @ a)
                        if mb_enabled:
                            alpha_vec = float(a_eff_alpha) * np.asarray(mb_alpha_vec, dtype="float64")
                            update_score = base_score + (alpha_vec * coupled_score)
                        else:
                            update_score = base_score + float(a_eff_alpha) * coupled_score
                    except Exception:
                        update_score = np.zeros(df.shape[0], dtype="float64")
                        for i_w, wi in enumerate(weights_used):
                            update_score += float(a[i_w]) * z_by_feature[wi.feature]
                else:
                    for i_w, wi in enumerate(weights_used):
                        update_score += float(a[i_w]) * z_by_feature[wi.feature]
            else:
                for i_w, wi in enumerate(weights_used):
                    update_score += float(a[i_w]) * z_by_feature[wi.feature]

            if denom <= 1e-12:
                denom = 1.0
            pred = pred + (lr * residual_std / denom) * (update_score / eta_scale_vec)

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
                gcycle = int(n_cycles_eff + stage_cycle)
                lr2 = float(_lr_effective(gcycle, total_schedule_cycles)) * 0.6
                shear_eff = float(_shear_at(gcycle, total_schedule_cycles))
                residual = y - pred
                residual_train = residual[train_mask]
                residual_std = float(np.nanstd(residual_train))
                if not math.isfinite(residual_std) or residual_std <= 1e-12:
                    residual_std = 1.0
                eta_scale = _target_induced_viscosity_multiplier(residual_train, residual_std)

                mb_visc_vec, mb_alpha_vec = _multibuffer_multipliers(pred)
                eta_scale_vec = np.asarray(mb_visc_vec, dtype="float64") * float(eta_scale)
                eta_scale_vec = np.clip(eta_scale_vec, 1e-6, 1e9)

                update_score = np.zeros(df.shape[0], dtype="float64")
                denom = 0.0

                # Stage-2 Field-Effect coupling among zone features.
                zone_cols = sorted(zone1_features)
                zone_a = np.zeros(len(zone_cols), dtype="float64")

                for i_c, col in enumerate(zone_cols):
                    z = z_by_feature[col]
                    charge = _pearson_corr(z[train_mask], residual_train)
                    if not math.isfinite(charge) or abs(charge) < 1e-8:
                        continue

                    medium = migration_by_feature[col]
                    bond_factor = bond_factors.get(col, 1.0)
                    certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                    density = (1.0 + max(0.0, medium.kl_divergence)) * certainty
                    eta_base = max(1e-6, medium.viscosity)
                    bond_factor, eta_base = _apply_shatter_to_feature(
                        global_cycle_1based=gcycle,
                        medium=medium,
                        col=col,
                        bond_factor=bond_factor,
                        eta_base=eta_base,
                    )
                    eta_dynamic = max(1e-6, eta_base / (1.0 + shear_eff * abs(charge) * bond_factor))

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
                    if vib_enabled:
                        eta_dynamic = max(
                            1e-6,
                            eta_dynamic
                            * _vibrational_viscosity_multiplier(
                                n_cycles_eff + stage_cycle,
                                enabled=vib_enabled,
                                period=vib_period,
                                amplitude=vib_amp,
                                waveform=vib_wave,
                                phase=2.0 * math.pi * float(x_pos),
                            ),
                        )
                    v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                    denom += abs(v)
                    amp = float(pcr_amp_by_feature.get(col, 1.0))
                    zone_a[i_c] = float(amp * v)

                if field_Z is not None and field_C is not None and _field_active(gcycle) and len(zone_cols) >= 2:
                    try:
                        idx = [field_pos[c] for c in zone_cols]
                        Zsub = field_Z[:, idx]
                        Csub = field_C[np.ix_(idx, idx)]
                        a_eff_alpha = _field_alpha_at(gcycle)
                        base_score = Zsub @ zone_a
                        coupled_score = Zsub @ (Csub @ zone_a)
                        if mb_enabled:
                            alpha_vec = float(a_eff_alpha) * np.asarray(mb_alpha_vec, dtype="float64")
                            update_score = base_score + (alpha_vec * coupled_score)
                        else:
                            update_score = base_score + float(a_eff_alpha) * coupled_score
                    except Exception:
                        update_score = np.zeros(df.shape[0], dtype="float64")
                        for i_c, col in enumerate(zone_cols):
                            update_score += float(zone_a[i_c]) * z_by_feature[col]
                else:
                    for i_c, col in enumerate(zone_cols):
                        update_score += float(zone_a[i_c]) * z_by_feature[col]

                if denom <= 1e-12:
                    denom = 1.0
                pred = pred + (lr2 * residual_std / denom) * (update_score / eta_scale_vec)

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
                        zone_id=200 + k,
                        features=feats_list,
                        avg_pI=float(np.mean([pI_map.get(f, 0.5) for f in feats_list])),
                        avg_momentum=0.0,
                        strength=float(len(feats_list) / max(1, len(zone1_features))),
                    )
                )

            # Scavenger pass: recycle the weakest third of Zone 1.
            waste_features = sorted(zone1_features, key=lambda f: abs(float(weights_by_feature[f].weight)))
            waste_features = waste_features[: max(1, len(waste_features) // 3)]
            for f in sorted(unstable_features):
                if f in zone1_features and f not in waste_features:
                    waste_features.append(f)
            if int(scavenger_cycles) > 0 and waste_features:
                rng_scav = np.random.default_rng(int(random_seed) + 51017)
                for sc_cycle in range(1, int(scavenger_cycles) + 1):
                    gcycle = int(n_cycles_eff + max(1, int(stage2_cycles)) + sc_cycle)
                    lr3 = float(_lr_effective(gcycle, total_schedule_cycles)) * 0.25
                    shear_eff = float(_shear_at(gcycle, total_schedule_cycles))
                    residual = y - pred
                    residual_train = residual[train_mask]
                    residual_std = float(np.nanstd(residual_train))
                    if not math.isfinite(residual_std) or residual_std <= 1e-12:
                        residual_std = 1.0

                    eta_scale = _target_induced_viscosity_multiplier(residual_train, residual_std)
                    mb_visc_vec, mb_alpha_vec = _multibuffer_multipliers(pred)
                    eta_scale_vec = np.asarray(mb_visc_vec, dtype="float64") * float(eta_scale)
                    eta_scale_vec = np.clip(eta_scale_vec, 1e-6, 1e9)

                    update_score = np.zeros(df.shape[0], dtype="float64")
                    denom = 0.0

                    waste_cols = list(waste_features)
                    waste_a = np.zeros(len(waste_cols), dtype="float64")
                    for i_c, col in enumerate(waste_cols):
                        z = z_by_feature[col]
                        charge = _pearson_corr(z[train_mask], residual_train)
                        if not math.isfinite(charge) or abs(charge) < 1e-8:
                            continue

                        medium = migration_by_feature[col]
                        bond_factor = bond_factors.get(col, 1.0)
                        certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                        density = (1.0 + max(0.0, medium.kl_divergence)) * certainty
                        eta_base = max(1e-6, medium.viscosity)
                        bond_factor, eta_base = _apply_shatter_to_feature(
                            global_cycle_1based=gcycle,
                            medium=medium,
                            col=col,
                            bond_factor=bond_factor,
                            eta_base=eta_base,
                        )
                        eta_dynamic = max(1e-6, eta_base / (1.0 + shear_eff * abs(charge) * bond_factor))
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
                        if vib_enabled:
                            eta_dynamic = max(
                                1e-6,
                                eta_dynamic
                                * _vibrational_viscosity_multiplier(
                                    n_cycles_eff + max(1, int(stage2_cycles)) + sc_cycle,
                                    enabled=vib_enabled,
                                    period=vib_period,
                                    amplitude=vib_amp,
                                    waveform=vib_wave,
                                    phase=2.0 * math.pi * float(x_pos),
                                ),
                            )
                        v = q * field / eta_dynamic

                        denom += abs(v)
                        amp = float(pcr_amp_by_feature.get(col, 1.0))
                        waste_a[i_c] = float(amp * v)

                    if field_Z is not None and field_C is not None and _field_active(gcycle) and len(waste_cols) >= 2:
                        try:
                            idx = [field_pos[c] for c in waste_cols]
                            Zsub = field_Z[:, idx]
                            Csub = field_C[np.ix_(idx, idx)]
                            a_eff_alpha = _field_alpha_at(gcycle)
                            base_score = Zsub @ waste_a
                            coupled_score = Zsub @ (Csub @ waste_a)
                            if mb_enabled:
                                alpha_vec = float(a_eff_alpha) * np.asarray(mb_alpha_vec, dtype="float64")
                                update_score = base_score + (alpha_vec * coupled_score)
                            else:
                                update_score = base_score + float(a_eff_alpha) * coupled_score
                        except Exception:
                            # fallback: already accumulated or can recompute
                            if not np.any(update_score):
                                for i_c, col in enumerate(waste_cols):
                                    update_score += float(waste_a[i_c]) * z_by_feature[col]
                    else:
                        if not np.any(update_score):
                            for i_c, col in enumerate(waste_cols):
                                update_score += float(waste_a[i_c]) * z_by_feature[col]

                    if denom <= 1e-12:
                        denom = 1.0
                    pred = pred + (lr3 * residual_std / denom) * (update_score / eta_scale_vec)

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

        test_row_indices = None
        test_actual = None
        test_predicted = None
        if return_predictions:
            idx_list = [int(i) for i in test_indices.tolist()]
            test_row_indices = idx_list
            if target_kind == "datetime":
                test_actual = [
                    None if not math.isfinite(float(y[i])) else pd.to_datetime(int(y[i])).isoformat() for i in idx_list
                ]
                test_predicted = [
                    None if not math.isfinite(float(pred[i])) else pd.to_datetime(int(pred[i])).isoformat() for i in idx_list
                ]
            else:
                test_actual = [None if not math.isfinite(float(y[i])) else float(y[i]) for i in idx_list]
                test_predicted = [None if not math.isfinite(float(pred[i])) else float(pred[i]) for i in idx_list]

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
            buffer_ionization=buffer_ionization,
            buffer_normality_p=buffer_normality_p,
            gel_band_sharpness=_gel_health_regression(y[test_mask], pred[test_mask])[0],
            gel_smearing=_gel_health_regression(y[test_mask], pred[test_mask])[1],
            gel_ghost_band_rate=_gel_health_regression(y[test_mask], pred[test_mask])[2],
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

        equilibrium_zones.extend(
            _fractionate_kw_zones(
                df,
                feature_cols_used=feature_cols_used,
                feature_kinds=feature_kinds,
                pI_map=pI_map,
                target_series=target_series,
                target_kind=target_kind,
                train_mask=train_mask,
                start_zone_id=100,
            )
        )

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
            diagnostics=(
                None
                if (cleaning_diag is None and multibuffer_diag is None)
                else {
                    **({} if cleaning_diag is None else {"cleaning": cleaning_diag}),
                    **({} if multibuffer_diag is None else {"multibuffer": multibuffer_diag}),
                }
            ),
            test_row_indices=test_row_indices,
            test_actual=test_actual,
            test_predicted=test_predicted,
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

    # Keep PCR amps only for the used features.
    if pcr_amp_by_feature:
        pcr_amp_by_feature = {k: v for k, v in pcr_amp_by_feature.items() if k in set(feature_cols_used)}

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
    grad1 = 0.35
    rng_stage1 = np.random.default_rng(int(random_seed) + 30011)
    rng_primary_sieve = np.random.default_rng(int(random_seed) + 50123)

    primary_sieve_enabled = bool(low_confidence_primary_sieve_enabled)
    primary_sieve_cycle_a = int(low_confidence_primary_sieve_cycle_a)
    primary_sieve_cycle_b = int(low_confidence_primary_sieve_cycle_b)
    primary_sieve_shake_cycles = int(max(0, low_confidence_primary_sieve_shake_cycles))
    primary_sieve_reverse = float(max(0.0, low_confidence_primary_sieve_reverse_multiplier))
    primary_sieve_noise = float(max(0.0, low_confidence_primary_sieve_noise_std))
    primary_sieve_inst_min = float(np.clip(float(low_confidence_primary_sieve_instability_min), 0.0, 1.0))
    primary_sieve_conf_delta_max = float(max(0.0, low_confidence_primary_sieve_conf_delta_max))

    es_patience = int(max(0, early_stop_patience))
    es_tol = float(max(0.0, early_stop_tol))
    es_prev_conf: np.ndarray | None = None
    es_stable_steps: int = 0

    # Instability (PCR "smear") tracking for categorical targets.
    # We measure how much per-row confidence jitters and how often argmax flips across cycles.
    inst_prev_conf: np.ndarray | None = None
    inst_prev_pred_idx: np.ndarray | None = None
    inst_jitter_sum = np.zeros(df.shape[0], dtype="float64")
    inst_flip_count = np.zeros(df.shape[0], dtype="int32")
    inst_steps: int = 0

    # Optional: delay the Stage-2 focusing/shattering until a specific global cycle.
    # This allows the stage-1 gel to settle before complex dissociation.
    stage2_trigger = int(stage2_trigger_cycle)
    if stage2_trigger <= 0:
        stage2_trigger = 0
    if stage2_trigger and int(n_cycles_eff) >= stage2_trigger:
        n_cycles_eff = max(1, stage2_trigger - 1)

    total_schedule_cycles = int(n_cycles_eff)
    if cascade_enabled and len(zone1_features) >= 3:
        total_schedule_cycles += int(max(1, stage2_cycles))
        if int(scavenger_cycles) > 0:
            total_schedule_cycles += int(scavenger_cycles)
    total_schedule_cycles = int(max(1, total_schedule_cycles))

    for cycle in range(1, n_cycles_eff + 1):
        lr = float(_lr_at(cycle, total_schedule_cycles))
        shear = max(0.0, float(_shear_at(cycle, total_schedule_cycles)))
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
                    tmp = pd.DataFrame({"x": x_cat[train_mask].to_numpy(dtype=object), "r": residual_train})
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
                eta_base = max(1e-6, medium.viscosity)
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
                if vib_enabled:
                    eta_dynamic = max(
                        1e-6,
                        eta_dynamic
                        * _vibrational_viscosity_multiplier(
                            cycle,
                            enabled=vib_enabled,
                            period=vib_period,
                            amplitude=vib_amp,
                            waveform=vib_wave,
                            phase=2.0 * math.pi * float(x_pos),
                        ),
                    )
                v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                denom += abs(v)
                amp = float(pcr_amp_by_feature.get(col, 1.0))
                class_score += (amp * v) * z

            if denom <= 1e-12:
                denom = 1.0
            cycle_update[:, j] = class_score / denom

        logits += lr * cycle_update
        probs = _softmax(logits)

        conf_now = np.max(probs, axis=1)
        pred_now = np.argmax(probs, axis=1)

        # v4.3 scheduled micro-shakes: apply two small pulses at cycle A/B to tangled rows.
        if (
            primary_sieve_enabled
            and primary_sieve_shake_cycles > 0
            and (cycle == primary_sieve_cycle_a or cycle == primary_sieve_cycle_b)
            and inst_prev_conf is not None
            and inst_prev_pred_idx is not None
        ):
            try:
                # Running instability estimate based on observed jitter/flip stats so far.
                if inst_steps > 0:
                    conf_jitter = inst_jitter_sum / float(inst_steps)
                    flip_rate = inst_flip_count.astype("float64") / float(inst_steps)
                    inst_now = np.clip(0.60 * flip_rate + 0.40 * conf_jitter, 0.0, 1.0)
                else:
                    inst_now = np.zeros(df.shape[0], dtype="float64")

                conf_delta = np.asarray(conf_now, dtype="float64") - np.asarray(inst_prev_conf, dtype="float64")
                tangled = (
                    (inst_now >= primary_sieve_inst_min)
                    & (conf_delta <= primary_sieve_conf_delta_max)
                )
                if bool(np.any(tangled)):
                    for _ in range(primary_sieve_shake_cycles):
                        if primary_sieve_reverse > 0.0:
                            logits[tangled, :] += float(lr) * (-primary_sieve_reverse) * cycle_update[tangled, :]
                        if primary_sieve_noise > 0.0:
                            logits[tangled, :] += rng_primary_sieve.normal(
                                loc=0.0,
                                scale=primary_sieve_noise,
                                size=(int(np.sum(tangled.astype("int32"))), logits.shape[1]),
                            )
                    probs = _softmax(logits)
                    conf_now = np.max(probs, axis=1)
                    pred_now = np.argmax(probs, axis=1)
            except Exception:
                pass

        if inst_prev_conf is not None and inst_prev_pred_idx is not None:
            inst_jitter_sum += np.abs(conf_now - inst_prev_conf)
            inst_flip_count += (pred_now != inst_prev_pred_idx).astype("int32")
            inst_steps += 1
        inst_prev_conf = conf_now
        inst_prev_pred_idx = pred_now

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

        if es_patience > 0:
            if es_prev_conf is None:
                es_prev_conf = conf_now.copy()
                es_stable_steps = 0
            else:
                delta = float(np.mean(np.abs(conf_now - es_prev_conf)))
                es_prev_conf = conf_now.copy()
                if delta <= es_tol:
                    es_stable_steps += 1
                else:
                    es_stable_steps = 0
                # Require a few warmup cycles to avoid stopping too early.
                if cycle >= 8 and es_stable_steps >= es_patience:
                    break

    # Stage 2 (Fractional Distillation): automatically re-focus the strongest cluster (Zone 1)
    # with higher field strength + narrower gradient, plus inhibition + thermal noise.
    if cascade_enabled and len(zone1_features) >= 3:
        # Narrowed gradient: use variance within the cluster relative to global feature variance.
        global_var = float(np.mean([m.variance for m in migration_map])) if migration_map else 1.0
        cluster_var = float(np.mean([migration_by_feature[f].variance for f in zone1_features]))
        grad = float(np.clip(0.55 * (cluster_var / (global_var + 1e-9)), 0.05, 0.9))
        E = float(stage2_voltage_multiplier) * float(plane_mobility)

        rng = np.random.default_rng(int(random_seed) + 40007)

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
            gcycle = int(n_cycles_eff + stage_cycle)
            lr2 = float(_lr_at(gcycle, total_schedule_cycles)) * 0.75
            shear = max(0.0, float(_shear_at(gcycle, total_schedule_cycles)))
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
                        tmp = pd.DataFrame({"x": x_cat[train_mask].to_numpy(dtype=object), "r": residual_train})
                        rates = tmp.groupby("x")["r"].mean()
                        x_raw = x_cat.map(rates).fillna(0.0).to_numpy(dtype="float64")

                    z = _zscore_with_train_stats(x_raw, train_mask)
                    charge = _pearson_corr(z[train_mask], residual_train)
                    if not math.isfinite(charge) or abs(charge) < 1e-8:
                        continue

                    medium = migration_by_feature[col]
                    bond_factor = bond_factors.get(col, 1.0)
                    if bool(stage2_shatter_complexes) and medium.complex_size is not None and int(medium.complex_size) >= 2:
                        bond_factor = 1.0
                    certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                    density = (1.0 + max(0.0, medium.kl_divergence)) * certainty

                    eta_base = max(1e-6, medium.viscosity)
                    if bool(stage2_shatter_complexes) and medium.complex_size is not None and int(medium.complex_size) >= 2:
                        # Undo complex drag in Stage-2 so individual members can migrate independently.
                        eta_base = max(1e-6, eta_base / float(complex_drag_by_feature.get(col, 1.0)))
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
                    if vib_enabled:
                        eta_dynamic = max(
                            1e-6,
                            eta_dynamic
                            * _vibrational_viscosity_multiplier(
                                n_cycles_eff + stage_cycle,
                                enabled=vib_enabled,
                                period=vib_period,
                                amplitude=vib_amp,
                                waveform=vib_wave,
                                phase=2.0 * math.pi * float(x_pos),
                            ),
                        )
                    v = (q * field + thermal_term) / (eta_dynamic + inhibition)

                    denom += abs(v)
                    amp = float(pcr_amp_by_feature.get(col, 1.0))
                    class_score += (amp * v) * z

                if denom <= 1e-12:
                    denom = 1.0
                cycle_update[:, j] = class_score / denom

            logits += lr2 * cycle_update
            probs = _softmax(logits)

            conf_now = np.max(probs, axis=1)
            pred_now = np.argmax(probs, axis=1)
            if inst_prev_conf is not None and inst_prev_pred_idx is not None:
                inst_jitter_sum += np.abs(conf_now - inst_prev_conf)
                inst_flip_count += (pred_now != inst_prev_pred_idx).astype("int32")
                inst_steps += 1
            inst_prev_conf = conf_now
            inst_prev_pred_idx = pred_now

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
                    zone_id=200 + k,
                    features=feats_list,
                    avg_pI=float(np.mean([pI_map.get(f, 0.5) for f in feats_list])),
                    avg_momentum=float(0.0),
                    strength=float(len(feats_list) / max(1, len(zone1_features))),
                )
            )

        # Residual recycling: send the "waste" (lowest-signal third) back for a scavenger pass.
        waste_features = sorted(zone1_features, key=lambda f: abs(float(feature_info[f].weight)))
        waste_features = waste_features[: max(1, len(waste_features) // 3)]
        for f in sorted(unstable_features):
            if f in zone1_features and f not in waste_features:
                waste_features.append(f)
        if scavenger_cycles > 0 and waste_features:
            rng2 = np.random.default_rng(int(random_seed) + 50021)
            for sc_cycle in range(1, int(scavenger_cycles) + 1):
                gcycle = int(n_cycles_eff + n2 + sc_cycle)
                lr3 = float(_lr_at(gcycle, total_schedule_cycles)) * 0.25
                shear = max(0.0, float(_shear_at(gcycle, total_schedule_cycles)))
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
                            tmp = pd.DataFrame({"x": x_cat[train_mask].to_numpy(dtype=object), "r": residual_train})
                            rates = tmp.groupby("x")["r"].mean()
                            x_raw = x_cat.map(rates).fillna(0.0).to_numpy(dtype="float64")

                        z = _zscore_with_train_stats(x_raw, train_mask)
                        charge = _pearson_corr(z[train_mask], residual_train)
                        if not math.isfinite(charge) or abs(charge) < 1e-8:
                            continue

                        medium = migration_by_feature[col]
                        bond_factor = bond_factors.get(col, 1.0)
                        if bool(stage2_shatter_complexes) and medium.complex_size is not None and int(medium.complex_size) >= 2:
                            bond_factor = 1.0
                        certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                        density = (1.0 + max(0.0, medium.kl_divergence)) * certainty

                        eta_base = max(1e-6, medium.viscosity)
                        if bool(stage2_shatter_complexes) and medium.complex_size is not None and int(medium.complex_size) >= 2:
                            eta_base = max(1e-6, eta_base / float(complex_drag_by_feature.get(col, 1.0)))
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
                        if vib_enabled:
                            eta_dynamic = max(
                                1e-6,
                                eta_dynamic
                                * _vibrational_viscosity_multiplier(
                                    n_cycles_eff + n2 + sc_cycle,
                                    enabled=vib_enabled,
                                    period=vib_period,
                                    amplitude=vib_amp,
                                    waveform=vib_wave,
                                    phase=2.0 * math.pi * float(x_pos),
                                ),
                            )
                        v = (q * field) / eta_dynamic

                        denom += abs(v)
                        amp = float(pcr_amp_by_feature.get(col, 1.0))
                        class_score += (amp * v) * z

                    if denom <= 1e-12:
                        denom = 1.0
                    cycle_update[:, j] = class_score / denom

                logits += lr3 * cycle_update
                probs = _softmax(logits)

                conf_now = np.max(probs, axis=1)
                pred_now = np.argmax(probs, axis=1)
                if inst_prev_conf is not None and inst_prev_pred_idx is not None:
                    inst_jitter_sum += np.abs(conf_now - inst_prev_conf)
                    inst_flip_count += (pred_now != inst_prev_pred_idx).astype("int32")
                    inst_steps += 1
                inst_prev_conf = conf_now
                inst_prev_pred_idx = pred_now

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

    # PCR-style per-row band density analysis: optionally flag or abstain when the gel smears.
    mode = str(low_confidence_mode).lower().strip()
    conf = np.max(probs, axis=1)
    entropy = _normalized_entropy(probs)

    instability = np.zeros(df.shape[0], dtype="float64")
    if inst_steps > 0:
        conf_jitter = inst_jitter_sum / float(inst_steps)
        flip_rate = inst_flip_count.astype("float64") / float(inst_steps)
        instability = np.clip(0.60 * flip_rate + 0.40 * conf_jitter, 0.0, 1.0)

    # Ionization gate (p-value-driven): compute a per-row ionization signal and a viscosity path.
    # These are only needed for selective gating when enabled.
    need_ion_gate = bool(low_confidence_require_ionized)
    need_visc_override = bool(low_confidence_viscosity_override)
    need_confirmatory = bool(low_confidence_confirmatory_enabled)
    need_secondary = bool(low_confidence_secondary_enabled) and int(low_confidence_secondary_cycles) > 0
    need_row_diagnostics = mode in ("flag", "abstain") and (
        need_ion_gate or need_visc_override or need_confirmatory or need_secondary
    )

    ionized_hit_count = np.zeros(df.shape[0], dtype="int32")
    ionization_mass_row = np.zeros(df.shape[0], dtype="float64")
    viscosity_path = np.full(df.shape[0], float("nan"), dtype="float64")
    confirmatory_consensus = np.zeros(df.shape[0], dtype="float64")
    zone_cluster_max_votes = np.zeros(df.shape[0], dtype="int32")
    z_cache: dict[str, np.ndarray] = {}
    ionized_features: list[str] = []
    if need_row_diagnostics:
        try:
            alpha_p = float(low_confidence_ionization_pvalue)
            z_min = float(max(0.0, low_confidence_ionization_z_min))
            class_to_idx = {str(c): int(i) for i, c in enumerate(classes)}
            y_idx = np.array([float(class_to_idx.get(str(v), -1)) for v in y_cat.to_numpy(dtype="object")], dtype="float64")
            y_idx_train_mask = (y_idx >= 0.0) & train_mask
            y_idx_mean = float(np.mean(y_idx[y_idx_train_mask])) if bool(np.any(y_idx_train_mask)) else 0.0

            if need_ion_gate:
                for col in feature_cols_used:
                    m = migration_by_feature.get(col)
                    if m is None or m.p_value is None:
                        continue
                    if float(m.p_value) <= alpha_p:
                        ionized_features.append(col)

            # Build a single consistent encoding per feature to assess row-level signals.
            for col in feature_cols_used:
                fk = feature_info[col].feature_kind
                if fk in ("numeric", "datetime", "bool"):
                    x_raw = x_encoded_by_feature[col]
                else:
                    x_cat = df[col].astype("string").fillna("__MISSING__")
                    tmp = pd.DataFrame({"x": x_cat[y_idx_train_mask], "y": y_idx[y_idx_train_mask]})
                    means = tmp.groupby("x")["y"].mean()
                    x_raw = x_cat.map(means).fillna(y_idx_mean).to_numpy(dtype="float64")
                z_cache[col] = _zscore_with_train_stats(x_raw, train_mask)

            if need_visc_override:
                visc_num = np.zeros(df.shape[0], dtype="float64")
                visc_den = np.zeros(df.shape[0], dtype="float64")
                for col in feature_cols_used:
                    z = z_cache[col]
                    absz = np.abs(z)
                    visc = float(migration_by_feature[col].viscosity)
                    visc_num += absz * visc
                    visc_den += absz
                viscosity_path = np.where(visc_den > 1e-12, visc_num / visc_den, float(_plane_base_viscosity(plane)))

            if ionized_features:
                for col in ionized_features:
                    z = z_cache[col]
                    absz = np.abs(z)
                    m = migration_by_feature[col]
                    mass_norm = float(np.clip(float(m.mass) / 6.0, 0.0, 1.0))
                    ionization_mass_row += mass_norm * absz
                    ionized_hit_count += (absz >= z_min).astype("int32")

            # Confirmatory band consensus: if most of the row's signal (|z| weighted by mass)
            # concentrates into a single zone, it's a "confirmatory" band.
            if need_confirmatory and z_cache:
                zone_to_weight: dict[int, np.ndarray] = {}
                total = np.zeros(df.shape[0], dtype="float64")
                for col in feature_cols_used:
                    z = z_cache.get(col)
                    if z is None:
                        continue
                    absz = np.abs(z)
                    m = migration_by_feature[col]
                    mass_norm = float(np.clip(float(m.mass) / 6.0, 0.0, 1.0))
                    w = absz * mass_norm
                    zid = int(zone_assignment_stage1.get(col, 0))
                    if zid not in zone_to_weight:
                        zone_to_weight[zid] = np.zeros(df.shape[0], dtype="float64")
                    zone_to_weight[zid] += w
                    total += w
                if zone_to_weight:
                    best = np.zeros(df.shape[0], dtype="float64")
                    for arr in zone_to_weight.values():
                        best = np.maximum(best, arr)
                    confirmatory_consensus = np.where(total > 1e-12, best / total, 0.0)

            # Cluster voting (fractional voting): count how many features per row land in each zone
            # with |z| >= z_min and promote rows with a tight cluster even if individual z are low.
            if need_secondary and z_cache:
                z_prom = float(max(0.0, low_confidence_secondary_promote_z_min))
                zone_to_votes: dict[int, np.ndarray] = {}
                for col in feature_cols_used:
                    z = z_cache.get(col)
                    if z is None:
                        continue
                    votes = (np.abs(z) >= z_prom).astype("int32")
                    zid = int(zone_assignment_stage1.get(col, 0))
                    if zid not in zone_to_votes:
                        zone_to_votes[zid] = np.zeros(df.shape[0], dtype="int32")
                    zone_to_votes[zid] += votes
                if zone_to_votes:
                    best_votes = np.zeros(df.shape[0], dtype="int32")
                    for arr in zone_to_votes.values():
                        best_votes = np.maximum(best_votes, arr)
                    zone_cluster_max_votes = best_votes
        except Exception:
            pass

    smear_metric_name = str(low_confidence_smear_metric or "entropy").lower().strip()
    if smear_metric_name == "instability":
        smear = instability
    else:
        smear_metric_name = "entropy"
        smear = entropy

    combine_rule = str(low_confidence_combine_rule or "or").lower().strip()
    if combine_rule not in ("or", "and"):
        combine_rule = "or"

    # Selective diagnostics: we compute both stage-by-stage abstain rates and a final reason breakdown.
    selective_diagnostics: dict[str, Any] | None = None
    mask_pre_reion: np.ndarray | None = None
    mask_post_reion: np.ndarray | None = None

    # Auto-calibration: if thresholds are <= 0, derive them from the TEST distribution.
    conf_thr = float(low_confidence_threshold)
    ent_thr = float(low_confidence_entropy_threshold)
    if mode in ("flag", "abstain"):
        conf_test = conf[test_mask]
        smear_test = smear[test_mask]
        q_conf = float(np.clip(float(low_confidence_auto_conf_quantile), 0.0, 1.0))
        q_smear = float(np.clip(float(low_confidence_auto_smear_quantile), 0.0, 1.0))
        if conf_thr <= 0.0:
            # Treat the weakest q_conf confidence as "low" by default.
            conf_thr = float(np.quantile(conf_test, q_conf)) if conf_test.size else 0.0
        if ent_thr <= 0.0:
            # Treat the noisiest (1 - q_smear) smear tail as "smeared" by default.
            ent_thr = float(np.quantile(smear_test, q_smear)) if smear_test.size else 1.0

        # Safeguard: if thresholds would abstain almost everything, relax them.
        if combine_rule == "and":
            prelim = (conf < conf_thr) & (smear > ent_thr)
        else:
            prelim = (conf < conf_thr) | (smear > ent_thr)
        prelim_rate = float(np.mean(prelim[test_mask].astype("float64"))) if conf_test.size else 0.0
        max_abstain = float(np.clip(float(low_confidence_safeguard_max_abstain), 0.0, 1.0))
        if prelim_rate >= max_abstain and conf_test.size and max_abstain < 1.0:
            conf_thr = float(np.quantile(conf_test, 0.05))
            ent_thr = float(np.quantile(smear_test, 0.95))

    if combine_rule == "and":
        low_conf_mask = (conf < float(conf_thr)) & (smear > float(ent_thr))
    else:
        low_conf_mask = (conf < float(conf_thr)) | (smear > float(ent_thr))

    # Inverse instability logic: if instability is the smear metric, allow low-viscosity paths
    # to override smear-based rejection (these are often complex-but-correct interactions).
    if mode in ("flag", "abstain") and smear_metric_name == "instability" and bool(low_confidence_viscosity_override):
        try:
            visc_thr = float(low_confidence_viscosity_override_threshold)
            if math.isfinite(visc_thr):
                override = np.isfinite(viscosity_path) & (viscosity_path <= visc_thr)
                low_conf_mask = np.asarray(low_conf_mask, dtype=bool) & (~override)
        except Exception:
            pass

    # Chi-square / p-value ionization gate: rows with zero "ionized" feature hits are discarded first.
    if mode in ("flag", "abstain") and bool(low_confidence_require_ionized):
        try:
            low_conf_mask = np.asarray(low_conf_mask, dtype=bool) | (ionized_hit_count <= 0)
        except Exception:
            pass

    # Coverage expansion: confirmatory-band override.
    # If the row's signal concentrates in one zone (high consensus) and the confidence is in a mid-range,
    # keep it even if it was marked low-confidence.
    if mode in ("flag", "abstain") and bool(low_confidence_confirmatory_enabled):
        try:
            cmin = float(low_confidence_confirmatory_conf_min)
            cmax = float(low_confidence_confirmatory_conf_max)
            if cmax < cmin:
                cmin, cmax = cmax, cmin
            cons_thr = float(np.clip(float(low_confidence_confirmatory_consensus_threshold), 0.0, 1.0))
            min_hits = int(max(0, low_confidence_confirmatory_min_ion_hits))
            mid = (conf >= cmin) & (conf <= cmax)
            consensus_ok = confirmatory_consensus >= cons_thr
            if min_hits > 0:
                consensus_ok = consensus_ok & (ionized_hit_count >= min_hits)
            keep_confirmatory = mid & consensus_ok
            low_conf_mask = np.asarray(low_conf_mask, dtype=bool) & (~keep_confirmatory)
        except Exception:
            pass

    if mode in ("flag", "abstain"):
        # Snapshot right before re-ionization/secondary passes.
        try:
            mask_pre_reion = np.asarray(low_conf_mask, dtype=bool).copy()
        except Exception:
            mask_pre_reion = None

    # Target re-ionization: sub-cycles that only update logits for low-confidence rows.
    # This can recover some abstained rows without disturbing already-confident ones.
    rein_cycles = int(max(0, low_confidence_reionization_cycles))
    if mode in ("flag", "abstain") and rein_cycles > 0:
        try:
            low_mask = np.asarray(low_conf_mask, dtype=bool)
            if bool(np.any(low_mask)):
                shear_re = float(max(0.0, shear * float(low_confidence_reionization_shear_multiplier)))
                inhib_re = float(max(0.0, float(inhibition_strength) * float(low_confidence_reionization_inhibition_multiplier)))
                # Reuse stage-1 style update, but apply only on low_mask rows.
                for _ in range(rein_cycles):
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
                                tmp = pd.DataFrame({"x": x_cat[train_mask].to_numpy(dtype=object), "r": residual_train})
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
                            eta_base = max(1e-6, medium.viscosity)
                            eta_dynamic = max(1e-6, eta_base / (1.0 + shear_re * abs(charge) * bond_factor))

                            inhibition = 0.0
                            if competitive_inhibition and col != dominant_global:
                                inhibition = inhib_re * abs(float(affinity_to_global_dominant.get(col, 0.0)))

                            eff_charge = float(charge)
                            if eff_charge < 0:
                                eff_charge *= neg_mult
                            q = eff_charge * density * bond_factor
                            x_pos = float(pI_map.get(col, 0.5))
                            field = float(plane_mobility) - float(grad1) * x_pos
                            v = (q * field) / (eta_dynamic + inhibition)

                            denom += abs(v)
                            amp = float(pcr_amp_by_feature.get(col, 1.0))
                            class_score += (amp * v) * z

                        if denom <= 1e-12:
                            denom = 1.0
                        cycle_update[:, j] = class_score / denom

                    # Apply updates only to low-confidence rows.
                    logits[low_mask, :] += (float(lr) * cycle_update[low_mask, :])
                    probs = _softmax(logits)
                    conf = np.max(probs, axis=1)
                    entropy = _normalized_entropy(probs)

                    # Recompute smear and low_mask using current thresholds.
                    if smear_metric_name == "instability":
                        smear = instability
                    else:
                        smear = entropy

                    if combine_rule == "and":
                        low_mask = (conf < float(conf_thr)) & (smear > float(ent_thr))
                    else:
                        low_mask = (conf < float(conf_thr)) | (smear > float(ent_thr))

                    if bool(low_confidence_require_ionized):
                        low_mask = np.asarray(low_mask, dtype=bool) | (ionized_hit_count <= 0)

                    if bool(low_confidence_confirmatory_enabled):
                        cmin = float(low_confidence_confirmatory_conf_min)
                        cmax = float(low_confidence_confirmatory_conf_max)
                        if cmax < cmin:
                            cmin, cmax = cmax, cmin
                        cons_thr = float(np.clip(float(low_confidence_confirmatory_consensus_threshold), 0.0, 1.0))
                        min_hits = int(max(0, low_confidence_confirmatory_min_ion_hits))
                        mid = (conf >= cmin) & (conf <= cmax)
                        consensus_ok = confirmatory_consensus >= cons_thr
                        if min_hits > 0:
                            consensus_ok = consensus_ok & (ionized_hit_count >= min_hits)
                        keep_confirmatory = mid & consensus_ok
                        low_mask = np.asarray(low_mask, dtype=bool) & (~keep_confirmatory)

                    if bool(low_confidence_viscosity_override) and smear_metric_name == "instability":
                        visc_thr = float(low_confidence_viscosity_override_threshold)
                        if math.isfinite(visc_thr):
                            override = np.isfinite(viscosity_path) & (viscosity_path <= visc_thr)
                            low_mask = np.asarray(low_mask, dtype=bool) & (~override)

                    if not bool(np.any(low_mask)):
                        break

                # Refresh final predictions after re-ionization.
                pred_idx = np.argmax(probs, axis=1)
                pred_labels = [classes[int(i)] for i in pred_idx]

                # Persist final low_conf_mask after re-ionization.
                low_conf_mask = np.asarray(low_mask, dtype=bool)
        except Exception:
            pass

    if mode in ("flag", "abstain"):
        try:
            mask_post_reion = np.asarray(low_conf_mask, dtype=bool).copy()
        except Exception:
            mask_post_reion = None

    secondary_sieve_diag: dict[str, Any] | None = None

    # Stage-3 Secondary Ionization (Cascade Expansion): second pass over low-confidence rows.
    # This is intentionally row-restricted so we don't perturb already-confident rows.
    sec_cycles = int(max(0, low_confidence_secondary_cycles))
    if mode in ("flag", "abstain") and bool(low_confidence_secondary_enabled) and sec_cycles > 0:
        try:
            low_mask = np.asarray(low_conf_mask, dtype=bool)
            if bool(np.any(low_mask)) and z_cache:
                rng_sec = np.random.default_rng(int(random_seed) + 17717)
                shear_sec = float(max(0.0, shear * float(low_confidence_secondary_shear_multiplier)))
                inhib_sec = float(
                    max(0.0, float(inhibition_strength) * float(low_confidence_secondary_inhibition_multiplier))
                )
                visc_mult_end = float(np.clip(float(low_confidence_secondary_viscosity_multiplier), 0.10, 2.50))
                start_raw = low_confidence_secondary_viscosity_multiplier_start
                visc_mult_start = visc_mult_end
                if start_raw is not None:
                    try:
                        visc_mult_start = float(np.clip(float(start_raw), 0.10, 2.50))
                    except Exception:
                        visc_mult_start = visc_mult_end
                anneal_visc = bool(low_confidence_secondary_viscosity_anneal) and sec_cycles > 1

                relax_ion = bool(low_confidence_secondary_relax_ionization_gate) and bool(low_confidence_require_ionized)
                z_min_sec = float(max(0.0, low_confidence_secondary_ionization_z_min))
                relaxed_ion_conf_min = float(np.clip(float(low_confidence_secondary_relaxed_ion_conf_min), 0.0, 1.0))

                use_spearman = bool(low_confidence_secondary_use_spearman)
                spear_min_abs = float(max(0.0, low_confidence_secondary_spearman_min_abs))
                spear_margin = float(max(0.0, low_confidence_secondary_spearman_margin))

                promote_votes = int(max(0, low_confidence_secondary_promote_min_zone_votes))
                promote_conf_min = float(np.clip(float(low_confidence_secondary_promote_conf_min), 0.0, 1.0))

                sieve_enabled = bool(low_confidence_secondary_sieve_enabled)
                sieve_cycles = int(max(0, low_confidence_secondary_sieve_cycles))
                sieve_reverse = float(max(0.0, low_confidence_secondary_sieve_reverse_multiplier))
                sieve_noise = float(max(0.0, low_confidence_secondary_sieve_noise_std))
                sieve_inst_min = float(np.clip(float(low_confidence_secondary_sieve_instability_min), 0.0, 1.0))
                sieve_conf_delta_max = float(max(0.0, low_confidence_secondary_sieve_conf_delta_max))
                sieve_update_max = float(max(0.0, low_confidence_secondary_sieve_update_norm_max))
                sieve_events = 0
                sieve_rows_total = 0

                for cycle_i in range(sec_cycles):
                    conf_prev = np.asarray(conf, dtype="float64").copy()
                    if anneal_visc:
                        t = float(cycle_i) / float(max(1, sec_cycles - 1))
                        visc_mult = float(visc_mult_start + t * (visc_mult_end - visc_mult_start))
                        visc_mult = float(np.clip(visc_mult, 0.10, 2.50))
                    else:
                        visc_mult = visc_mult_end
                    cycle_update = np.zeros_like(logits)
                    for j, cls in enumerate(classes):
                        y01 = (y_cat == str(cls)).astype("float64").to_numpy(dtype="float64")
                        p = _sigmoid(logits[:, j])
                        residual = y01 - p
                        residual_train = residual[train_mask]

                        class_score = np.zeros(df.shape[0], dtype="float64")
                        denom = 0.0
                        for col in feature_cols_used:
                            z = z_cache.get(col)
                            if z is None:
                                continue

                            # Primary (Pearson) charge.
                            charge_p = _pearson_corr(z[train_mask], residual_train)
                            if not math.isfinite(charge_p):
                                charge_p = 0.0

                            # Secondary (Spearman) charge for nonparametric features when useful.
                            medium = migration_by_feature[col]
                            charge = float(charge_p)
                            slow_mover = False
                            if use_spearman and medium.ionization == "nonparametric" and _sp_stats is not None:
                                r_s, _ = _safe_spearmanr(z[train_mask], residual_train)
                                if math.isfinite(r_s) and (abs(float(r_s)) >= spear_min_abs):
                                    if abs(float(r_s)) >= abs(float(charge_p)) + spear_margin:
                                        charge = float(r_s)
                                        slow_mover = True

                            if not math.isfinite(charge) or abs(charge) < 1e-8:
                                continue

                            bond_factor = bond_factors.get(col, 1.0)
                            certainty = 1.0 / (1.0 + max(0.0, medium.standard_error))
                            density = (1.0 + max(0.0, medium.kl_divergence)) * certainty

                            eta_base = max(1e-6, float(medium.viscosity))
                            # Slow-movers get a thinner buffer (lower viscosity) so they can contribute.
                            if slow_mover:
                                eta_base = max(1e-6, eta_base * visc_mult)
                            eta_dynamic = max(1e-6, eta_base / (1.0 + shear_sec * abs(charge) * bond_factor))

                            inhibition = 0.0
                            if competitive_inhibition and col != dominant_global:
                                inhibition = inhib_sec * abs(float(affinity_to_global_dominant.get(col, 0.0)))

                            eff_charge = float(charge)
                            if eff_charge < 0:
                                eff_charge *= neg_mult
                            q = eff_charge * density * bond_factor
                            x_pos = float(pI_map.get(col, 0.5))
                            field = float(plane_mobility) - float(grad1) * x_pos
                            v = (q * field) / (eta_dynamic + inhibition)

                            denom += abs(v)
                            amp = float(pcr_amp_by_feature.get(col, 1.0))
                            class_score += (amp * v) * z

                        if denom <= 1e-12:
                            denom = 1.0
                        cycle_update[:, j] = class_score / denom

                    # Normal secondary update on the currently-low-confidence rows.
                    logits[low_mask, :] += (float(lr) * cycle_update[low_mask, :])
                    probs = _softmax(logits)
                    conf = np.max(probs, axis=1)
                    entropy = _normalized_entropy(probs)
                    if smear_metric_name == "instability":
                        smear = instability
                    else:
                        smear = entropy

                    if combine_rule == "and":
                        low_mask = (conf < float(conf_thr)) & (smear > float(ent_thr))
                    else:
                        low_mask = (conf < float(conf_thr)) | (smear > float(ent_thr))

                    # Relaxed ionization gate for the secondary pass.
                    if relax_ion and ionized_features:
                        ion_hits_sec = np.zeros(df.shape[0], dtype="int32")
                        for col in ionized_features:
                            z = z_cache.get(col)
                            if z is None:
                                continue
                            ion_hits_sec += (np.abs(z) >= z_min_sec).astype("int32")
                        strict_zero = (ionized_hit_count <= 0)
                        relaxed_ok = (ion_hits_sec > 0)
                        allow = (~strict_zero) | (relaxed_ok & (conf >= relaxed_ion_conf_min))
                        low_mask = np.asarray(low_mask, dtype=bool) | (~allow)
                    elif bool(low_confidence_require_ionized):
                        low_mask = np.asarray(low_mask, dtype=bool) | (ionized_hit_count <= 0)

                    # Cluster promotion: if enough features (votes) from the row land in the same zone,
                    # promote it to a prediction even if it remains low-confidence.
                    if promote_votes > 0:
                        promote = (conf >= promote_conf_min) & (zone_cluster_max_votes >= promote_votes)
                        low_mask = np.asarray(low_mask, dtype=bool) & (~promote)

                    # Reciprocating Sieve: if a row stays low-confidence and doesn't improve
                    # its confidence after a secondary update, shake it with a reverse-step + noise.
                    if sieve_enabled and sieve_cycles > 0:
                        try:
                            conf_delta = np.asarray(conf, dtype="float64") - conf_prev
                            update_norm = np.sqrt(np.sum(np.square(cycle_update), axis=1))
                            tangled = (
                                np.asarray(low_mask, dtype=bool)
                                & (np.asarray(instability, dtype="float64") >= sieve_inst_min)
                                & (np.asarray(conf_delta, dtype="float64") <= sieve_conf_delta_max)
                            )
                            if sieve_update_max > 0.0:
                                tangled = tangled & (np.asarray(update_norm, dtype="float64") <= sieve_update_max)
                            if bool(np.any(tangled)):
                                sieve_events += 1
                                sieve_rows_total += int(np.sum(tangled.astype("int32")))
                                for _ in range(sieve_cycles):
                                    if sieve_reverse > 0.0:
                                        logits[tangled, :] += float(lr) * (-sieve_reverse) * cycle_update[tangled, :]
                                    if sieve_noise > 0.0:
                                        logits[tangled, :] += rng_sec.normal(
                                            loc=0.0,
                                            scale=sieve_noise,
                                            size=(int(np.sum(tangled.astype("int32"))), logits.shape[1]),
                                        )
                                probs = _softmax(logits)
                                conf = np.max(probs, axis=1)
                                entropy = _normalized_entropy(probs)
                                if smear_metric_name == "instability":
                                    smear = instability
                                else:
                                    smear = entropy

                                if combine_rule == "and":
                                    low_mask = (conf < float(conf_thr)) & (smear > float(ent_thr))
                                else:
                                    low_mask = (conf < float(conf_thr)) | (smear > float(ent_thr))

                                if relax_ion and ionized_features:
                                    ion_hits_sec = np.zeros(df.shape[0], dtype="int32")
                                    for col in ionized_features:
                                        z = z_cache.get(col)
                                        if z is None:
                                            continue
                                        ion_hits_sec += (np.abs(z) >= z_min_sec).astype("int32")
                                    strict_zero = (ionized_hit_count <= 0)
                                    relaxed_ok = (ion_hits_sec > 0)
                                    allow = (~strict_zero) | (relaxed_ok & (conf >= relaxed_ion_conf_min))
                                    low_mask = np.asarray(low_mask, dtype=bool) | (~allow)
                                elif bool(low_confidence_require_ionized):
                                    low_mask = np.asarray(low_mask, dtype=bool) | (ionized_hit_count <= 0)

                                if promote_votes > 0:
                                    promote = (conf >= promote_conf_min) & (zone_cluster_max_votes >= promote_votes)
                                    low_mask = np.asarray(low_mask, dtype=bool) & (~promote)
                        except Exception:
                            pass

                    if not bool(np.any(low_mask)):
                        break

                # Refresh final predictions after secondary ionization.
                pred_idx = np.argmax(probs, axis=1)
                pred_labels = [classes[int(i)] for i in pred_idx]
                low_conf_mask = np.asarray(low_mask, dtype=bool)

                # Persist sieve counters for diagnostics (best-effort, TEST-only usage).
                try:
                    secondary_sieve_diag = {
                        "enabled": bool(sieve_enabled),
                        "events": int(sieve_events),
                        "rows_total": int(sieve_rows_total),
                        "cycles": int(sieve_cycles),
                        "reverse_multiplier": float(sieve_reverse),
                        "noise_std": float(sieve_noise),
                        "instability_min": float(sieve_inst_min),
                        "conf_delta_max": float(sieve_conf_delta_max),
                        "update_norm_max": float(sieve_update_max),
                    }
                except Exception:
                    secondary_sieve_diag = None
        except Exception:
            pass

    # Build selective diagnostics (TEST-only) after all reionization/secondary passes.
    if mode in ("flag", "abstain"):
        try:
            test_idx = np.asarray(test_mask, dtype=bool)
            n_test = int(np.sum(test_idx))

            def _mask_stats(mask_all_rows: np.ndarray | None) -> dict[str, Any] | None:
                if mask_all_rows is None:
                    return None
                low_test = np.asarray(mask_all_rows, dtype=bool)[test_idx]
                if low_test.size <= 0:
                    return {
                        "n_test": n_test,
                        "n_abstain": 0,
                        "n_keep": 0,
                        "abstain_rate": 0.0,
                        "coverage": 0.0,
                    }
                n_abstain = int(np.sum(low_test))
                n_keep = int(low_test.size - n_abstain)
                abstain_rate = float(np.mean(low_test.astype("float64")))
                return {
                    "n_test": n_test,
                    "n_abstain": n_abstain,
                    "n_keep": n_keep,
                    "abstain_rate": abstain_rate,
                    "coverage": float(1.0 - abstain_rate),
                }

            final_mask = np.asarray(low_conf_mask, dtype=bool)

            conf_low = conf < float(conf_thr)
            smear_high = smear > float(ent_thr)
            if combine_rule == "and":
                base_low = conf_low & smear_high
            else:
                base_low = conf_low | smear_high

            # Ionization-gate blocking status, reconstructed to match the final gating logic.
            ion_gate_blocked = np.zeros(df.shape[0], dtype=bool)
            require_ion = bool(low_confidence_require_ionized)
            relax_ion_enabled = bool(low_confidence_secondary_relax_ionization_gate) and require_ion
            z_min_sec_diag = float(max(0.0, low_confidence_secondary_ionization_z_min))
            relaxed_ion_conf_min_diag = float(np.clip(float(low_confidence_secondary_relaxed_ion_conf_min), 0.0, 1.0))
            if require_ion:
                if relax_ion_enabled and ionized_features and z_cache:
                    ion_hits_sec = np.zeros(df.shape[0], dtype="int32")
                    for col in ionized_features:
                        z = z_cache.get(col)
                        if z is None:
                            continue
                        ion_hits_sec += (np.abs(z) >= z_min_sec_diag).astype("int32")
                    strict_zero = ionized_hit_count <= 0
                    relaxed_ok = ion_hits_sec > 0
                    allow = (~strict_zero) | (relaxed_ok & (conf >= relaxed_ion_conf_min_diag))
                    ion_gate_blocked = ~allow
                else:
                    ion_gate_blocked = ionized_hit_count <= 0

            low_test_final = final_mask[test_idx]
            n_abstain_final = int(np.sum(low_test_final))

            def _count(mask: np.ndarray) -> int:
                return int(np.sum(mask.astype("int32")))

            reasons: dict[str, Any] = {
                "n_test": n_test,
                "n_abstain": n_abstain_final,
                "n_keep": int(n_test - n_abstain_final),
            }
            if n_abstain_final > 0:
                abstain_sel = test_idx & final_mask
                conf_low_n = _count(abstain_sel & conf_low)
                smear_high_n = _count(abstain_sel & smear_high)
                base_low_n = _count(abstain_sel & base_low)
                ion_gate_n = _count(abstain_sel & ion_gate_blocked)
                both_base_ion_n = _count(abstain_sel & base_low & ion_gate_blocked)
                neither_n = _count(abstain_sel & (~base_low) & (~ion_gate_blocked))

                def _pct(n: int) -> float:
                    return float(n) / float(n_abstain_final)

                reasons.update(
                    {
                        "conf_low": {"count": conf_low_n, "pct_of_abstain": _pct(conf_low_n)},
                        "smear_high": {"count": smear_high_n, "pct_of_abstain": _pct(smear_high_n)},
                        "base_low": {"count": base_low_n, "pct_of_abstain": _pct(base_low_n)},
                        "ion_gate_blocked": {"count": ion_gate_n, "pct_of_abstain": _pct(ion_gate_n)},
                        "base_low_and_ion_gate": {"count": both_base_ion_n, "pct_of_abstain": _pct(both_base_ion_n)},
                        "neither_base_nor_ion_gate": {"count": neither_n, "pct_of_abstain": _pct(neither_n)},
                    }
                )

            selective_diagnostics = {
                "mode": mode,
                "smear_metric": smear_metric_name,
                "combine_rule": combine_rule,
                "thresholds": {
                    "conf_thr": float(conf_thr),
                    "smear_thr": float(ent_thr),
                    "auto_conf_quantile": float(np.clip(float(low_confidence_auto_conf_quantile), 0.0, 1.0)),
                    "auto_smear_quantile": float(np.clip(float(low_confidence_auto_smear_quantile), 0.0, 1.0)),
                    "require_ionized": bool(low_confidence_require_ionized),
                    "ionization_pvalue": float(low_confidence_ionization_pvalue),
                    "ionization_z_min": float(max(0.0, low_confidence_ionization_z_min)),
                    "secondary_relax_ion_gate": bool(relax_ion_enabled),
                    "secondary_ionization_z_min": float(z_min_sec_diag),
                    "secondary_relaxed_ion_conf_min": float(relaxed_ion_conf_min_diag),
                    "secondary_viscosity_anneal": bool(low_confidence_secondary_viscosity_anneal),
                    "secondary_viscosity_multiplier_start": None
                    if low_confidence_secondary_viscosity_multiplier_start is None
                    else float(low_confidence_secondary_viscosity_multiplier_start),
                    "secondary_viscosity_multiplier_end": float(low_confidence_secondary_viscosity_multiplier),
                    "secondary_sieve": secondary_sieve_diag,
                },
                "ionized_features": {
                    "count": int(len(ionized_features)),
                    "sample": ionized_features[:20],
                },
                "test_stages": {
                    "pre_reionization": _mask_stats(mask_pre_reion),
                    "post_reionization": _mask_stats(mask_post_reion),
                    "final": _mask_stats(final_mask),
                },
                "final_abstain_reasons": reasons,
            }
        except Exception:
            selective_diagnostics = None
    if mode == "abstain":
        # Replace prediction with a sentinel label (keeps output JSON-compatible).
        pred_labels = [str(low_confidence_label) if bool(low_conf_mask[i]) else pred_labels[i] for i in range(len(pred_labels))]

    y_all = y_cat.to_numpy(dtype="object")
    pred_all = np.array(pred_labels, dtype="object")
    accuracy = float(np.mean((y_all[test_mask] == pred_all[test_mask]).astype("float64")))

    # Selective metrics on TEST only.
    abstain_rate: float | None = None
    coverage: float | None = None
    selective_accuracy: float | None = None
    if mode in ("flag", "abstain"):
        low_test = np.asarray(low_conf_mask, dtype=bool)[test_mask]
        abstain_rate = float(np.mean(low_test.astype("float64"))) if low_test.size else 0.0
        coverage = float(1.0 - abstain_rate)
        keep = ~low_test
        if int(np.sum(keep)) >= 1:
            selective_accuracy = float(np.mean((y_all[test_mask][keep] == pred_all[test_mask][keep]).astype("float64")))
        else:
            selective_accuracy = None

    # PCR-style gel readout on TEST only.
    probs_test = probs[test_mask]
    y_true_test = y_all[test_mask]
    y_pred_test = pred_all[test_mask]
    (
        gel_sharp,
        gel_smear,
        gel_ghost,
        gel_conf_mean,
        gel_conf_std,
    ) = _gel_health_classification(probs_test, y_true_test, y_pred_test)
    best_iter = max(iteration_gains, key=lambda it: it.test_accuracy) if iteration_gains else None

    preview = []
    test_indices = np.flatnonzero(test_mask)
    for idx in test_indices[: min(max_preview_rows, len(test_indices))]:
        i = int(idx)
        row_preview = {
            "row": i,
            "actual": str(y_cat.iloc[i]),
            "predicted": str(pred_labels[i]),
        }
        if mode in ("flag", "abstain"):
            row_preview["confidence"] = None if not math.isfinite(float(conf[i])) else float(conf[i])
            row_preview["smearing"] = None if not math.isfinite(float(smear[i])) else float(smear[i])
            row_preview["smear_metric"] = smear_metric_name
            row_preview["entropy"] = None if not math.isfinite(float(entropy[i])) else float(entropy[i])
            row_preview["instability"] = None if not math.isfinite(float(instability[i])) else float(instability[i])
            row_preview["ionized_hits"] = int(ionized_hit_count[i]) if int(ionized_hit_count.size) > i else 0
            row_preview["ionization_mass"] = None if not math.isfinite(float(ionization_mass_row[i])) else float(ionization_mass_row[i])
            row_preview["viscosity_path"] = None if not math.isfinite(float(viscosity_path[i])) else float(viscosity_path[i])
            row_preview["confirmatory_consensus"] = None if not math.isfinite(float(confirmatory_consensus[i])) else float(confirmatory_consensus[i])
            row_preview["low_confidence"] = bool(low_conf_mask[i])
            # Helpful per-row reason flags (best-effort).
            try:
                row_preview["reason_conf_low"] = bool(conf_low[i])
                row_preview["reason_smear_high"] = bool(smear_high[i])
                row_preview["reason_ion_gate"] = bool(ion_gate_blocked[i])
            except Exception:
                pass
        preview.append(row_preview)

    test_row_indices = None
    test_actual = None
    test_predicted = None
    if return_predictions:
        idx_list = [int(i) for i in test_indices.tolist()]
        test_row_indices = idx_list
        # Ensure plain Python strings for JSON/metrics tooling.
        test_actual = [str(x) for x in y_cat.iloc[idx_list].to_list()]
        test_predicted = [str(x) for x in pred_all[test_mask].tolist()]

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
        buffer_ionization=buffer_ionization,
        buffer_normality_p=buffer_normality_p,
        gel_band_sharpness=gel_sharp,
        gel_smearing=gel_smear,
        gel_ghost_band_rate=gel_ghost,
        gel_confidence_mean=gel_conf_mean,
        gel_confidence_std=gel_conf_std,
        abstain_rate=abstain_rate,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
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

    equilibrium_zones.extend(
        _fractionate_kw_zones(
            df,
            feature_cols_used=feature_cols_used,
            feature_kinds=feature_kinds,
            pI_map=pI_map,
            target_series=target_series,
            target_kind=target_kind,
            train_mask=train_mask,
            start_zone_id=100,
        )
    )

    if bool(abstain_if_uncertain) and should_abstain_from_prediction(
        metrics,
        confidence_std_threshold=float(abstain_confidence_std_threshold),
        smearing_threshold=float(abstain_smearing_threshold),
        min_selective_accuracy=float(abstain_min_selective_accuracy),
    ):
        return None

    diagnostics_payload = (
        None
        if (cleaning_diag is None and selective_diagnostics is None and isotope_diag is None)
        else {
            **({} if cleaning_diag is None else {"cleaning": cleaning_diag}),
            **({} if selective_diagnostics is None else {"selective": selective_diagnostics}),
            **({} if isotope_diag is None else {"isotopes": isotope_diag}),
        }
    )

    result = PredictionResult(
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
        diagnostics=diagnostics_payload,
        test_row_indices=test_row_indices,
        test_actual=test_actual,
        test_predicted=test_predicted,
    )

    if runtime_state is not None:
        try:
            update_predictor_state_from_result(
                runtime_state,
                result,
                low_confidence_rows=[row for row in preview if bool(row.get("low_confidence"))],
                low_confidence_limit=max(1, int(max_preview_rows)),
            )
        except Exception:
            pass

    return result
