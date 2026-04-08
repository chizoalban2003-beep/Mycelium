from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import math

import numpy as np
import pandas as pd

from mycelium_app.physics_predictor import (
    BondInfo,
    PredictorError,
    _build_bonding_map,
    _categorical_entropy_and_variance,
    _collinearity_complexes,
    _global_categorical_reference,
    _global_numeric_reference,
    _kl_for_feature,
    _numeric_entropy_and_variance,
    _to_float_array,
    infer_feature_kind,
)

SettlingLayer = Literal["top", "middle", "bottom"]


@dataclass(frozen=True, slots=True)
class SettlingFeatureInfo:
    feature: str
    kind: str
    entropy: float
    variance: float
    standard_error: float
    kl_divergence: float
    mean_affinity: float
    weighted_degree: float
    bond_count: int
    collinearity_complex_id: int | None
    collinearity_complex_size: int
    mass: float
    viscosity: float
    settling_velocity: float
    settling_score: float
    depth: float
    layer: SettlingLayer


@dataclass(frozen=True, slots=True)
class FeatureSettlingResult:
    feature_order: list[str]
    features: list[SettlingFeatureInfo]
    bonding_map: list[BondInfo]
    collinearity_complexes: dict[int, list[str]]
    diagnostics: dict[str, Any]


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _layer_from_depth(depth: float) -> SettlingLayer:
    if depth >= (2.0 / 3.0):
        return "bottom"
    if depth >= (1.0 / 3.0):
        return "middle"
    return "top"


def _summed_weight(weights: tuple[float, ...]) -> float:
    total = float(sum(max(0.0, float(w)) for w in weights))
    return total if total > 0.0 else 1.0


def _rough_intrinsic_density(series: pd.Series, kind: str) -> float:
    missing_ratio = float(series.isna().mean())
    completeness = 1.0 - _clamp01(missing_ratio)
    if kind in ("numeric", "datetime", "bool"):
        arr = _to_float_array(series, kind=kind)  # type: ignore[arg-type]
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            spread = 0.0
        else:
            std = float(np.nanstd(arr))
            spread = float((std**2) / (1.0 + std**2))
    else:
        probs = (
            series.astype("string")
            .fillna("__MISSING__")
            .value_counts(dropna=False, normalize=True)
            .to_numpy(dtype="float64")
        )
        spread = float(1.0 - np.sum(probs**2)) if probs.size else 0.0
    return _clamp01(0.5 * completeness + 0.5 * spread)


def _preselect_feature_columns(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_kinds: dict[str, str],
    max_features: int,
) -> tuple[list[str], list[str]]:
    if len(feature_cols) <= int(max_features):
        return feature_cols, []

    ranked: list[tuple[float, str]] = []
    for col in feature_cols:
        score = _rough_intrinsic_density(df[col], feature_kinds[col])
        ranked.append((float(score), str(col)))
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    selected = [name for _, name in ranked[: int(max_features)]]
    selected_set = set(selected)
    dropped = [c for c in feature_cols if c not in selected_set]
    return selected, dropped


def run_feature_settling(
    df: pd.DataFrame,
    *,
    exclude_cols: list[str] | None = None,
    train_fraction: float = 1.0,
    random_seed: int = 42,
    max_features: int = 120,
    top_bond_pairs: int = 80,
    min_bond_affinity: float = 0.08,
    collinearity_threshold: float = 0.90,
    weight_affinity: float = 0.45,
    weight_variance: float = 0.25,
    weight_low_entropy: float = 0.20,
    weight_kl_divergence: float = 0.10,
) -> FeatureSettlingResult:
    if df.empty:
        raise PredictorError("Dataset is empty")
    if len(df.columns) < 2:
        raise PredictorError("Need at least two columns to compute settling")
    if len(df) < 3:
        raise PredictorError("Need at least 3 rows")

    exclude = {str(c).strip() for c in (exclude_cols or []) if str(c).strip()}
    feature_cols = [str(c) for c in df.columns if str(c) not in exclude]
    if len(feature_cols) < 2:
        raise PredictorError("Need at least two usable features after exclusions")

    feature_kinds = {col: infer_feature_kind(df[col]) for col in feature_cols}
    feature_cols, dropped_cols = _preselect_feature_columns(
        df,
        feature_cols=feature_cols,
        feature_kinds={k: str(v) for k, v in feature_kinds.items()},
        max_features=max(2, int(max_features)),
    )
    feature_kinds = {col: infer_feature_kind(df[col]) for col in feature_cols}

    tf = float(train_fraction)
    tf = 1.0 if tf >= 0.999 else float(np.clip(tf, 0.05, 0.95))
    if tf >= 0.999:
        train_df = df[feature_cols].copy()
    else:
        rng = np.random.default_rng(int(random_seed))
        n = len(df)
        n_train = max(2, min(n - 1, int(round(n * tf))))
        idx = rng.permutation(n)[:n_train]
        train_df = df.iloc[idx][feature_cols].copy()

    if len(train_df) < 3:
        raise PredictorError("Training slice too small for settling analysis")

    global_numeric_ref = _global_numeric_reference(train_df, feature_kinds)
    global_categorical_ref = _global_categorical_reference(train_df, feature_kinds)

    entropy_map: dict[str, float] = {}
    variance_map: dict[str, float] = {}
    stderr_map: dict[str, float] = {}
    kl_map: dict[str, float] = {}
    for col in feature_cols:
        kind = feature_kinds[col]
        series = train_df[col]
        if kind in ("numeric", "datetime", "bool"):
            entropy, variance, stderr = _numeric_entropy_and_variance(_to_float_array(series, kind=kind))
        else:
            entropy, variance, stderr = _categorical_entropy_and_variance(series)
        entropy_map[col] = float(entropy)
        variance_map[col] = float(variance)
        stderr_map[col] = float(stderr)
        kl_map[col] = float(
            _kl_for_feature(
                series,
                kind,
                global_numeric_ref=global_numeric_ref,
                global_categorical_ref=global_categorical_ref,
            )
        )

    # Build full bonding map (subject to affinity threshold) and derive per-feature centrality.
    n_pairs = (len(feature_cols) * (len(feature_cols) - 1)) // 2
    bonds = _build_bonding_map(
        train_df,
        feature_cols,
        feature_kinds,
        top_pairs=max(int(top_bond_pairs), int(n_pairs)),
        min_affinity=float(min_bond_affinity),
    )
    weighted_degree = {col: 0.0 for col in feature_cols}
    bond_count = {col: 0 for col in feature_cols}
    for b in bonds:
        weighted_degree[b.feature_a] = float(weighted_degree[b.feature_a] + float(b.affinity))
        weighted_degree[b.feature_b] = float(weighted_degree[b.feature_b] + float(b.affinity))
        bond_count[b.feature_a] = int(bond_count[b.feature_a] + 1)
        bond_count[b.feature_b] = int(bond_count[b.feature_b] + 1)

    max_possible_neighbors = max(1, len(feature_cols) - 1)
    mean_affinity = {
        col: float(weighted_degree[col]) / float(max_possible_neighbors)
        for col in feature_cols
    }

    complex_by_feature, members_by_complex, _ = _collinearity_complexes(
        train_df,
        feature_cols,
        feature_kinds,
        train_mask=np.ones(len(train_df), dtype=bool),
        threshold=float(collinearity_threshold),
    )

    w_total = _summed_weight((weight_affinity, weight_variance, weight_low_entropy, weight_kl_divergence))
    w_affinity = float(max(0.0, weight_affinity) / w_total)
    w_variance = float(max(0.0, weight_variance) / w_total)
    w_low_entropy = float(max(0.0, weight_low_entropy) / w_total)
    w_kl = float(max(0.0, weight_kl_divergence) / w_total)

    features: list[SettlingFeatureInfo] = []
    for col in feature_cols:
        entropy = _clamp01(entropy_map[col])
        variance = _clamp01(variance_map[col])
        affinity = _clamp01(mean_affinity[col])
        low_entropy = _clamp01(1.0 - entropy)
        kl_norm = float(kl_map[col] / (1.0 + max(0.0, kl_map[col])))
        kl_norm = _clamp01(kl_norm)

        mass = (
            w_affinity * affinity
            + w_variance * variance
            + w_low_entropy * low_entropy
            + w_kl * kl_norm
        )
        c_id = complex_by_feature.get(col)
        c_size = len(members_by_complex.get(c_id, [])) if c_id is not None else 1
        complex_gain = float(1.0 + 0.05 * math.log1p(max(0, c_size - 1)))
        mass = _clamp01(mass * complex_gain)

        # Higher affinity and larger complexes increase drag (viscosity).
        viscosity = float(1.0 + 1.5 * affinity + 0.2 * max(0, c_size - 1))
        settling_velocity = float(mass / max(1e-6, viscosity))
        settling_score = float(0.65 * mass + 0.35 * settling_velocity)

        features.append(
            SettlingFeatureInfo(
                feature=col,
                kind=str(feature_kinds[col]),
                entropy=float(entropy),
                variance=float(variance),
                standard_error=float(stderr_map[col]),
                kl_divergence=float(kl_map[col]),
                mean_affinity=float(affinity),
                weighted_degree=float(weighted_degree[col]),
                bond_count=int(bond_count[col]),
                collinearity_complex_id=None if c_id is None else int(c_id),
                collinearity_complex_size=int(c_size),
                mass=float(mass),
                viscosity=float(viscosity),
                settling_velocity=float(settling_velocity),
                settling_score=float(settling_score),
                depth=0.0,
                layer="top",
            )
        )

    features = sorted(
        features,
        key=lambda f: (float(f.settling_score), float(f.mass), float(f.mean_affinity), f.feature),
        reverse=True,
    )
    n_feat = len(features)
    out_features: list[SettlingFeatureInfo] = []
    for rank, f in enumerate(features):
        depth = 1.0 if n_feat <= 1 else float(rank) / float(max(1, n_feat - 1))
        depth = float(1.0 - depth)  # deeper = denser = closer to 1
        out_features.append(
            SettlingFeatureInfo(
                feature=f.feature,
                kind=f.kind,
                entropy=f.entropy,
                variance=f.variance,
                standard_error=f.standard_error,
                kl_divergence=f.kl_divergence,
                mean_affinity=f.mean_affinity,
                weighted_degree=f.weighted_degree,
                bond_count=f.bond_count,
                collinearity_complex_id=f.collinearity_complex_id,
                collinearity_complex_size=f.collinearity_complex_size,
                mass=f.mass,
                viscosity=f.viscosity,
                settling_velocity=f.settling_velocity,
                settling_score=f.settling_score,
                depth=float(depth),
                layer=_layer_from_depth(depth),
            )
        )

    feature_order = [f.feature for f in out_features]
    col_complexes = {int(cid): members for cid, members in members_by_complex.items() if members}
    diagnostics = {
        "n_rows": int(len(df)),
        "n_train_rows": int(len(train_df)),
        "train_fraction": float(tf),
        "n_features_input": int(len(df.columns)),
        "n_features_used": int(len(feature_cols)),
        "excluded_columns": sorted(list(exclude)),
        "dropped_columns_due_to_max_features": dropped_cols,
        "min_bond_affinity": float(min_bond_affinity),
        "collinearity_threshold": float(collinearity_threshold),
        "weight_profile": {
            "affinity": float(w_affinity),
            "variance": float(w_variance),
            "low_entropy": float(w_low_entropy),
            "kl_divergence": float(w_kl),
        },
    }

    return FeatureSettlingResult(
        feature_order=feature_order,
        features=out_features,
        bonding_map=bonds[: max(0, int(top_bond_pairs))],
        collinearity_complexes=col_complexes,
        diagnostics=diagnostics,
    )
