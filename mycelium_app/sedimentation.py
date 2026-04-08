"""Unsupervised gravitational sedimentation engine.

When no designated target column exists, the dataset ecosystem arranges itself
so that denser (higher information-density) features settle to the bottom and
lighter (high-entropy / noisy) features float to the top.

The implementation mirrors the physics predictor's metaphor:

    - **Density** = sum of absolute correlations × variance (how "heavy" a feature is)
    - **Viscosity** = entropy-based resistance to movement
    - **Settling velocity** = density / viscosity (Stokes-law analogue)
    - **Flocculation** = correlated features clump into aggregates whose combined
      mass lets them sink faster
    - **Layers** = the settled output is binned into strata:
        Turbulent Surface (gas)  → high-entropy froth
        Suspension Layer (liquid) → moderate-density active complexes
        Bedrock Floor (solid)    → dense, stable foundations

This module is intentionally self-contained: it does not require a target column,
does not import from physics_predictor (to stay decoupled), and uses only numpy,
pandas, and scipy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FeatureSediment:
    """One feature's sedimentation profile."""

    feature: str
    density: float
    viscosity: float
    settling_velocity: float
    depth: float
    layer: Literal["bedrock", "suspension", "turbulent"]
    complex_id: int | None
    complex_size: int | None
    entropy: float
    variance: float
    correlation_sum: float
    vif: float | None


@dataclass(frozen=True, slots=True)
class Complex:
    """A flocculated group of features that settled together."""

    complex_id: int
    features: tuple[str, ...]
    combined_density: float
    mean_settling_velocity: float
    mean_depth: float
    layer: Literal["bedrock", "suspension", "turbulent"]
    internal_cohesion: float


@dataclass(frozen=True, slots=True)
class SedimentationResult:
    """Full result of one sedimentation run."""

    features: list[FeatureSediment]
    complexes: list[Complex]
    layer_summary: dict[str, Any]
    correlation_matrix: dict[str, dict[str, float]]
    n_rows: int
    n_features: int
    digest: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_entropy(series: pd.Series) -> float:
    """Shannon entropy of a discrete or discretized series."""
    try:
        if series.dtype.kind == "f" or series.dtype.kind == "i":
            binned = pd.cut(series.dropna(), bins=min(20, max(2, len(series) // 5)), labels=False)
            counts = binned.value_counts(dropna=True)
        else:
            counts = series.value_counts(dropna=True)
        if counts.empty:
            return 0.0
        probs = counts.values / counts.values.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
    except Exception:
        return 0.0


def _safe_variance(series: pd.Series) -> float:
    try:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) < 2:
            return 0.0
        return float(np.var(numeric, ddof=1))
    except Exception:
        return 0.0


def _numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract or encode columns into a fully numeric matrix for correlation."""
    parts: list[pd.Series] = []
    for col in df.columns:
        s = df[col]
        if s.dtype.kind in ("f", "i", "u"):
            parts.append(s.astype(float))
        elif s.dtype.kind == "b":
            parts.append(s.astype(float))
        else:
            try:
                numeric = pd.to_numeric(s, errors="coerce")
                if numeric.notna().sum() > len(s) * 0.5:
                    parts.append(numeric)
                    continue
            except Exception:
                pass
            try:
                codes = s.astype("category").cat.codes.astype(float)
                codes[codes < 0] = np.nan
                parts.append(codes)
            except Exception:
                continue
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, axis=1)
    result.columns = [df.columns[i] for i in range(len(parts))] if len(parts) == len(df.columns) else [p.name for p in parts]
    return result


def _compute_vif(corr_matrix: np.ndarray, idx: int) -> float | None:
    """VIF for feature at index `idx` from a correlation matrix."""
    n = corr_matrix.shape[0]
    if n < 2:
        return None
    try:
        inv = np.linalg.inv(corr_matrix)
        return float(inv[idx, idx])
    except np.linalg.LinAlgError:
        return None


def _find_complexes(
    corr_abs: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.7,
) -> list[list[int]]:
    """Union-find on |correlation| >= threshold to produce feature groups."""
    n = corr_abs.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if corr_abs[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return [members for members in groups.values() if len(members) >= 2]


def _layer_from_depth(depth: float, n_layers: int = 3) -> Literal["bedrock", "suspension", "turbulent"]:
    if depth <= 1.0 / n_layers:
        return "turbulent"
    elif depth <= 2.0 / n_layers:
        return "suspension"
    else:
        return "bedrock"


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_sedimentation(
    df: pd.DataFrame,
    *,
    flocculation_threshold: float = 0.7,
    max_features: int = 200,
    n_iterations: int = 10,
    gravity: float = 1.0,
) -> SedimentationResult:
    """Run unsupervised gravitational sedimentation on a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The dataset. All columns are treated as features (no target).
    flocculation_threshold : float
        Minimum |correlation| for two features to flocculate into a complex.
    max_features : int
        Cap features to avoid excessive computation.
    n_iterations : int
        Number of iterative settling passes. Each pass refines depth by
        pulling complexes tighter and re-normalizing.
    gravity : float
        Global gravity multiplier. Higher = faster separation between layers.

    Returns
    -------
    SedimentationResult
    """
    if df.empty:
        return SedimentationResult(
            features=[], complexes=[], layer_summary={},
            correlation_matrix={}, n_rows=0, n_features=0, digest="",
        )

    df_clean = df.iloc[:, :max_features].copy()
    df_clean = df_clean.dropna(axis=1, how="all")
    if df_clean.empty:
        return SedimentationResult(
            features=[], complexes=[], layer_summary={},
            correlation_matrix={}, n_rows=len(df), n_features=0, digest="",
        )

    num_matrix = _numeric_matrix(df_clean)
    if num_matrix.empty or num_matrix.shape[1] < 2:
        return SedimentationResult(
            features=[], complexes=[], layer_summary={},
            correlation_matrix={}, n_rows=len(df), n_features=0, digest="",
        )

    feature_names = list(num_matrix.columns)
    n_features = len(feature_names)

    filled = num_matrix.fillna(num_matrix.median())
    corr = filled.corr(method="pearson").fillna(0.0).values
    np.fill_diagonal(corr, 1.0)
    corr_abs = np.abs(corr)

    # --- Per-feature raw statistics ---
    entropies = np.array([_safe_entropy(df_clean[f]) for f in feature_names])
    variances = np.array([_safe_variance(df_clean[f]) for f in feature_names])
    corr_sums = np.array([float(corr_abs[i].sum() - 1.0) for i in range(n_features)])

    # VIF computation
    vifs: list[float | None] = []
    try:
        for i in range(n_features):
            vifs.append(_compute_vif(corr, i))
    except Exception:
        vifs = [None] * n_features

    # --- Flocculation: find complexes ---
    raw_complexes = _find_complexes(corr_abs, feature_names, threshold=flocculation_threshold)
    feature_to_complex: dict[int, int] = {}
    for cid, members in enumerate(raw_complexes):
        for m in members:
            feature_to_complex[m] = cid

    # --- Density & viscosity ---
    # Density = correlation_sum × (1 + log(1 + variance)) — how "heavy" the feature is
    # Viscosity = 1 + entropy — resistance to settling
    log_var = np.log1p(np.maximum(variances, 0.0))
    raw_density = corr_sums * (1.0 + log_var)
    raw_density = np.maximum(raw_density, 1e-9)

    raw_viscosity = 1.0 + entropies

    # Complex boost: features in a complex get their density boosted
    # by the aggregate of their complex members (flocculation effect)
    boosted_density = raw_density.copy()
    for cid, members in enumerate(raw_complexes):
        aggregate_mass = float(sum(raw_density[m] for m in members))
        for m in members:
            boosted_density[m] = aggregate_mass / len(members) + raw_density[m]

    # --- Settling velocity (Stokes-law analogue) ---
    settling_velocity = gravity * boosted_density / raw_viscosity

    # --- Iterative settling ---
    # Each iteration pulls features in the same complex closer and re-normalizes
    depths = settling_velocity.copy()
    for _iteration in range(n_iterations):
        # Complex cohesion: average depth within complex, pull members toward it
        for _cid, members in enumerate(raw_complexes):
            mean_depth = float(np.mean(depths[members]))
            for m in members:
                depths[m] = 0.7 * depths[m] + 0.3 * mean_depth

        # Re-normalize to [0, 1]
        d_min, d_max = float(depths.min()), float(depths.max())
        if d_max - d_min > 1e-12:
            depths = (depths - d_min) / (d_max - d_min)
        else:
            depths = np.full_like(depths, 0.5)

    # --- Build result objects ---
    feature_sediments: list[FeatureSediment] = []
    for i, feat in enumerate(feature_names):
        cid = feature_to_complex.get(i)
        csize = len(raw_complexes[cid]) if cid is not None else None
        feature_sediments.append(FeatureSediment(
            feature=feat,
            density=round(float(boosted_density[i]), 6),
            viscosity=round(float(raw_viscosity[i]), 6),
            settling_velocity=round(float(settling_velocity[i]), 6),
            depth=round(float(depths[i]), 6),
            layer=_layer_from_depth(float(depths[i])),
            complex_id=cid,
            complex_size=csize,
            entropy=round(float(entropies[i]), 6),
            variance=round(float(variances[i]), 6),
            correlation_sum=round(float(corr_sums[i]), 6),
            vif=round(float(vifs[i]), 4) if vifs[i] is not None else None,
        ))

    feature_sediments.sort(key=lambda s: s.depth, reverse=True)

    # Complex summaries
    complex_results: list[Complex] = []
    for cid, members in enumerate(raw_complexes):
        member_names = tuple(feature_names[m] for m in members)
        member_depths = [float(depths[m]) for m in members]
        member_velocities = [float(settling_velocity[m]) for m in members]
        mean_d = float(np.mean(member_depths))
        # Internal cohesion: mean pairwise |correlation| within the complex
        if len(members) >= 2:
            pairs = []
            for ii in range(len(members)):
                for jj in range(ii + 1, len(members)):
                    pairs.append(float(corr_abs[members[ii], members[jj]]))
            cohesion = float(np.mean(pairs)) if pairs else 0.0
        else:
            cohesion = 1.0

        complex_results.append(Complex(
            complex_id=cid,
            features=member_names,
            combined_density=round(float(sum(boosted_density[m] for m in members)), 6),
            mean_settling_velocity=round(float(np.mean(member_velocities)), 6),
            mean_depth=round(mean_d, 6),
            layer=_layer_from_depth(mean_d),
            internal_cohesion=round(cohesion, 6),
        ))

    complex_results.sort(key=lambda c: c.mean_depth, reverse=True)

    # Layer summary
    layers = {"bedrock": [], "suspension": [], "turbulent": []}
    for fs in feature_sediments:
        layers[fs.layer].append(fs.feature)

    layer_summary = {}
    for layer_name, feats in layers.items():
        layer_summary[layer_name] = {
            "count": len(feats),
            "features": feats[:20],
            "mean_density": round(float(np.mean([
                fs.density for fs in feature_sediments if fs.layer == layer_name
            ])), 6) if feats else 0.0,
            "mean_viscosity": round(float(np.mean([
                fs.viscosity for fs in feature_sediments if fs.layer == layer_name
            ])), 6) if feats else 0.0,
        }

    # Correlation matrix as nested dict (capped for JSON serialization)
    corr_dict: dict[str, dict[str, float]] = {}
    cap = min(n_features, 50)
    for i in range(cap):
        row: dict[str, float] = {}
        for j in range(cap):
            row[feature_names[j]] = round(float(corr[i, j]), 4)
        corr_dict[feature_names[i]] = row

    # Digest for deduplication
    digest_input = json.dumps({
        "n_rows": len(df),
        "n_features": n_features,
        "columns": sorted(feature_names),
    }, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:16]

    return SedimentationResult(
        features=feature_sediments,
        complexes=complex_results,
        layer_summary=layer_summary,
        correlation_matrix=corr_dict,
        n_rows=len(df),
        n_features=n_features,
        digest=digest,
    )
