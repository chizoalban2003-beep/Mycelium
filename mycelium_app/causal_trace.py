from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mycelium_app.physics_predictor import PredictionResult, WeightInfo


@dataclass(frozen=True)
class CausalTrace:
    ok: bool
    method: str = "weights_shift"
    narrative: str | None = None
    top_shifts: list[dict[str, Any]] | None = None
    notes: str | None = None


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def _weights_by_feature(weights: list[WeightInfo] | None) -> dict[str, WeightInfo]:
    out: dict[str, WeightInfo] = {}
    for wi in weights or []:
        try:
            feat = str(wi.feature)
        except Exception:
            continue
        out[feat] = wi
    return out


def _shift_direction(delta: float, signed: bool) -> str:
    if delta > 0:
                return "more"
    if delta < 0:
                return "less" if signed else "weaker"
    return "steady"


def _format_shift(row: dict[str, Any]) -> str:
    feature = str(row.get("feature", "feature"))
    delta = _safe_float(row.get("delta_weight"))
    change = str(row.get("change", "shifted"))
    method = str(row.get("method", ""))
    feature_kind = str(row.get("feature_kind", ""))
    signed = bool(row.get("signed", False))
    direction = _shift_direction(delta, signed)

    parts = [feature, f"{direction} pull"]
    if change != "shifted":
        parts.append(change)
    if feature_kind:
        parts.append(feature_kind)
    if method:
        parts.append(method)
    parts.append(f"Δ={delta:+.4f}")
    return " • ".join(parts)


def extract_causal_trace(
    baseline: PredictionResult,
    trial: PredictionResult,
    *,
    top_k: int = 5,
    min_abs_delta: float = 1e-6,
) -> CausalTrace:
    """Compute a simple explanation from baseline vs trial feature weights.

    The PhysicsPredictor already produces an interpretable `weights` list (top-k
    selected features). We compare that list between runs to surface the largest
    shifts, including added/removed features.

    This is intentionally lightweight and dependency-free.
    """

    try:
        b_map = _weights_by_feature(getattr(baseline, "weights", None))
        t_map = _weights_by_feature(getattr(trial, "weights", None))
    except Exception as e:
        return CausalTrace(ok=False, notes=f"weights_unavailable:{type(e).__name__}")

    if not b_map and not t_map:
        return CausalTrace(ok=False, notes="no_weights")

    all_features = sorted(set(b_map.keys()) | set(t_map.keys()))
    shifts: list[dict[str, Any]] = []
    for feat in all_features:
        bw = _safe_float(b_map.get(feat).weight if feat in b_map else 0.0)
        tw = _safe_float(t_map.get(feat).weight if feat in t_map else 0.0)
        delta = float(tw - bw)
        if abs(delta) < float(min_abs_delta):
            continue

        src = t_map.get(feat) or b_map.get(feat)
        method = str(getattr(src, "method", ""))
        feature_kind = str(getattr(src, "feature_kind", ""))
        signed = bool(getattr(src, "signed", False))

        shifts.append(
            {
                "feature": feat,
                "baseline_weight": bw,
                "trial_weight": tw,
                "delta_weight": delta,
                "method": method,
                "feature_kind": feature_kind,
                "signed": signed,
                "change": (
                    "added" if (feat not in b_map and feat in t_map) else "removed" if (feat in b_map and feat not in t_map) else "shifted"
                ),
            }
        )

    if not shifts:
        return CausalTrace(ok=False, notes="no_weight_deltas")

    # Rank by magnitude of delta.
    shifts_sorted = sorted(shifts, key=lambda r: abs(_safe_float(r.get("delta_weight"))), reverse=True)
    top = shifts_sorted[: max(1, min(int(top_k), 25))]

    top_text = _format_shift(top[0])
    if len(top) == 1:
        narrative = f"Top shift: {top_text}."
    else:
        next_text = _format_shift(top[1])
        narrative = f"Top shifts: {top_text}; {next_text}."

    return CausalTrace(ok=True, narrative=narrative, top_shifts=top)


def dumps_top_shifts(trace: CausalTrace) -> str:
    try:
        return json.dumps(trace.top_shifts or [], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "[]"
