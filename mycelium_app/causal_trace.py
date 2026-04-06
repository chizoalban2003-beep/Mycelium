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

    def _fmt_feat(r: dict[str, Any]) -> str:
        f = str(r.get("feature", "feature"))
        d = float(_safe_float(r.get("delta_weight")))
        direction = "more" if d > 0 else "less"
        return f"{f} ({direction} pull)"

    # Build a compact narrative.
    if len(top) == 1:
        narrative = f"I’m leaning {_fmt_feat(top[0])} than before."
    else:
        narrative = f"I’m leaning more on {_fmt_feat(top[0])} and {_fmt_feat(top[1])}."

    return CausalTrace(ok=True, narrative=narrative, top_shifts=top)


def dumps_top_shifts(trace: CausalTrace) -> str:
    try:
        return json.dumps(trace.top_shifts or [], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "[]"
