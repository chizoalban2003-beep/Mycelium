"""Shared math utilities used across the codebase."""

from __future__ import annotations

import math


def r2_from_actual_pred(actual: list[object] | None, predicted: list[object] | None) -> float | None:
    """Compute R-squared from actual and predicted lists."""
    if not actual or not predicted:
        return None
    pairs: list[tuple[float, float]] = []
    for a, b in zip(actual, predicted, strict=False):
        if a is None or b is None:
            continue
        try:
            af = float(a)
            bf = float(b)
        except Exception:
            continue
        if math.isfinite(af) and math.isfinite(bf):
            pairs.append((af, bf))

    if len(pairs) < 2:
        return None

    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    y_bar = sum(y_true) / float(len(y_true))
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=False))
    ss_tot = sum((a - y_bar) ** 2 for a in y_true)
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)
