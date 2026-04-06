from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlmodel import Session, select

from mycelium_app.models import PhysicsLedgerEntry


@dataclass(frozen=True)
class LedgerSignature:
    feature_cols: tuple[str, ...]
    dtypes: dict[str, str]


def compute_signature(df: pd.DataFrame, *, target_col: str) -> LedgerSignature:
    cols = [c for c in df.columns if c != target_col]
    cols_sorted = tuple(sorted(map(str, cols)))
    dtypes = {str(c): str(df[c].dtype) for c in cols_sorted if c in df.columns}
    return LedgerSignature(feature_cols=cols_sorted, dtypes=dtypes)


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union)


def _loads_json(obj: str, default: Any) -> Any:
    try:
        return json.loads(obj)
    except Exception:
        return default


def recall_best_kwargs(
    session: Session,
    *,
    user_id: int,
    signature: LedgerSignature,
    target_kind: str,
    max_candidates: int,
    min_jaccard: float,
) -> tuple[dict[str, Any] | None, PhysicsLedgerEntry | None, float]:
    stmt = (
        select(PhysicsLedgerEntry)
        .where(PhysicsLedgerEntry.created_by_user_id == int(user_id))
        .where(PhysicsLedgerEntry.target_kind == str(target_kind))
        .order_by(PhysicsLedgerEntry.score_value.desc(), PhysicsLedgerEntry.created_at.desc())
        .limit(int(max_candidates))
    )
    candidates = session.exec(stmt).all()

    best_entry: PhysicsLedgerEntry | None = None
    best_j = 0.0
    for e in candidates:
        cols = tuple(_loads_json(e.feature_cols_json, []))
        j = _jaccard(signature.feature_cols, cols)
        if j < float(min_jaccard):
            continue
        if best_entry is None:
            best_entry = e
            best_j = j
            continue
        # Primary sort: score_value already DESC from query; break ties by higher Jaccard.
        if j > best_j:
            best_entry = e
            best_j = j

    if not best_entry:
        return None, None, 0.0

    kwargs = _loads_json(best_entry.applied_kwargs_json, {})
    if not isinstance(kwargs, dict):
        return None, None, best_j

    return kwargs, best_entry, best_j


def store_ledger_entry(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    signature: LedgerSignature,
    target_kind: str,
    target_col: str,
    preset_name: str | None,
    preset_display: str | None,
    applied_kwargs: dict[str, Any],
    score_metric: str,
    score_value: float,
) -> PhysicsLedgerEntry:
    def _jsonable(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        # Common enum-like objects (PhysicsPlane, etc.)
        if hasattr(v, "value"):
            try:
                vv = getattr(v, "value")
                if isinstance(vv, (str, int, float, bool)) or vv is None:
                    return vv
            except Exception:
                pass
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        return str(v)

    applied_kwargs = _jsonable(applied_kwargs)
    entry = PhysicsLedgerEntry(
        created_by_user_id=int(user_id),
        project_id=int(project_id) if project_id is not None else None,
        target_kind=str(target_kind),
        target_col=str(target_col or ""),
        feature_cols_json=json.dumps(list(signature.feature_cols)),
        dtypes_json=json.dumps(dict(signature.dtypes)),
        preset_name=preset_name,
        preset_display=preset_display,
        applied_kwargs_json=json.dumps(applied_kwargs),
        score_metric=str(score_metric),
        score_value=float(score_value),
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def extract_recallable_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs down to the knobs we consider safe to recall.

    This is intentionally conservative: we only keep the "physics" + cleaning knobs
    and avoid copying request-specific items like target_col or max_rows.
    """

    allow_prefixes = (
        "multibuffer_",
        "mycelium_",
        "low_confidence_",
        "cleaning_",
        "stage2_",
    )
    allow_exact = {
        "plane",
        "n_cycles",
        "cycle_learning_rate",
        "cascade_enabled",
        "competitive_inhibition",
        "thermal_noise",
        "top_k_weights",
        "train_fraction",
        "random_seed",
        "inhibition_strength",
        "scavenger_cycles",
    }

    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in ("target_col", "return_predictions"):
            continue
        if k in allow_exact:
            # Store enums as their underlying values for safe JSON.
            if hasattr(v, "value"):
                try:
                    out[k] = getattr(v, "value")
                    continue
                except Exception:
                    pass
            out[k] = v
            continue
        if any(str(k).startswith(p) for p in allow_prefixes):
            out[k] = v

    return out
