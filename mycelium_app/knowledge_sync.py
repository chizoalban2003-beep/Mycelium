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


@dataclass(frozen=True)
class LedgerDecision:
    recalled: bool
    stored: bool
    recalled_entry_id: int | None = None
    stored_entry_id: int | None = None
    jaccard: float | None = None
    score_metric: str | None = None
    score_value: float | None = None


class MemoryManager:
    """Stateful wrapper around the Physics Ledger.

    This is the "brain" piece Gemini is describing: a single place where we decide:
    - whether to recall prior physics knobs for similar schemas
    - whether to store the current run's knobs for future recall

    Defaults are conservative: it will not override locked presets unless allowed.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        recall_enabled: bool,
        store_enabled: bool,
        allow_override_locked_presets: bool,
        max_candidates: int,
        min_jaccard: float,
        min_r2_to_store: float,
        min_accuracy_to_store: float,
        min_gel_confidence_mean_to_store: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.recall_enabled = bool(recall_enabled)
        self.store_enabled = bool(store_enabled)
        self.allow_override_locked_presets = bool(allow_override_locked_presets)
        self.max_candidates = int(max_candidates)
        self.min_jaccard = float(min_jaccard)
        self.min_r2_to_store = float(min_r2_to_store)
        self.min_accuracy_to_store = float(min_accuracy_to_store)
        self.min_gel_confidence_mean_to_store = float(min_gel_confidence_mean_to_store)

    def recall(
        self,
        session: Session,
        *,
        user_id: int,
        df: pd.DataFrame,
        target_col: str,
        target_kind: str,
        locked_preset_applied: bool,
    ) -> tuple[dict[str, Any] | None, LedgerDecision, PhysicsLedgerEntry | None]:
        if not (self.enabled and self.recall_enabled):
            return None, LedgerDecision(recalled=False, stored=False), None

        if locked_preset_applied and (not self.allow_override_locked_presets):
            return None, LedgerDecision(recalled=False, stored=False), None

        sig = compute_signature(df, target_col=target_col)
        recalled, entry, jacc = recall_best_kwargs(
            session,
            user_id=int(user_id),
            signature=sig,
            target_kind=str(target_kind),
            max_candidates=int(self.max_candidates),
            min_jaccard=float(self.min_jaccard),
        )

        if not recalled or not entry:
            return None, LedgerDecision(recalled=False, stored=False), None

        decision = LedgerDecision(
            recalled=True,
            stored=False,
            recalled_entry_id=int(entry.id or 0),
            jaccard=float(jacc),
            score_metric=str(entry.score_metric),
            score_value=float(entry.score_value),
        )
        return recalled, decision, entry

    def maybe_store(
        self,
        session: Session,
        *,
        user_id: int,
        project_id: int | None,
        df: pd.DataFrame,
        target_col: str,
        target_kind: str,
        preset_name: str | None,
        preset_display: str | None,
        applied_kwargs: dict[str, Any],
        r2: float | None,
        accuracy: float | None,
        gel_confidence_mean: float | None,
    ) -> tuple[int | None, str | None, float | None]:
        if not (self.enabled and self.store_enabled):
            return None, None, None

        score_metric = ""
        score_value: float | None = None
        if str(target_kind) in ("numeric", "datetime"):
            if r2 is None:
                return None, None, None
            score_metric = "r2"
            score_value = float(r2)
            if score_value < float(self.min_r2_to_store):
                return None, None, None
        else:
            # For categorical, prefer a confidence signal if we have it.
            if gel_confidence_mean is not None and float(gel_confidence_mean) >= float(self.min_gel_confidence_mean_to_store):
                score_metric = "gel_confidence_mean"
                score_value = float(gel_confidence_mean)
            elif accuracy is not None:
                score_metric = "accuracy"
                score_value = float(accuracy)
                if score_value < float(self.min_accuracy_to_store):
                    return None, None, None
            else:
                return None, None, None

        sig = compute_signature(df, target_col=target_col)
        applied = extract_recallable_kwargs(dict(applied_kwargs))
        entry = store_ledger_entry(
            session,
            user_id=int(user_id),
            project_id=project_id,
            signature=sig,
            target_kind=str(target_kind),
            target_col=str(target_col),
            preset_name=preset_name,
            preset_display=preset_display,
            applied_kwargs=applied,
            score_metric=score_metric,
            score_value=float(score_value),
        )
        return int(entry.id or 0), score_metric, float(score_value)
