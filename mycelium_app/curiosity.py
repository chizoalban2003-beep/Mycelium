from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from mycelium_app.hive_empathy import queue_outbox_message
from mycelium_app.models import CuriosityAnswer, CuriosityCase, NexusNudge
from mycelium_app.physics_predictor import PredictionResult
from mycelium_app.settings import settings


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _loads_list(s: str | None) -> list[Any]:
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def dataframe_digest(df: pd.DataFrame) -> str:
    """Best-effort digest for a dataframe without storing raw content."""

    h = hashlib.sha256()
    try:
        h.update(str(tuple(df.columns)).encode("utf-8"))
        h.update(str(tuple(str(t) for t in df.dtypes)).encode("utf-8"))
        h.update(str(int(df.shape[0])).encode("utf-8"))
        h.update(str(int(df.shape[1])).encode("utf-8"))

        # Add a tiny sketch of values (non-reversible, truncated).
        head = df.head(25)
        for col in head.columns[: min(20, int(head.shape[1]))]:
            s = head[col]
            vals = []
            for v in s[:30].tolist():
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    continue
                vals.append(str(v)[:80])
                if len(vals) >= 10:
                    break
            h.update(str(col).encode("utf-8"))
            h.update("|".join(vals).encode("utf-8"))

        return h.hexdigest()
    except Exception:
        return hashlib.sha256(str(datetime.utcnow().isoformat()).encode("utf-8")).hexdigest()


def _parse_safe_columns(df: pd.DataFrame) -> list[str]:
    raw = str(getattr(settings, "nexus_active_curiosity_safe_columns_csv", "") or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    cols = [p for p in parts if p]
    # Only keep columns that exist.
    return [c for c in cols if c in df.columns]


def _stable_case_uuid(*, user_id: int, project_id: int | None, dataset_digest: str, target_col: str, row_index: int) -> str:
    obj = {
        "user_id": int(user_id),
        "project_id": project_id,
        "dataset_digest": str(dataset_digest),
        "target_col": str(target_col),
        "row_index": int(row_index),
    }
    return hashlib.sha256(_dumps(obj).encode("utf-8")).hexdigest()


def _row_fingerprint(*, dataset_digest: str, row_index: int, excerpt: dict[str, Any]) -> str:
    obj = {"dataset_digest": str(dataset_digest), "row_index": int(row_index), "excerpt": excerpt}
    return hashlib.sha256(_dumps(obj).encode("utf-8")).hexdigest()


def _compute_errors(pred: PredictionResult) -> tuple[str, list[float], list[int]]:
    idxs = getattr(pred, "test_row_indices", None) or []
    actual = getattr(pred, "test_actual", None) or []
    predicted = getattr(pred, "test_predicted", None) or []

    if not idxs or not actual or not predicted:
        return "none", [], []

    tk = str(getattr(pred, "target_kind", ""))

    errs: list[float] = []
    out_idxs: list[int] = []
    if tk in ("numeric", "datetime"):
        for i, (row_i, a, p) in enumerate(zip(idxs, actual, predicted, strict=False)):
            try:
                af = float(a)
                pf = float(p)
            except Exception:
                continue
            if not (math.isfinite(af) and math.isfinite(pf)):
                continue
            errs.append(float(abs(af - pf)))
            out_idxs.append(int(row_i))
        return "abs_error", errs, out_idxs

    # categorical
    for row_i, a, p in zip(idxs, actual, predicted, strict=False):
        if a is None or p is None:
            continue
        miss = 0.0 if str(a) == str(p) else 1.0
        errs.append(float(miss))
        out_idxs.append(int(row_i))
    return "miss", errs, out_idxs


def _question_for_case(*, target_kind: str, predicted: object, actual: object, error_kind: str, error_value: float) -> str:
    if error_kind == "miss":
        return (
            f"I predicted '{predicted}' but the label was '{actual}'. "
            "What detail should I pay more attention to next time?"
        )

    # numeric/datetime
    try:
        pv = float(predicted)  # type: ignore[arg-type]
        av = float(actual)  # type: ignore[arg-type]
        pv_s = str(round(pv, 4))
        av_s = str(round(av, 4))
    except Exception:
        pv_s = str(predicted)
        av_s = str(actual)

    ev_s = str(round(float(error_value), 4))
    return (
        f"I predicted {pv_s} but the value was {av_s} (error≈{ev_s}). "
        "Is there a hidden benefit, rare skill, or special case that explains this?"
    )


def capture_agitated_cases(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    df: pd.DataFrame,
    target_col: str,
    pred: PredictionResult,
) -> list[int]:
    """Capture a few high-error samples from a predictor run.

    Returns IDs of new CuriosityCase rows created.
    """

    if not bool(getattr(settings, "nexus_active_curiosity_enabled", False)):
        return []

    error_kind, errs, row_idxs = _compute_errors(pred)
    if not errs or not row_idxs:
        return []

    max_cases = max(0, min(int(getattr(settings, "nexus_active_curiosity_max_cases_per_run", 3)), 25))
    if max_cases <= 0:
        return []

    min_abs = float(getattr(settings, "nexus_active_curiosity_min_abs_error", 0.0) or 0.0)
    q = float(getattr(settings, "nexus_active_curiosity_min_error_quantile", 0.97) or 0.0)
    q = float(np.clip(q, 0.0, 1.0))

    thresh = 0.0
    try:
        thresh = float(np.quantile(np.array(errs, dtype=float), q))
    except Exception:
        thresh = 0.0

    # Pair and rank by error desc.
    ranked = sorted(list(zip(row_idxs, errs, strict=False)), key=lambda t: float(t[1]), reverse=True)

    dataset_dig = dataframe_digest(df)
    safe_cols = _parse_safe_columns(df)

    created_ids: list[int] = []
    created_any = False

    for row_index, err in ranked:
        if len(created_ids) >= max_cases:
            break
        if error_kind == "abs_error" and float(err) < max(float(thresh), float(min_abs)):
            continue
        if error_kind == "miss" and float(err) < float(thresh):
            continue

        if row_index < 0 or row_index >= int(df.shape[0]):
            continue

        # Build excerpt (allowlisted columns only).
        excerpt: dict[str, Any] = {}
        try:
            row = df.iloc[int(row_index)]
            for c in safe_cols:
                v = row.get(c)
                if v is None:
                    excerpt[str(c)] = None
                else:
                    s = str(v)
                    excerpt[str(c)] = (s[:200] + "…") if len(s) > 200 else s
        except Exception:
            excerpt = {}

        # Pull predicted/actual.
        predicted_v: object = None
        actual_v: object = None
        try:
            # Find matching entry by index position.
            pos = row_idxs.index(int(row_index))
            predicted_v = (getattr(pred, "test_predicted", None) or [None])[pos]
            actual_v = (getattr(pred, "test_actual", None) or [None])[pos]
        except Exception:
            predicted_v = None
            actual_v = None

        case_uuid = _stable_case_uuid(
            user_id=int(user_id),
            project_id=project_id,
            dataset_digest=str(dataset_dig),
            target_col=str(target_col),
            row_index=int(row_index),
        )

        # Avoid duplicates.
        existing = session.exec(select(CuriosityCase).where(CuriosityCase.case_uuid == str(case_uuid))).first()
        if existing is not None:
            continue

        fp = _row_fingerprint(dataset_digest=str(dataset_dig), row_index=int(row_index), excerpt=excerpt)
        question = _question_for_case(
            target_kind=str(getattr(pred, "target_kind", "")),
            predicted=predicted_v,
            actual=actual_v,
            error_kind=str(error_kind),
            error_value=float(err),
        )

        row = CuriosityCase(
            created_by_user_id=int(user_id),
            project_id=int(project_id) if project_id is not None else None,
            case_uuid=str(case_uuid),
            dataset_digest=str(dataset_dig),
            target_col=str(target_col),
            target_kind=str(getattr(pred, "target_kind", "")),
            row_index=int(row_index),
            row_fingerprint=str(fp),
            predicted_json=_dumps(predicted_v),
            actual_json=_dumps(actual_v),
            error_value=float(err),
            error_kind=str(error_kind),
            excerpt_json=_dumps(excerpt),
            question=str(question),
            status="pending",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        if row.id is not None:
            created_ids.append(int(row.id))
            created_any = True

    if created_any and bool(getattr(settings, "nexus_active_curiosity_nudge_enabled", True)):
        _maybe_nudge_curiosity(session, user_id=int(user_id), project_id=project_id)

    return created_ids


def _maybe_nudge_curiosity(session: Session, *, user_id: int, project_id: int | None) -> None:
    throttle_min = max(5, min(int(getattr(settings, "nexus_active_curiosity_nudge_throttle_minutes", 120)), 7 * 24 * 60))
    since = datetime.utcnow() - timedelta(minutes=int(throttle_min))

    # Pending cases?
    q = select(CuriosityCase).where(CuriosityCase.created_by_user_id == int(user_id)).where(CuriosityCase.status == "pending")
    if project_id is None:
        q = q.where(CuriosityCase.project_id.is_(None))
    else:
        q = q.where(CuriosityCase.project_id == int(project_id))
    pending = session.exec(q.limit(1)).first()
    if pending is None:
        return

    # Throttle by last nudge.
    qn = (
        select(NexusNudge)
        .where(NexusNudge.created_by_user_id == int(user_id))
        .where(NexusNudge.kind == "curiosity")
        .where(NexusNudge.created_at >= since)
        .order_by(NexusNudge.created_at.desc())
        .limit(1)
    )
    if project_id is None:
        qn = qn.where(NexusNudge.project_id.is_(None))
    else:
        qn = qn.where(NexusNudge.project_id == int(project_id))

    recent = session.exec(qn).first()
    if recent is not None:
        return

    n = NexusNudge(
        created_by_user_id=int(user_id),
        project_id=int(project_id) if project_id is not None else None,
        kind="curiosity",
        title="Quick question (Active Curiosity)",
        message="I found a high-error sample. Can you share a short explanation or correction? Visit /curiosity.",
        payload_json=_dumps({"pending": True}),
    )
    session.add(n)
    session.commit()


def list_recent_cases(session: Session, *, user_id: int, project_id: int | None, limit: int = 20) -> list[CuriosityCase]:
    limit = max(1, min(int(limit), 100))
    q = select(CuriosityCase).where(CuriosityCase.created_by_user_id == int(user_id))
    if project_id is None:
        q = q.where(CuriosityCase.project_id.is_(None))
    else:
        q = q.where(CuriosityCase.project_id == int(project_id))
    q = q.order_by(CuriosityCase.created_at.desc()).limit(limit)
    return list(session.exec(q).all())


def next_pending_case(session: Session, *, user_id: int, project_id: int | None) -> CuriosityCase | None:
    q = select(CuriosityCase).where(CuriosityCase.created_by_user_id == int(user_id)).where(CuriosityCase.status == "pending")
    if project_id is None:
        q = q.where(CuriosityCase.project_id.is_(None))
    else:
        q = q.where(CuriosityCase.project_id == int(project_id))
    q = q.order_by(CuriosityCase.created_at.asc()).limit(1)
    return session.exec(q).first()


def answer_case(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    case_id: int,
    answer_text: str,
    corrected_target: object | None = None,
    tags: list[str] | None = None,
    export_to_hive: bool = True,
) -> int:
    row = session.exec(
        select(CuriosityCase)
        .where(CuriosityCase.id == int(case_id))
        .where(CuriosityCase.created_by_user_id == int(user_id))
    ).first()
    if row is None:
        raise ValueError("not_found")
    if str(row.status) != "pending":
        raise ValueError("not_pending")

    tags0 = [str(t).strip() for t in (tags or []) if str(t).strip()]
    ans = CuriosityAnswer(
        created_by_user_id=int(user_id),
        project_id=int(project_id) if project_id is not None else None,
        case_id=int(case_id),
        answer_text=str(answer_text or "")[:2000],
        corrected_target_json=_dumps(corrected_target),
        tags_json=_dumps(tags0),
    )
    session.add(ans)
    session.commit()
    session.refresh(ans)

    row.status = "answered"
    row.answered_at = datetime.utcnow()
    session.add(row)
    session.commit()

    if export_to_hive:
        _queue_answer_feedback(session, case=row, ans=ans)

    return int(ans.id or 0)


def dismiss_case(session: Session, *, user_id: int, case_id: int) -> None:
    row = session.exec(
        select(CuriosityCase)
        .where(CuriosityCase.id == int(case_id))
        .where(CuriosityCase.created_by_user_id == int(user_id))
    ).first()
    if row is None:
        raise ValueError("not_found")
    if str(row.status) != "pending":
        return
    row.status = "dismissed"
    row.dismissed_at = datetime.utcnow()
    session.add(row)
    session.commit()


def _queue_answer_feedback(session: Session, *, case: CuriosityCase, ans: CuriosityAnswer) -> None:
    # Export only minimal, non-sensitive aggregate signals.
    try:
        if not bool(getattr(settings, "hive_enabled", False)):
            return

        payload = {
            "meta": {
                "created_at": ans.created_at.isoformat() + "Z",
                "kind": "curiosity_feedback",
                "version": "1",
                "project_id": case.project_id,
            },
            "case": {
                "case_uuid": str(case.case_uuid),
                "dataset_digest": str(case.dataset_digest),
                "target_kind": str(case.target_kind),
                "target_col": str(case.target_col),
                "error_kind": str(case.error_kind),
                "error_value": float(case.error_value),
            },
            "feedback": {
                # Do NOT export freeform answer text (could contain sensitive info).
                "tags": _loads_list(ans.tags_json)[:20],
                "has_corrected_target": (str(ans.corrected_target_json or "null") != "null"),
            },
        }

        queue_outbox_message(
            session,
            user_id=int(ans.created_by_user_id),
            project_id=case.project_id,
            device_id=str(getattr(settings, "nexus_device_id", "local") or "local"),
            kind="curiosity_feedback",
            payload=payload,
        )
        ans.exported_to_hive_at = datetime.utcnow()
        session.add(ans)
        session.commit()
    except Exception:
        return


def curiosity_export_summary(session: Session, *, user_id: int, project_id: int | None, window_days: int = 30) -> dict[str, Any]:
    """Return non-sensitive aggregates useful for Hive or dashboards."""

    since = datetime.utcnow() - timedelta(days=max(1, min(int(window_days), 365)))

    q = select(CuriosityAnswer).where(CuriosityAnswer.created_by_user_id == int(user_id)).where(CuriosityAnswer.created_at >= since)
    if project_id is None:
        q = q.where(CuriosityAnswer.project_id.is_(None))
    else:
        q = q.where(CuriosityAnswer.project_id == int(project_id))

    rows = session.exec(q).all()
    tags: dict[str, int] = {}
    n_corrected = 0
    for r in rows:
        for t in _loads_list(r.tags_json):
            ts = str(t).strip()
            if not ts:
                continue
            tags[ts] = int(tags.get(ts, 0)) + 1
        if str(r.corrected_target_json or "null") != "null":
            n_corrected += 1

    return {
        "window_days": int(window_days),
        "n_answers": int(len(rows)),
        "n_corrected_targets": int(n_corrected),
        "top_tags": sorted([{"tag": k, "count": v} for k, v in tags.items()], key=lambda x: x["count"], reverse=True)[:20],
    }
