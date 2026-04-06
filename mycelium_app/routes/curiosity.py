from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from mycelium_app.curiosity import (
    answer_case,
    curiosity_export_summary,
    dismiss_case,
    list_recent_cases,
    next_pending_case,
)
from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import CuriosityCase, User
from mycelium_app.schemas import (
    CuriosityAnswerRequest,
    CuriosityAnswerResponse,
    CuriosityCaseListResponse,
    CuriosityCasePublic,
    CuriosityDismissRequest,
    CuriosityDismissResponse,
    CuriosityExportSummaryResponse,
)


router = APIRouter(prefix="/api/nexus/curiosity", tags=["curiosity"])


def _loads(s: str | None) -> object:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _to_public(row: CuriosityCase) -> CuriosityCasePublic:
    return CuriosityCasePublic(
        id=int(row.id or 0),
        created_at=row.created_at,
        project_id=row.project_id,
        status=str(row.status or ""),
        dataset_digest=str(row.dataset_digest or ""),
        target_col=str(row.target_col or ""),
        target_kind=str(row.target_kind or ""),
        row_index=row.row_index,
        error_kind=str(row.error_kind or ""),
        error_value=float(row.error_value or 0.0),
        predicted=_loads(row.predicted_json),
        actual=_loads(row.actual_json),
        excerpt=(_loads(row.excerpt_json) if isinstance(_loads(row.excerpt_json), dict) else {}),
        question=str(row.question or ""),
        answered_at=row.answered_at,
        dismissed_at=row.dismissed_at,
    )


@router.get("/next", response_model=CuriosityCasePublic)
def get_next(
    project_id: int | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    row = next_pending_case(session, user_id=int(current_user.id or 0), project_id=project_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No pending cases")
    return _to_public(row)


@router.get("/recent", response_model=CuriosityCaseListResponse)
def recent(
    project_id: int | None = None,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    rows = list_recent_cases(session, user_id=int(current_user.id or 0), project_id=project_id, limit=limit)
    return CuriosityCaseListResponse(cases=[_to_public(r) for r in rows])


@router.post("/answer", response_model=CuriosityAnswerResponse)
def answer(
    payload: CuriosityAnswerRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    try:
        ans_id = answer_case(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            case_id=int(payload.case_id),
            answer_text=str(payload.answer_text or ""),
            corrected_target=payload.corrected_target,
            tags=list(payload.tags or []),
            export_to_hive=bool(payload.export_to_hive),
        )
        return CuriosityAnswerResponse(ok=True, answer_id=int(ans_id))
    except ValueError as e:
        code = str(e)
        if code == "not_found":
            raise HTTPException(status_code=404, detail="Not found")
        if code == "not_pending":
            raise HTTPException(status_code=409, detail="Not pending")
        raise HTTPException(status_code=400, detail="Invalid")


@router.post("/dismiss", response_model=CuriosityDismissResponse)
def dismiss(
    payload: CuriosityDismissRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    try:
        dismiss_case(session, user_id=user_id, case_id=int(payload.case_id))
        return CuriosityDismissResponse(ok=True)
    except ValueError as e:
        if str(e) == "not_found":
            raise HTTPException(status_code=404, detail="Not found")
        raise HTTPException(status_code=400, detail="Invalid")


@router.get("/export/summary", response_model=CuriosityExportSummaryResponse)
def export_summary(
    project_id: int | None = None,
    window_days: int = 30,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    summary = curiosity_export_summary(session, user_id=int(current_user.id or 0), project_id=project_id, window_days=window_days)
    return CuriosityExportSummaryResponse(ok=True, summary=summary)
