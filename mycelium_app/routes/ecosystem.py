"""API routes for the living ecosystem — signal collection, learning, and narrative."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.ecosystem_bridge import build_ecosystem_dataframe, build_ecosystem_summary
from mycelium_app.growth import compute_growth_stage
from mycelium_app.learning_daemon import run_learning_tick, run_signal_collection_tick
from mycelium_app.models import User
from mycelium_app.narrative import generate_ecosystem_narrative
from mycelium_app.sedimentation import run_sedimentation
from mycelium_app.signal_collector import CollectorState
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/ecosystem", tags=["ecosystem"])

_shared_collector_state = CollectorState()


@router.post("/collect")
def collect_signals(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Trigger one signal collection tick (for manual testing or on-demand use)."""
    n = run_signal_collection_tick(
        _shared_collector_state,
        user_id=int(current_user.id or 0),
        device_id=str(settings.nexus_device_id or "local"),
    )
    return {"ok": True, "signals_collected": n, "tick": _shared_collector_state.tick_count}


@router.post("/learn")
def trigger_learning(
    window_hours: int = 6,
    bucket_minutes: int = 30,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Trigger one learning cycle (for manual testing or on-demand use)."""
    result = run_learning_tick(
        user_id=int(current_user.id or 0),
        device_id=str(settings.nexus_device_id or "local"),
        window_hours=max(1, min(window_hours, 168)),
        bucket_minutes=max(5, min(bucket_minutes, 120)),
    )
    return result


@router.get("/state")
def ecosystem_state(
    window_hours: int = 24,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get the current ecosystem state — summary, stage, sedimentation, narrative."""
    user_id = int(current_user.id or 0)

    stage, unlocked, growth_stats = compute_growth_stage(
        session, user_id=user_id, project_id=None,
    )

    summary = build_ecosystem_summary(
        session, user_id=user_id, window_hours=window_hours,
    )

    # Build ecosystem DataFrame and run sedimentation
    df = build_ecosystem_dataframe(
        session, user_id=user_id, window_hours=window_hours, bucket_minutes=30,
    )

    sed_info = None
    if not df.empty and df.shape[1] >= 3:
        try:
            sed_result = run_sedimentation(df, flocculation_threshold=0.7)
            sed_info = {
                "n_features": sed_result.n_features,
                "n_complexes": len(sed_result.complexes),
                "layers": {
                    layer: info["count"]
                    for layer, info in sed_result.layer_summary.items()
                },
                "top_bedrock": [
                    f.feature for f in sed_result.features if f.layer == "bedrock"
                ][:5],
                "top_turbulent": [
                    f.feature for f in sed_result.features if f.layer == "turbulent"
                ][:5],
            }
        except Exception:
            pass

    narrative = generate_ecosystem_narrative(
        stage=stage,
        summary=summary,
        sedimentation=sed_info,
        prediction=None,
    )

    return {
        "ok": True,
        "stage": stage,
        "unlocked": unlocked,
        "growth_stats": growth_stats,
        "summary": summary,
        "sedimentation": sed_info,
        "narrative": narrative,
        "ecosystem_shape": {
            "rows": df.shape[0] if not df.empty else 0,
            "columns": df.shape[1] if not df.empty else 0,
        },
    }


@router.get("/dataframe")
def ecosystem_dataframe(
    window_hours: int = 6,
    bucket_minutes: int = 30,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Return the raw ecosystem DataFrame as JSON (for debugging/visualization)."""
    df = build_ecosystem_dataframe(
        session,
        user_id=int(current_user.id or 0),
        window_hours=max(1, min(window_hours, 168)),
        bucket_minutes=max(5, min(bucket_minutes, 120)),
    )

    if df.empty:
        return {"ok": True, "rows": 0, "columns": 0, "data": [], "column_names": []}

    return {
        "ok": True,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "data": df.to_dict(orient="records"),
    }
