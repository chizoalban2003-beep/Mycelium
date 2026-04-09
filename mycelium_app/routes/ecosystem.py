"""API routes for the living ecosystem — signal collection, learning, and narrative."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.ecosystem_bridge import build_ecosystem_dataframe, build_ecosystem_summary
from mycelium_app.growth import compute_growth_stage
from mycelium_app.learning_daemon import run_learning_tick, run_signal_collection_tick
from mycelium_app.humanizer import humanize_app, humanize_apps_dict, humanize_feature, humanize_layer, humanize_signal
from mycelium_app.jarvis import chat as jarvis_chat
from mycelium_app.models import User
from mycelium_app.pattern_engine import analyze_patterns, generate_proactive_suggestions
from mycelium_app.narrative import generate_ecosystem_narrative
from mycelium_app.sedimentation import run_sedimentation
from mycelium_app.signal_collector import CollectorState
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/ecosystem", tags=["ecosystem"])

_shared_collector_state = CollectorState()
_last_sedimentation_cache: dict[int, dict] = {}  # user_id → last result


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


@router.get("/live")
def live_ecosystem(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Real-time ecosystem snapshot optimized for the live canvas.

    Returns current signals, sedimentation layers, agent state, and
    graph data in a single payload for efficient polling.
    """
    from mycelium_app.force_graph import build_sedimentation_graph
    from mycelium_app.self_reflection import compute_self_reflection
    from mycelium_app.assistant_profile import get_assistant_profile_effective

    user_id = int(current_user.id or 0)

    # Growth stage
    stage, unlocked, _ = compute_growth_stage(session, user_id=user_id, project_id=None)

    # Recent signals (last 5 minutes for live view)
    from datetime import datetime, timedelta
    since = datetime.utcnow() - timedelta(minutes=5)
    from mycelium_app.models import SignalLedgerEvent as SLE
    recent = session.exec(
        select(SLE)
        .where(SLE.created_by_user_id == user_id, SLE.created_at >= since)
        .order_by(SLE.created_at.desc())
        .limit(20)
    ).all()

    live_signals = []
    for r in recent:
        live_signals.append({
            "type": humanize_signal(r.signal_type),
            "raw_type": str(r.signal_type or ""),
            "device": str(r.device_id or ""),
            "at": r.created_at.isoformat() if r.created_at else "",
        })

    # Ecosystem summary
    summary = build_ecosystem_summary(session, user_id=user_id, window_hours=1)

    # Sedimentation + graph (use cached or recompute)
    graph_data = None
    sed_layers = None
    sed_features = None
    df = build_ecosystem_dataframe(session, user_id=user_id, window_hours=6, bucket_minutes=15)
    if not df.empty and df.shape[1] >= 3:
        try:
            sed = run_sedimentation(df, flocculation_threshold=0.7)
            graph_data = build_sedimentation_graph(sed)
            sed_layers = {k: v["count"] for k, v in sed.layer_summary.items()}
            sed_features = [
                {
                    "feature": humanize_feature(f.feature),
                    "raw_feature": f.feature,
                    "depth": f.depth,
                    "layer": f.layer,
                    "layer_label": humanize_layer(f.layer),
                    "density": f.density,
                    "velocity": f.settling_velocity,
                    "complex_id": f.complex_id,
                }
                for f in sed.features[:30]
            ]
        except Exception:
            pass

    # Agent state (reflection + profile)
    agent_state = {"mood": "curious", "identity_hash": "", "display_name": "Myco"}
    try:
        reflection = compute_self_reflection(session, user_id=user_id)
        agent_state["mood"] = str(getattr(reflection, "mood", "curious"))
        agent_state["identity_hash"] = str(getattr(reflection, "identity_hash", ""))
    except Exception:
        pass

    profile = get_assistant_profile_effective(session, user_id=user_id, project_id=None)
    agent_state["given_name"] = str(profile.get("given_name", "Myco"))
    agent_state["gender"] = str(profile.get("gender_identity", "neutral"))

    # Narrative
    from mycelium_app.narrative import generate_ecosystem_narrative
    narrative = generate_ecosystem_narrative(
        stage=stage, summary=summary,
        sedimentation={"layers": sed_layers, "top_bedrock": [], "n_complexes": 0} if sed_layers else None,
    )

    return {
        "ok": True,
        "ts": datetime.utcnow().isoformat(),
        "stage": stage,
        "unlocked": unlocked,
        "agent": agent_state,
        "narrative": narrative,
        "summary": {
            "n_signals": summary.get("n_signals", 0),
            "top_apps": humanize_apps_dict(summary.get("top_apps", {})),
            "cpu_mean": summary.get("cpu_mean"),
            "battery_mean": summary.get("battery_mean"),
        },
        "live_signals": live_signals,
        "sedimentation": {
            "layers": {humanize_layer(k): v for k, v in sed_layers.items()} if sed_layers else None,
            "layers_raw": sed_layers,
            "features": sed_features,
        } if sed_layers else None,
        "graph": _humanize_graph(graph_data) if graph_data else None,
    }


def _humanize_graph(graph: dict) -> dict:
    """Add human-friendly labels to graph nodes."""
    if not graph:
        return graph
    for node in graph.get("nodes", []):
        node["label"] = humanize_feature(node.get("id", ""))
        node["layer_label"] = humanize_layer(node.get("layer", ""))
    return graph


@router.get("/patterns")
def get_patterns(
    window_hours: int = 48,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get detected behavioral patterns and proactive suggestions."""
    from mycelium_app.growth import compute_growth_stage

    user_id = int(current_user.id or 0)
    stage, _, _ = compute_growth_stage(session, user_id=user_id, project_id=None)

    result = analyze_patterns(session, user_id=user_id, window_hours=window_hours)
    suggestions = generate_proactive_suggestions(result.get("patterns", []), stage=stage)
    result["suggestions"] = suggestions
    return result


from fastapi import Body

@router.post("/chat")
def ecosystem_chat(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """JARVIS-like chat — talk to your companion with full context."""
    from mycelium_app.growth import compute_growth_stage
    from mycelium_app.self_reflection import compute_self_reflection
    from mycelium_app.assistant_profile import get_assistant_profile_effective

    user_id = int(current_user.id or 0)
    message = str(payload.get("message", "")).strip()
    if not message:
        return {"ok": False, "reply": "I didn't catch that. Try again?"}

    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []

    # Gather context
    stage, _, _ = compute_growth_stage(session, user_id=user_id, project_id=None)
    profile = get_assistant_profile_effective(session, user_id=user_id, project_id=None)
    agent_name = str(profile.get("given_name", "Myco"))
    gender = str(profile.get("gender_identity", "neutral"))

    mood = "curious"
    try:
        reflection = compute_self_reflection(session, user_id=user_id)
        mood = str(getattr(reflection, "mood", "curious"))
    except Exception:
        pass

    # Live ecosystem state
    summary = build_ecosystem_summary(session, user_id=user_id, window_hours=6)
    eco_state = {"summary": humanize_apps_dict(summary.get("top_apps", {})) if summary.get("top_apps") else {}}
    eco_state["summary"] = {"n_signals": summary.get("n_signals", 0), "top_apps": humanize_apps_dict(summary.get("top_apps", {})), "cpu_mean": summary.get("cpu_mean"), "battery_mean": summary.get("battery_mean")}

    # Sedimentation
    df = build_ecosystem_dataframe(session, user_id=user_id, window_hours=6, bucket_minutes=15)
    if not df.empty and df.shape[1] >= 3:
        try:
            sed = run_sedimentation(df, flocculation_threshold=0.7)
            eco_state["sedimentation"] = {
                "layers_raw": {k: v["count"] for k, v in sed.layer_summary.items()},
                "features": [{"feature": humanize_feature(f.feature)} for f in sed.features[:10]],
            }
        except Exception:
            pass

    # Patterns
    pat_result = analyze_patterns(session, user_id=user_id, window_hours=24)
    suggestions = generate_proactive_suggestions(pat_result.get("patterns", []), stage=stage)
    pat_result["suggestions"] = suggestions

    reply = jarvis_chat(
        message,
        ecosystem=eco_state,
        patterns=pat_result,
        stage=stage,
        mood=mood,
        agent_name=agent_name,
        gender=gender,
        conversation_history=history,
    )

    return {
        "ok": True,
        "reply": reply,
        "agent": {"name": agent_name, "stage": stage, "mood": mood, "gender": gender},
    }
