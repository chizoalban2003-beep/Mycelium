"""API routes for the living ecosystem — signal collection, learning, and narrative."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.ecosystem_bridge import build_ecosystem_dataframe, build_ecosystem_summary
from mycelium_app.growth import compute_growth_stage
from mycelium_app.learning_daemon import run_learning_tick, run_signal_collection_tick
from mycelium_app.force_field import compute_force_field, serialize_force_field
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
        "force_field": _build_live_force_field(session, user_id),
    }


def _build_live_force_field(session, user_id: int) -> dict | None:
    """Build force field from recent signals for the live view."""
    try:
        from datetime import datetime as dt, timedelta as td
        since = dt.utcnow() - td(hours=6)
        from mycelium_app.models import SignalLedgerEvent as SLE
        rows = session.exec(
            select(SLE)
            .where(SLE.created_by_user_id == int(user_id), SLE.created_at >= since)
            .order_by(SLE.created_at)
        ).all()
        if not rows:
            return None

        import json as _json
        signals = []
        for r in rows:
            payload = {}
            try:
                payload = _json.loads(r.payload_json or "{}")
            except Exception:
                pass
            surface = payload.get("surface") or payload.get("stimulus") or payload
            signals.append({
                "signal_type": str(r.signal_type or ""),
                "app_name": str(surface.get("app_name", r.signal_type or "")),
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "session_seconds": surface.get("session_seconds", 0),
                "payload": surface,
            })

        ff = compute_force_field(signals, window_hours=6, n_iterations=20)
        serialized = serialize_force_field(ff)

        # Humanize particle names
        for p in serialized.get("particles", []):
            p["label"] = humanize_app(p.get("name", "")) or humanize_feature(p.get("name", ""))
        for b in serialized.get("bonds", []):
            b["source_label"] = humanize_app(b.get("source", ""))
            b["target_label"] = humanize_app(b.get("target", ""))

        return serialized
    except Exception:
        return None


def _humanize_graph(graph: dict) -> dict:
    """Add human-friendly labels to graph nodes."""
    if not graph:
        return graph
    for node in graph.get("nodes", []):
        node["label"] = humanize_feature(node.get("id", ""))
        node["layer_label"] = humanize_layer(node.get("layer", ""))
    return graph


@router.get("/field")
def force_field_state(
    window_hours: int = 6,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get the raw force field state — particles, forces, bonds, agent waveform."""
    ff = _build_live_force_field(session, int(current_user.id or 0))
    if ff is None:
        return {"ok": True, "force_field": None, "message": "No signals yet"}
    return {"ok": True, "force_field": ff}


@router.get("/anomalies")
def get_anomalies(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Detect anomalies by comparing current field to previous snapshot."""
    from mycelium_app.unified_field import detect_anomalies
    from mycelium_app.force_field import load_previous_field

    ff_data = _build_live_force_field(session, int(current_user.id or 0))
    if not ff_data:
        return {"ok": True, "anomalies": [], "message": "Not enough data"}

    prev = load_previous_field(user_id=int(current_user.id or 0))

    # Rebuild ForceFieldState from API data for anomaly detection
    from mycelium_app.force_field import compute_force_field
    import json as _json
    from datetime import datetime as dt, timedelta as td
    since = dt.utcnow() - td(hours=6)
    from mycelium_app.models import SignalLedgerEvent as SLE
    rows = session.exec(
        select(SLE).where(SLE.created_by_user_id == int(current_user.id or 0), SLE.created_at >= since)
    ).all()
    signals = []
    for r in rows:
        try:
            payload = _json.loads(r.payload_json or "{}")
        except Exception:
            payload = {}
        surface = payload.get("surface") or payload.get("stimulus") or payload
        signals.append({
            "signal_type": str(r.signal_type or ""),
            "app_name": str(surface.get("app_name", r.signal_type or "")),
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "payload": surface,
        })

    ff = compute_force_field(signals, window_hours=6, n_iterations=15)
    anomalies = detect_anomalies(ff, prev)

    return {"ok": True, "anomalies": anomalies, "n_anomalies": len(anomalies)}


@router.get("/digest")
def weekly_digest(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Generate a weekly ecosystem summary digest."""
    from mycelium_app.unified_field import generate_weekly_digest
    from mycelium_app.force_field import compute_force_field
    from mycelium_app.assistant_profile import get_assistant_profile_effective
    import json as _json
    from datetime import datetime as dt, timedelta as td

    user_id = int(current_user.id or 0)
    since = dt.utcnow() - td(hours=168)
    from mycelium_app.models import SignalLedgerEvent as SLE
    rows = session.exec(
        select(SLE).where(SLE.created_by_user_id == user_id, SLE.created_at >= since)
    ).all()

    signals = []
    for r in rows:
        try:
            payload = _json.loads(r.payload_json or "{}")
        except Exception:
            payload = {}
        surface = payload.get("surface") or payload.get("stimulus") or payload
        signals.append({
            "signal_type": str(r.signal_type or ""),
            "app_name": str(surface.get("app_name", r.signal_type or "")),
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "payload": surface,
        })

    ff = compute_force_field(signals, window_hours=168, n_iterations=20)
    pats = analyze_patterns(session, user_id=user_id, window_hours=168)
    profile = get_assistant_profile_effective(session, user_id=user_id, project_id=None)

    digest = generate_weekly_digest(
        ff, pats, agent_name=str(profile.get("given_name", "Myco")),
    )
    return {"ok": True, "digest": digest}


@router.get("/predict-next-app")
def predict_next(
    current_app: str = "unknown",
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Predict the next app the user will open."""
    from mycelium_app.unified_field import predict_next_app
    from mycelium_app.humanizer import humanize_app
    from datetime import datetime as dt, timedelta as td
    import json as _json

    since = dt.utcnow() - td(hours=24)
    from mycelium_app.models import SignalLedgerEvent as SLE
    rows = session.exec(
        select(SLE).where(
            SLE.created_by_user_id == int(current_user.id or 0),
            SLE.signal_type == "app_focus",
            SLE.created_at >= since,
        ).order_by(SLE.created_at)
    ).all()

    transitions = []
    for r in rows:
        try:
            payload = _json.loads(r.payload_json or "{}")
        except Exception:
            continue
        surface = payload.get("surface") or payload.get("stimulus") or payload
        app = str(surface.get("app_name", "")).lower()[:32]
        prev = str(surface.get("previous_app", "")).lower()[:32]
        if app and prev and app != prev:
            transitions.append((prev, app))

    result = predict_next_app(transitions, current_app.lower(), dt.utcnow().hour)

    if result.get("prediction"):
        result["prediction_label"] = humanize_app(result["prediction"])
        result["alternatives"] = [
            {**a, "label": humanize_app(a["app"])} for a in result.get("alternatives", [])
        ]

    return {"ok": True, **result}


@router.get("/health")
def ecosystem_health(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Observability: reports daemon status, DB stats, and system health."""
    from datetime import datetime as dt, timedelta as td
    from mycelium_app.models import SignalLedgerEvent as SLE, ForceFieldSnapshot, GrowthLedgerEntry, NexusNudge
    from mycelium_app.auto_tune import load_tuned_constants
    from mycelium_app.trend_analysis import compute_trends

    uid = int(current_user.id or 0)
    now = dt.utcnow()

    # Last signal
    last_signal = session.exec(
        select(SLE).where(SLE.created_by_user_id == uid).order_by(SLE.created_at.desc()).limit(1)
    ).first()

    # Last field snapshot
    last_snapshot = session.exec(
        select(ForceFieldSnapshot).where(ForceFieldSnapshot.user_id == uid).order_by(ForceFieldSnapshot.created_at.desc()).limit(1)
    ).first()

    # Last growth entry
    last_growth = session.exec(
        select(GrowthLedgerEntry).where(GrowthLedgerEntry.created_by_user_id == uid).order_by(GrowthLedgerEntry.created_at.desc()).limit(1)
    ).first()

    # Counts
    signal_count = len(session.exec(select(SLE).where(SLE.created_by_user_id == uid)).all())
    nudge_count = len(session.exec(select(NexusNudge).where(NexusNudge.created_by_user_id == uid)).all())

    # Tuned constants
    tc = load_tuned_constants(user_id=uid)

    # Trends (brief)
    try:
        trends = compute_trends(session, user_id=uid, window_hours=24)
        trend_summary = trends.get("trends", {}).get("coherence", {}).get("direction", "unknown")
    except Exception:
        trend_summary = "unavailable"

    def _age(ts):
        if not ts: return None
        delta = now - ts
        return f"{int(delta.total_seconds())}s ago"

    return {
        "ok": True,
        "user_id": uid,
        "daemons": {
            "last_signal_collection": _age(last_signal.created_at if last_signal else None),
            "last_force_field": _age(last_snapshot.created_at if last_snapshot else None),
            "last_growth_entry": _age(last_growth.created_at if last_growth else None),
        },
        "counts": {
            "total_signals": signal_count,
            "total_nudges": nudge_count,
        },
        "auto_tune": {
            "generation": tc.generation if tc else 0,
            "last_mae": round(tc.last_mae, 6) if tc and tc.last_mae else None,
            "constants": {"G": round(tc.G, 4), "K_E": round(tc.K_E, 4), "K_S": round(tc.K_S, 4), "K_W": round(tc.K_W, 4)} if tc else None,
        },
        "coherence_trend": trend_summary,
    }


@router.get("/trends")
def get_trends(
    window_hours: int = 168,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get ecosystem trend analysis over time."""
    from mycelium_app.trend_analysis import compute_trends
    return compute_trends(session, user_id=int(current_user.id or 0), window_hours=window_hours)


@router.get("/fast-predict")
def fast_predict(
    current_signal: str = "",
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Fast path prediction from force field bonds (<10ms)."""
    from mycelium_app.fast_predict import predict_from_bonds, classify_session_type
    from mycelium_app.force_field import compute_force_field
    import json as _json
    from datetime import datetime as dt, timedelta as td

    uid = int(current_user.id or 0)
    since = dt.utcnow() - td(hours=6)
    from mycelium_app.models import SignalLedgerEvent as SLE
    rows = session.exec(select(SLE).where(SLE.created_by_user_id == uid, SLE.created_at >= since)).all()

    signals = []
    recent_apps = []
    for r in rows:
        try: payload = _json.loads(r.payload_json or "{}")
        except: payload = {}
        surface = payload.get("surface") or payload.get("stimulus") or payload
        app = str(surface.get("app_name", r.signal_type or ""))
        signals.append({"signal_type": str(r.signal_type or ""), "app_name": app, "created_at": r.created_at.isoformat() if r.created_at else "", "payload": surface})
        if r.signal_type in ("app_focus", "app_open"):
            recent_apps.append(app.lower())

    ff = compute_force_field(signals, window_hours=6, n_iterations=10)
    prediction = predict_from_bonds(ff, current_signal or (recent_apps[-1] if recent_apps else ""))
    session_type = classify_session_type(ff, recent_apps[-5:] if recent_apps else [])

    return {"ok": True, "prediction": prediction, "session": session_type}


@router.post("/notify-test")
def test_desktop_notification(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Send a test desktop notification."""
    from mycelium_app.desktop_notify import send_desktop_notification
    ok = send_desktop_notification("Myco", "Your digital companion is alive and watching. 🌱")
    return {"ok": True, "sent": ok, "note": "Check your desktop for the notification"}


@router.post("/weekly-digest")
def send_weekly_digest(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Generate and deliver the weekly ecosystem digest."""
    from mycelium_app.weekly_digest import generate_and_deliver_digest
    result = generate_and_deliver_digest(session, user_id=int(current_user.id or 0))
    return result


@router.get("/tune")
def get_tuned_constants(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get the current auto-tuned force constants."""
    from mycelium_app.auto_tune import load_tuned_constants, TunedConstants
    from mycelium_app.force_field import _G, _K_E, _K_S, _K_W

    tc = load_tuned_constants(user_id=int(current_user.id or 0))
    if tc is None:
        return {
            "ok": True, "tuned": False,
            "constants": {"G": _G, "K_E": _K_E, "K_S": _K_S, "K_W": _K_W},
            "generation": 0, "last_mae": None,
        }
    return {
        "ok": True, "tuned": True,
        "constants": {"G": round(tc.G, 6), "K_E": round(tc.K_E, 6),
                      "K_S": round(tc.K_S, 6), "K_W": round(tc.K_W, 6)},
        "generation": tc.generation,
        "last_mae": round(tc.last_mae, 6) if tc.last_mae else None,
    }


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
