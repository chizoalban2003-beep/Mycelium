"""Learning daemon — the heartbeat of the digital organism.

This module implements two background loops:

1. **Signal Collector Loop** — collects OS-level signals via psutil every
   N seconds and stores them as SignalLedgerEvent rows.

2. **Learning Loop** — periodically builds a tabular ecosystem from recent
   signals, runs sedimentation (always) and physics prediction (when growth
   stage permits), updates the growth ledger, and generates narrative nudges.

The learning loop is growth-stage-aware:
    - **Infant** — observe only; run sedimentation to map the ecosystem
    - **Toddler** — begin supervised prediction on organic targets
    - **Adolescent+** — full prediction with ledger recall, proactive nudges

Both loops degrade gracefully: if any subsystem fails, the daemon continues.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from mycelium_app.db import engine
from mycelium_app.ecosystem_bridge import build_ecosystem_dataframe, build_ecosystem_summary
from mycelium_app.models import GrowthLedgerEntry, NexusNudge, SignalLedgerEvent
from mycelium_app.settings import settings
from mycelium_app.signal_collector import CollectorState, collect_all_signals
from mycelium_app.stimulus import record_stimulus_event


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Signal collection tick
# ---------------------------------------------------------------------------

def run_signal_collection_tick(
    state: CollectorState,
    *,
    user_id: int,
    device_id: str = "local",
) -> int:
    """Collect OS signals and store them. Returns number of signals stored."""
    signals = collect_all_signals(state)
    if not signals:
        return 0

    count = 0
    with Session(engine) as session:
        for sig in signals:
            try:
                record_stimulus_event(
                    session,
                    user_id=user_id,
                    project_id=None,
                    device_id=device_id,
                    source=str(sig.get("source", "os")),
                    modality=str(sig.get("modality", "telemetry")),
                    signal_type=str(sig.get("signal_type", "unknown")),
                    stimulus=sig.get("stimulus", {}),
                    occurred_at=datetime.utcnow(),
                )
                count += 1
            except Exception:
                continue
    return count


# ---------------------------------------------------------------------------
# Learning tick
# ---------------------------------------------------------------------------

def run_learning_tick(
    *,
    user_id: int,
    device_id: str = "local",
    window_hours: int = 6,
    bucket_minutes: int = 30,
) -> dict[str, Any]:
    """Execute one learning cycle. Returns a summary dict."""
    from mycelium_app.growth import compute_growth_stage
    from mycelium_app.sedimentation import run_sedimentation
    from mycelium_app.narrative import generate_ecosystem_narrative

    result: dict[str, Any] = {
        "ok": False,
        "stage": "infant",
        "actions": [],
        "sedimentation": None,
        "prediction": None,
        "narrative": None,
    }

    with Session(engine) as session:
        # 1. Determine growth stage
        stage, unlocked, stats = compute_growth_stage(
            session, user_id=user_id, project_id=None,
        )
        result["stage"] = stage
        result["unlocked"] = unlocked

        # 2. Build ecosystem DataFrame from recent signals
        df = build_ecosystem_dataframe(
            session,
            user_id=user_id,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
        )

        if df.empty or df.shape[1] < 3:
            result["actions"].append("skipped: insufficient signals")
            result["ok"] = True
            return result

        # 3. Always run sedimentation (unsupervised ecosystem map)
        try:
            sed_result = run_sedimentation(df, flocculation_threshold=0.7)
            result["sedimentation"] = {
                "n_features": sed_result.n_features,
                "n_complexes": len(sed_result.complexes),
                "layers": {
                    layer: info["count"]
                    for layer, info in sed_result.layer_summary.items()
                },
                "top_bedrock": [
                    f.feature for f in sed_result.features
                    if f.layer == "bedrock"
                ][:5],
            }
            result["actions"].append("sedimentation_complete")
        except Exception as e:
            result["actions"].append(f"sedimentation_error: {type(e).__name__}")

        # 3b. Compute force field (continuous time evolution)
        try:
            from mycelium_app.force_field import (
                compute_force_field, save_field_snapshot, load_previous_field,
                serialize_force_field,
            )

            # Load previous field for momentum continuity
            prev_field = load_previous_field(user_id=user_id)

            # Build signal list from recent events
            from mycelium_app.models import SignalLedgerEvent as SLE
            from sqlmodel import select as sel
            since_ff = datetime.utcnow() - timedelta(hours=window_hours)
            sig_rows = session.exec(
                sel(SLE)
                .where(SLE.created_by_user_id == user_id, SLE.created_at >= since_ff)
                .order_by(SLE.created_at)
            ).all()

            ff_signals = []
            for r in sig_rows:
                try:
                    payload = json.loads(r.payload_json or "{}")
                except Exception:
                    payload = {}
                surface = payload.get("surface") or payload.get("stimulus") or payload
                ff_signals.append({
                    "signal_type": str(r.signal_type or ""),
                    "app_name": str(surface.get("app_name", r.signal_type or "")),
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "session_seconds": surface.get("session_seconds", 0),
                    "payload": surface,
                })

            if ff_signals:
                ff_state = compute_force_field(ff_signals, window_hours=window_hours)

                # Apply previous momentum if available
                if prev_field and prev_field.get("particles"):
                    prev_particles = {p["name"]: p for p in prev_field["particles"]}
                    for p in ff_state.particles:
                        prev = prev_particles.get(p.name)
                        if prev:
                            p.vx = p.vx * 0.5 + float(prev.get("vx", 0)) * 0.5
                            p.vy_vel = p.vy_vel * 0.5 + float(prev.get("vy", 0)) * 0.5
                            p.vz = p.vz * 0.5 + float(prev.get("vz", 0)) * 0.5

                # Save snapshot for next cycle's continuity
                save_field_snapshot(ff_state, user_id=user_id)

                result["force_field"] = {
                    "n_particles": len(ff_state.particles),
                    "n_bonds": ff_state.n_bonds,
                    "total_energy": ff_state.total_energy,
                    "agent_stage": ff_state.agent.stage,
                    "agent_coherence": ff_state.agent.coherence,
                    "agent_crystallized": ff_state.agent.crystallized,
                    "dominant_force": max(
                        ff_state.forces_applied,
                        key=ff_state.forces_applied.get,
                        default="",
                    ) if ff_state.forces_applied else "",
                }
                result["actions"].append("force_field_computed")

                # Use force field agent stage if more advanced than growth stage
                ff_stage = ff_state.agent.stage
                stage_rank = {"infant": 0, "toddler": 1, "adolescent": 2, "adult": 3}
                if stage_rank.get(ff_stage, 0) > stage_rank.get(stage, 0):
                    stage = ff_stage
                    result["stage"] = stage

        except Exception as e:
            result["actions"].append(f"force_field_error: {type(e).__name__}: {e}")

        # 4. Supervised prediction (toddler+ only)
        if stage in ("toddler", "adolescent") and df.shape[0] >= 6:
            try:
                from mycelium_app.physics_predictor import (
                    PhysicsPlane,
                    run_physics_prediction,
                    infer_target_kind,
                )

                # Use cpu_mean as an organic target (predicting resource usage
                # from behavioral patterns) — available in all ecosystems
                target_candidates = ["cpu_mean", "memory_mean", "n_signals", "context_switches"]
                target_col = None
                for tc in target_candidates:
                    if tc in df.columns and df[tc].nunique() >= 3:
                        target_col = tc
                        break

                if target_col:
                    # Use unified field to derive predictor kwargs
                    pred_kwargs = {
                        "target_col": target_col,
                        "train_fraction": 0.7,
                        "random_seed": 42,
                        "top_k_weights": min(20, df.shape[1] - 1),
                        "plane": PhysicsPlane.liquid,
                        "n_cycles": 15,
                    }
                    try:
                        from mycelium_app.unified_field import field_to_predictor_kwargs
                        if 'ff_state' in dir() and ff_state:
                            pred_kwargs = field_to_predictor_kwargs(ff_state, base_kwargs=pred_kwargs)
                            result["actions"].append("unified_field_applied")
                    except Exception:
                        pass

                    pred = run_physics_prediction(df, **pred_kwargs)

                    if pred and pred.metrics:
                        r2 = None
                        if pred.test_actual and pred.test_predicted:
                            pairs = []
                            for a, b in zip(pred.test_actual, pred.test_predicted):
                                try:
                                    af, bf = float(a), float(b)
                                    if not (af != af or bf != bf):
                                        pairs.append((af, bf))
                                except Exception:
                                    continue
                            if len(pairs) >= 2:
                                y_bar = sum(p[0] for p in pairs) / len(pairs)
                                ss_res = sum((a - b) ** 2 for a, b in pairs)
                                ss_tot = sum((a - y_bar) ** 2 for a, _ in pairs)
                                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                        result["prediction"] = {
                            "target": target_col,
                            "target_kind": str(pred.target_kind),
                            "n_features_used": pred.metrics.n_features_used,
                            "mae": pred.metrics.mae,
                            "rmse": pred.metrics.rmse,
                            "r2": round(r2, 4) if r2 is not None else None,
                            "best_cycle": pred.metrics.best_cycle,
                            "top_weights": [
                                {"feature": w.feature, "weight": round(w.weight, 4)}
                                for w in (pred.weights or [])[:5]
                            ],
                        }
                        result["actions"].append("prediction_complete")

                        # Record in growth ledger
                        entry = GrowthLedgerEntry(
                            created_by_user_id=user_id,
                            project_id=None,
                            domain="ecosystem_learning",
                            metric="r2" if r2 is not None else "mae",
                            score=float(r2 if r2 is not None else (pred.metrics.mae or 0)),
                            accepted=bool(r2 is not None and r2 > 0.1),
                            proposal_json=_dumps({
                                "target": target_col,
                                "plane": "liquid",
                                "n_cycles": 15,
                                "n_features": df.shape[1] - 1,
                            }),
                            outcome_json=_dumps(result["prediction"]),
                        )
                        session.add(entry)
                        session.commit()
                        result["actions"].append("growth_recorded")

            except Exception as e:
                result["actions"].append(f"prediction_error: {type(e).__name__}")

        # 4b. Auto-tune force constants (adolescent+ or after prediction)
        if target_col and df.shape[0] >= 6:
            try:
                from mycelium_app.auto_tune import (
                    auto_tune_constants, save_tuned_constants,
                    load_tuned_constants, TunedConstants,
                )

                tc = load_tuned_constants(user_id=user_id)
                if tc is None:
                    tc = TunedConstants()

                # Only tune if we have enough signals
                if 'ff_signals' in dir() and ff_signals and len(ff_signals) >= 10:
                    tc = auto_tune_constants(
                        df, ff_signals, target_col, tc,
                        window_hours=window_hours,
                    )
                    save_tuned_constants(tc, user_id=user_id)

                    result["auto_tune"] = {
                        "generation": tc.generation,
                        "constants": {"G": round(tc.G, 4), "K_E": round(tc.K_E, 4),
                                      "K_S": round(tc.K_S, 4), "K_W": round(tc.K_W, 4)},
                        "last_mae": round(tc.last_mae, 6) if tc.last_mae else None,
                    }
                    result["actions"].append("auto_tune_complete")
            except Exception as e:
                result["actions"].append(f"auto_tune_error: {type(e).__name__}")

        # 4c. Record time series data point
        try:
            from mycelium_app.trend_analysis import record_ecosystem_tick
            mae_val = None
            if result.get("prediction") and result["prediction"].get("mae") is not None:
                mae_val = float(result["prediction"]["mae"])
            record_ecosystem_tick(
                session,
                user_id=user_id,
                field_state=ff_state if 'ff_state' in dir() else None,
                sedimentation=result.get("sedimentation"),
                n_signals=df.shape[0] if not df.empty else 0,
                mae=mae_val,
            )
            result["actions"].append("timeseries_recorded")
        except Exception as e:
            result["actions"].append(f"timeseries_error: {type(e).__name__}")

        # 4d. Check Hive readiness (nudge after ~7 days)
        try:
            from mycelium_app.hive_readiness import maybe_nudge_hive_readiness
            if maybe_nudge_hive_readiness(session, user_id=user_id):
                result["actions"].append("hive_readiness_nudge")
        except Exception:
            pass

        # 5. Generate narrative
        try:
            summary = build_ecosystem_summary(
                session, user_id=user_id, window_hours=window_hours,
            )
            narrative = generate_ecosystem_narrative(
                stage=stage,
                summary=summary,
                sedimentation=result.get("sedimentation"),
                prediction=result.get("prediction"),
            )
            result["narrative"] = narrative

            # Create a nudge if the narrative has content
            if narrative and narrative.get("headline"):
                recent_nudge = session.exec(
                    select(NexusNudge)
                    .where(NexusNudge.created_by_user_id == user_id)
                    .where(NexusNudge.kind == "ecosystem_learning")
                    .where(NexusNudge.created_at >= datetime.utcnow() - timedelta(hours=2))
                    .order_by(NexusNudge.created_at.desc())
                    .limit(1)
                ).first()

                if recent_nudge is None:
                    nudge = NexusNudge(
                        created_by_user_id=user_id,
                        project_id=None,
                        kind="ecosystem_learning",
                        title=str(narrative.get("headline", "Ecosystem update")),
                        message=str(narrative.get("body", "")),
                        payload_json=_dumps({
                            "stage": stage,
                            "summary": summary,
                            "sedimentation_layers": result.get("sedimentation", {}).get("layers"),
                            "prediction_r2": (result.get("prediction") or {}).get("r2"),
                        }),
                    )
                    session.add(nudge)
                    session.commit()
                    result["actions"].append("nudge_created")

        except Exception as e:
            result["actions"].append(f"narrative_error: {type(e).__name__}")

    result["ok"] = True
    return result
