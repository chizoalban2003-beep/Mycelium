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
                    pred = run_physics_prediction(
                        df,
                        target_col=target_col,
                        plane=PhysicsPlane.liquid,
                        train_fraction=0.7,
                        random_seed=42,
                        n_cycles=15,
                        top_k_weights=min(20, df.shape[1] - 1),
                    )

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
