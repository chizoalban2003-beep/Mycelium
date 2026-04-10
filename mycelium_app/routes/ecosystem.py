"""API routes for the living ecosystem — signal collection, learning, and narrative."""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean
import random
import json

from fastapi import APIRouter, Depends, Body
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.ecosystem_bridge import build_ecosystem_dataframe, build_ecosystem_summary
from mycelium_app.growth import compute_growth_stage
from mycelium_app.learning_daemon import run_learning_tick, run_signal_collection_tick
from mycelium_app.force_field import compute_force_field, serialize_force_field
from mycelium_app.humanizer import humanize_app, humanize_apps_dict, humanize_feature, humanize_layer, humanize_signal
from mycelium_app.jarvis import chat as jarvis_chat
from mycelium_app.models import (
    AutonomyPendingAction,
    AutonomyEpisode,
    AutonomyGenome,
    AutonomyGoalState,
    AutonomyLaw,
    AutonomyActionFeedback,
    EcosystemExperimentTick,
    EcosystemTimeSeries,
    User,
)
from mycelium_app.pattern_engine import analyze_patterns, generate_proactive_suggestions
from mycelium_app.narrative import generate_ecosystem_narrative
from mycelium_app.sedimentation import run_sedimentation
from mycelium_app.signal_collector import CollectorState
from mycelium_app.schemas import EcosystemExperimentRunRequest
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/ecosystem", tags=["ecosystem"])

_shared_collector_state = CollectorState()
_last_sedimentation_cache: dict[int, dict] = {}  # user_id → last result


def _bounded01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _autonomy_get_or_create_goal_state(session: Session, *, user_id: int) -> AutonomyGoalState:
    row = session.exec(
        select(AutonomyGoalState).where(AutonomyGoalState.user_id == int(user_id)).limit(1)
    ).first()
    if row:
        return row
    row = AutonomyGoalState(
        user_id=int(user_id),
        focus_progress=0.5,
        recovery_progress=0.5,
        novelty_progress=0.4,
        consistency_progress=0.45,
        last_7d_json=json.dumps(
            {"focus": [0.5], "recovery": [0.5], "novelty": [0.4], "consistency": [0.45]},
            sort_keys=True,
        ),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _autonomy_get_or_create_genome(session: Session, *, user_id: int) -> AutonomyGenome:
    row = session.exec(
        select(AutonomyGenome).where(AutonomyGenome.user_id == int(user_id)).limit(1)
    ).first()
    if row:
        return row
    row = AutonomyGenome(
        user_id=int(user_id),
        generation=0,
        fitness_score=0.0,
        weight_energy=0.8,
        weight_focus=1.0,
        weight_recovery=0.7,
        weight_novelty=0.6,
        mutation_rate=float(getattr(settings, "ecosystem_autonomy_mutation_rate", 0.08)),
        explore_bias=0.25,
        genome_json=json.dumps(
            {
                "action_bias": {"collect": 0.45, "learn": 0.35, "experiment": 0.20, "rest": 0.15},
                "selection_pressure": float(getattr(settings, "ecosystem_experiment_selection_pressure", 1.35)),
                "last_mutation": {},
            },
            sort_keys=True,
        ),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _autonomy_state_features(summary: dict, ts: EcosystemTimeSeries | None) -> dict:
    n_signals = float(summary.get("n_signals") or 0.0)
    cpu = float(summary.get("cpu_mean") or 0.0)
    battery = float(summary.get("battery_mean") or 0.0)
    signal_types = summary.get("signal_types") or {}
    diversity = float(len(signal_types.keys()))
    focus = _bounded01(min(1.0, n_signals / 120.0))
    energy = _bounded01((battery / 100.0) if battery > 0 else (1.0 - min(1.0, cpu / 100.0)))
    novelty = _bounded01(min(1.0, diversity / 8.0))
    coherence = _bounded01(float(getattr(ts, "coherence", 0.25) or 0.25))
    entropy = _bounded01(min(1.0, float(getattr(ts, "attention_entropy", 0.3) or 0.3)))
    recovery = _bounded01(1.0 - entropy)
    momentum = _bounded01(min(1.0, float(getattr(ts, "total_energy", 0.0) or 0.0) / 500.0))
    return {
        "focus": focus,
        "energy": energy,
        "novelty": novelty,
        "coherence": coherence,
        "entropy": entropy,
        "recovery": recovery,
        "momentum": momentum,
        "n_signals_window": int(n_signals),
    }


def _get_nudge_feedback_effect(session: Session, *, user_id: int, action_name: str) -> float:
    rows = session.exec(
        select(AutonomyActionFeedback)
        .where(
            AutonomyActionFeedback.user_id == int(user_id),
            AutonomyActionFeedback.action_name == str(action_name),
        )
        .order_by(AutonomyActionFeedback.created_at.desc())
        .limit(40)
    ).all()
    if not rows:
        return 0.0
    weighted = 0.0
    total_weight = 0.0
    for idx, row in enumerate(rows):
        w = 1.0 / (1.0 + idx * 0.25)
        decision = str(row.decision or "neutral")
        score = 0.5
        if decision in {"accepted", "approve"}:
            score = 0.7
        elif decision in {"rejected", "reject"}:
            score = 0.3
        weighted += score * w
        total_weight += w
    return _bounded01(weighted / max(0.0001, total_weight)) - 0.5


def _autonomy_expected_utility(features: dict, genome: AutonomyGenome, action: str, action_bias: dict) -> float:
    base = (
        float(genome.weight_focus) * float(features.get("focus", 0.0))
        + float(genome.weight_energy) * float(features.get("energy", 0.0))
        + float(genome.weight_recovery) * float(features.get("recovery", 0.0))
        + float(genome.weight_novelty) * float(features.get("novelty", 0.0))
    )
    base /= max(
        0.001,
        float(genome.weight_focus + genome.weight_energy + genome.weight_recovery + genome.weight_novelty),
    )
    action_term = float(action_bias.get(action, 0.2))
    if action == "experiment":
        action_term += float(features.get("novelty", 0.0)) * 0.20
    elif action == "learn":
        action_term += float(features.get("focus", 0.0)) * 0.15
    elif action == "rest":
        action_term += float(features.get("entropy", 0.0)) * 0.20
    elif action == "collect":
        action_term += (1.0 - float(features.get("focus", 0.0))) * 0.10
    return _bounded01((base * 0.75) + (action_term * 0.25))


def _autonomy_risk_score(features: dict, action: str) -> float:
    entropy = float(features.get("entropy", 0.3))
    energy = float(features.get("energy", 0.5))
    momentum = float(features.get("momentum", 0.2))
    risk = 0.0
    if action == "experiment":
        risk = 0.25 + entropy * 0.35 + momentum * 0.15 + (1.0 - energy) * 0.10
    elif action == "learn":
        risk = 0.18 + entropy * 0.30 + (1.0 - energy) * 0.12
    elif action == "collect":
        risk = 0.12 + (1.0 - energy) * 0.12
    else:  # rest
        risk = 0.06 + (1.0 - entropy) * 0.04
    return _bounded01(risk)


def _risk_level(risk: float) -> str:
    if risk < 0.35:
        return "safe"
    if risk < 0.65:
        return "caution"
    return "high"


def _counterfactual_v2(session: Session, *, user_id: int, action: str, expected_utility: float) -> float:
    rows = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.user_id == int(user_id), AutonomyEpisode.chosen_action == str(action))
        .order_by(AutonomyEpisode.created_at.desc())
        .limit(30)
    ).all()
    if not rows:
        return _bounded01(expected_utility * 0.94)
    observed = [float(r.observed_utility or 0.0) for r in rows]
    conf = [float(r.confidence or 0.0) for r in rows]
    trend = float(mean(observed)) if observed else 0.0
    certainty = float(mean(conf)) if conf else 0.5
    baseline = (_bounded01(expected_utility) * 0.45) + (trend * 0.45) + ((1.0 - certainty) * 0.10)
    return _bounded01(baseline)


def _adaptive_tick_minutes(session: Session, *, user_id: int) -> int:
    rows = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.user_id == int(user_id))
        .order_by(AutonomyEpisode.created_at.desc())
        .limit(24)
    ).all()
    base = int(getattr(settings, "ecosystem_autonomy_tick_minutes", 10))
    if not rows:
        return max(3, base)

    mean_conf = float(mean([float(r.confidence or 0.0) for r in rows]))
    mean_entropy = float(mean([float(r.attention_entropy or 0.0) for r in rows]))
    deltas: list[float] = []
    for row in rows:
        try:
            outcome = json.loads(row.outcome_json or "{}")
            deltas.append(float(outcome.get("delta", 0.0) or 0.0))
        except Exception:
            continue
    mean_delta = float(mean(deltas)) if deltas else 0.0

    if mean_conf > 0.72 and mean_entropy < 0.4 and mean_delta > 0.01:
        return max(3, base - 2)
    if mean_conf < 0.45 or mean_delta < -0.02:
        return min(45, base + 5)
    return base


def _autonomy_heat_state(
    session: Session,
    *,
    user_id: int,
    features_before: dict,
    features_after: dict,
    delta: float,
    confidence: float,
    risk_score: float,
    feedback_effect: float,
) -> dict:
    """Compute rolling heat budget used for exploration and governance gates."""
    rows = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.user_id == int(user_id))
        .order_by(AutonomyEpisode.created_at.desc())
        .limit(24)
    ).all()
    prev_heat = 0.35
    if rows:
        try:
            prev_state = json.loads(rows[0].state_json or "{}")
            prev_heat = float((prev_state if isinstance(prev_state, dict) else {}).get("heat_score", 0.35) or 0.35)
        except Exception:
            prev_heat = 0.35

    entropy_before = float(features_before.get("entropy", 0.3) or 0.3)
    entropy_after = float(features_after.get("entropy", 0.3) or 0.3)
    entropy_spike = max(0.0, entropy_after - entropy_before)
    risk_pressure = max(0.0, float(risk_score) - 0.45)
    low_conf_penalty = max(0.0, 0.55 - float(confidence))
    rejection_pressure = max(0.0, -float(feedback_effect))
    poor_delta_penalty = max(0.0, -float(delta))
    positive_cooling = max(0.0, float(delta)) * 0.45

    raw_heat = (
        (float(prev_heat) * 0.72)
        + (entropy_spike * 0.35)
        + (risk_pressure * 0.28)
        + (low_conf_penalty * 0.22)
        + (rejection_pressure * 0.20)
        + (poor_delta_penalty * 0.30)
        - positive_cooling
    )
    heat_score = _bounded01(raw_heat)
    band = "cool"
    if heat_score >= 0.72:
        band = "hot"
    elif heat_score >= 0.45:
        band = "warm"
    return {"score": round(heat_score, 4), "band": band}


def _compute_long_horizon_goals(session: Session, *, user_id: int) -> dict:
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    rows = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.user_id == int(user_id), AutonomyEpisode.created_at >= week_ago)
        .order_by(AutonomyEpisode.created_at.desc())
        .limit(500)
    ).all()
    if not rows:
        return {
            "window_days": 7,
            "focus_depth": 0.0,
            "recovery_balance": 0.0,
            "novelty_exploration": 0.0,
            "consistency": 0.0,
            "episodes": 0,
        }

    total = float(len(rows))
    learn_ratio = sum(1 for r in rows if str(r.chosen_action or "") == "learn") / total
    rest_ratio = sum(1 for r in rows if str(r.chosen_action or "") == "rest") / total
    exp_ratio = sum(1 for r in rows if str(r.chosen_action or "") == "experiment") / total
    success_ratio = sum(1 for r in rows if str(r.status or "") == "completed") / total
    avg_conf = sum(float(r.confidence or 0.0) for r in rows) / total
    return {
        "window_days": 7,
        "focus_depth": round(_bounded01((learn_ratio * 0.7) + (avg_conf * 0.3)), 4),
        "recovery_balance": round(_bounded01((rest_ratio * 0.8) + (success_ratio * 0.2)), 4),
        "novelty_exploration": round(_bounded01(exp_ratio), 4),
        "consistency": round(_bounded01(success_ratio), 4),
        "episodes": int(total),
    }


def _update_goal_state(goal: AutonomyGoalState, *, features: dict, delta: float) -> None:
    goal.focus_progress = _bounded01(
        (float(goal.focus_progress or 0.5) * 0.92) + (float(features.get("focus", 0.0)) * 0.08)
    )
    goal.recovery_progress = _bounded01(
        (float(goal.recovery_progress or 0.5) * 0.92) + (float(features.get("recovery", 0.0)) * 0.08)
    )
    goal.novelty_progress = _bounded01(
        (float(goal.novelty_progress or 0.4) * 0.92) + (float(features.get("novelty", 0.0)) * 0.08)
    )
    consistency_signal = 0.5 + max(-0.5, min(0.5, float(delta)))
    goal.consistency_progress = _bounded01(
        (float(goal.consistency_progress or 0.45) * 0.92) + (consistency_signal * 0.08)
    )
    goal.updated_at = datetime.utcnow()
    try:
        progress = json.loads(goal.last_7d_json or "{}")
    except Exception:
        progress = {}
    for k, v in (
        ("focus", goal.focus_progress),
        ("recovery", goal.recovery_progress),
        ("novelty", goal.novelty_progress),
        ("consistency", goal.consistency_progress),
    ):
        arr = list(progress.get(k) or [])
        arr.append(round(float(v), 4))
        progress[k] = arr[-14:]
    goal.last_7d_json = json.dumps(progress, sort_keys=True)


def _maybe_upsert_learning_law(
    session: Session,
    *,
    user_id: int,
    action_name: str,
    expected_utility: float,
    observed_utility: float,
    delta: float,
) -> None:
    if abs(delta) < float(getattr(settings, "ecosystem_autonomy_learning_law_threshold", 0.03)):
        return
    sign = "improves" if delta > 0 else "degrades"
    law_name = f"{action_name}:{sign}"
    row = session.exec(
        select(AutonomyLaw)
        .where(AutonomyLaw.user_id == int(user_id), AutonomyLaw.law_name == law_name)
        .limit(1)
    ).first()
    evidence = {
        "action": action_name,
        "expected_utility": round(float(expected_utility), 4),
        "observed_utility": round(float(observed_utility), 4),
        "delta": round(float(delta), 4),
        "recorded_at": datetime.utcnow().isoformat(),
    }
    if row is None:
        row = AutonomyLaw(
            user_id=int(user_id),
            law_name=law_name,
            confidence=0.55 if delta > 0 else 0.45,
            support_n=1,
            law_json=json.dumps({"evidence": [evidence]}, sort_keys=True),
        )
    else:
        row.support_n = int(row.support_n or 0) + 1
        row.confidence = _bounded01(
            (float(row.confidence or 0.5) * 0.85) + ((0.65 if delta > 0 else 0.35) * 0.15)
        )
        try:
            payload = json.loads(row.law_json or "{}")
            existing = list((payload if isinstance(payload, dict) else {}).get("evidence") or [])
        except Exception:
            existing = []
        existing.append(evidence)
        row.law_json = json.dumps({"evidence": existing[-40:]}, sort_keys=True)
    session.add(row)


def _mutate_genome(genome: AutonomyGenome, action_bias: dict) -> dict:
    mutation_rate = max(0.01, min(0.9, float(genome.mutation_rate or 0.08)))
    mutation_delta: dict[str, float] = {}
    for field_name in ("weight_energy", "weight_focus", "weight_recovery", "weight_novelty", "explore_bias"):
        if random.random() < mutation_rate:
            delta = random.uniform(-0.12, 0.12)
            old = float(getattr(genome, field_name))
            new = max(0.05, min(1.5 if field_name.startswith("weight_") else 0.9, old + delta))
            setattr(genome, field_name, new)
            mutation_delta[field_name] = round(new - old, 4)
    for action_name in ("collect", "learn", "experiment", "rest"):
        if random.random() < mutation_rate * 0.9:
            old = float(action_bias.get(action_name, 0.2))
            new = max(0.01, min(0.95, old + random.uniform(-0.1, 0.1)))
            action_bias[action_name] = new
            mutation_delta[f"action_bias.{action_name}"] = round(new - old, 4)
    return mutation_delta


def _run_autonomy_episode(
    session: Session,
    *,
    user_id: int,
    mode: str = "manual",
    forced_action: str | None = None,
) -> dict:
    user_id = int(user_id)
    genome = _autonomy_get_or_create_genome(session, user_id=user_id)
    goal_state = _autonomy_get_or_create_goal_state(session, user_id=user_id)

    try:
        genome_payload = json.loads(genome.genome_json or "{}")
    except Exception:
        genome_payload = {}
    action_bias = dict(genome_payload.get("action_bias") or {})

    summary_before = build_ecosystem_summary(session, user_id=user_id, window_hours=6)
    ts_before = session.exec(
        select(EcosystemTimeSeries)
        .where(EcosystemTimeSeries.user_id == user_id)
        .order_by(EcosystemTimeSeries.created_at.desc())
        .limit(1)
    ).first()
    features_before = _autonomy_state_features(summary_before, ts_before)
    goal_alignment = _compute_long_horizon_goals(session, user_id=user_id)

    actions = [
        ("collect", "Need fresher signals for state estimation."),
        ("learn", "Need a tighter model of force interactions."),
        ("experiment", "Need novelty search to discover stronger dynamics."),
        ("rest", "Need recovery to reduce entropy drift."),
    ]
    candidates: list[dict] = []
    explainability: dict[str, dict] = {}
    for action_name, rationale in actions:
        feedback_effect = _get_nudge_feedback_effect(session, user_id=user_id, action_name=action_name)
        expected = _autonomy_expected_utility(features_before, genome, action_name, action_bias)
        adjusted_expected = _bounded01(expected + (feedback_effect * 0.08))
        risk = _autonomy_risk_score(features_before, action_name)
        risk_level = _risk_level(risk)
        risk_threshold = float(getattr(settings, "ecosystem_autonomy_high_risk_threshold", 0.72))
        safety_gate = not (risk > risk_threshold and action_name in {"learn", "experiment"})
        effective_score = adjusted_expected if safety_gate else adjusted_expected * 0.65
        candidates.append(
            {
                "action": action_name,
                "reason": rationale,
                "expected_utility": round(adjusted_expected, 4),
                "effective_score": round(effective_score, 4),
                "feedback_effect": round(feedback_effect, 4),
                "risk": {"score": round(risk, 4), "risk_level": risk_level, "safety_gate": safety_gate},
            }
        )
        explainability[action_name] = {
            "focus": round(float(genome.weight_focus) * float(features_before.get("focus", 0.0)), 4),
            "energy": round(float(genome.weight_energy) * float(features_before.get("energy", 0.0)), 4),
            "recovery": round(float(genome.weight_recovery) * float(features_before.get("recovery", 0.0)), 4),
            "novelty": round(float(genome.weight_novelty) * float(features_before.get("novelty", 0.0)), 4),
            "feedback_signal": round(feedback_effect, 4),
        }
    candidates.sort(key=lambda x: float(x["effective_score"]), reverse=True)
    chosen = dict(candidates[0])
    if random.random() < float(genome.explore_bias or 0.2) and len(candidates) > 1:
        chosen = dict(random.choice(candidates[: min(3, len(candidates))]))
    if forced_action:
        forced = str(forced_action).strip().lower()
        match = next((c for c in candidates if str(c.get("action") or "") == forced), None)
        if match:
            chosen = dict(match)

    chosen_action = str(chosen["action"])
    rationale = str(chosen["reason"])
    expected_utility = float(chosen["expected_utility"])
    risk_payload = dict(chosen.get("risk") or {})
    risk_score = float(risk_payload.get("score", 0.0) or 0.0)
    risk_level = str(risk_payload.get("risk_level", "safe") or "safe")
    safety_gate = bool(risk_payload.get("safety_gate", True))
    outcome_note = ""
    status = "completed"

    try:
        if chosen_action == "collect":
            n = run_signal_collection_tick(
                _shared_collector_state, user_id=user_id, device_id=str(settings.nexus_device_id or "local")
            )
            outcome_note = f"Collected {int(n)} signals."
        elif chosen_action == "learn":
            r = run_learning_tick(
                user_id=user_id,
                device_id=str(settings.nexus_device_id or "local"),
                window_hours=max(1, min(int(getattr(settings, "ecosystem_learning_window_hours", 6)), 24)),
                bucket_minutes=max(5, min(int(getattr(settings, "ecosystem_learning_bucket_minutes", 30)), 120)),
            )
            outcome_note = "Learning cycle completed." if bool(r.get("ok", True)) else "Learning returned partial state."
        elif chosen_action == "experiment":
            r = run_force_experiment_tick(
                session,
                user_id=user_id,
                cycles=48,
                mutation_rate=float(getattr(settings, "ecosystem_experiment_mutation_rate", 0.12)),
                selection_pressure_ui=0.50,
                thermal_noise=float(getattr(settings, "ecosystem_experiment_thermal_noise", 0.08)),
            )
            out = r.get("output") or {}
            outcome_note = f"Experiment produced {int(out.get('n_species', 0))} species."
        else:
            _ = analyze_patterns(session, user_id=user_id, window_hours=24)
            outcome_note = "Recovery cycle synthesized recent patterns."
    except Exception as exc:
        try:
            n = run_signal_collection_tick(
                _shared_collector_state, user_id=user_id, device_id=str(settings.nexus_device_id or "local")
            )
            status = "completed"
            outcome_note = f"Primary action error ({type(exc).__name__}); fallback collect succeeded with {int(n)} signals."
        except Exception as fallback_exc:
            status = "failed"
            outcome_note = (
                f"Action error: {type(exc).__name__}: {exc}; "
                f"fallback error: {type(fallback_exc).__name__}: {fallback_exc}"
            )

    summary_after = build_ecosystem_summary(session, user_id=user_id, window_hours=6)
    ts_after = session.exec(
        select(EcosystemTimeSeries)
        .where(EcosystemTimeSeries.user_id == user_id)
        .order_by(EcosystemTimeSeries.created_at.desc())
        .limit(1)
    ).first()
    features_after = _autonomy_state_features(summary_after, ts_after)
    observed_utility = _autonomy_expected_utility(features_after, genome, chosen_action, action_bias)
    counterfactual_utility = _counterfactual_v2(
        session, user_id=user_id, action=chosen_action, expected_utility=expected_utility
    )
    delta = observed_utility - counterfactual_utility
    confidence = _bounded01((expected_utility * 0.6) + (observed_utility * 0.4))
    chosen_feedback_effect = _get_nudge_feedback_effect(session, user_id=user_id, action_name=chosen_action)
    heat = _autonomy_heat_state(
        session,
        user_id=user_id,
        features_before=features_before,
        features_after=features_after,
        delta=delta,
        confidence=confidence,
        risk_score=risk_score,
        feedback_effect=chosen_feedback_effect,
    )

    high_impact = chosen_action in {"experiment", "learn"}
    confidence_floor = float(getattr(settings, "ecosystem_autonomy_governance_min_confidence", 0.55))
    risk_gate = float(getattr(settings, "ecosystem_autonomy_governance_risk_threshold", 0.72))
    heat_gate = float(getattr(settings, "ecosystem_autonomy_heat_governance_threshold", 0.78))
    requires_confirmation = bool(
        high_impact
        and (
            (risk_score >= risk_gate)
            or (heat.get("score", 0.0) >= heat_gate)
            or (confidence < confidence_floor)
        )
    )
    proposal_status = "proposed" if requires_confirmation else "none"

    _update_goal_state(goal_state, features=features_after, delta=delta)
    session.add(goal_state)
    _maybe_upsert_learning_law(
        session,
        user_id=user_id,
        action_name=chosen_action,
        expected_utility=expected_utility,
        observed_utility=observed_utility,
        delta=delta,
    )

    action_bias[chosen_action] = max(
        0.01, min(0.95, float(action_bias.get(chosen_action, 0.2)) + (0.07 if delta > 0 else -0.03))
    )
    mutation_delta = {}
    if random.random() < float(getattr(settings, "ecosystem_autonomy_mutation_rate", 0.08)):
        mutation_delta = _mutate_genome(genome, action_bias)
        genome.generation = int(genome.generation) + 1
    genome.fitness_score = (float(genome.fitness_score or 0.0) * 0.85) + (observed_utility * 0.15)
    genome.genome_json = json.dumps(
        {
            "action_bias": action_bias,
            "selection_pressure": float(getattr(settings, "ecosystem_experiment_selection_pressure", 1.35)),
            "last_mutation": mutation_delta,
        },
        sort_keys=True,
    )
    session.add(genome)

    recommended_tick = _adaptive_tick_minutes(session, user_id=user_id)
    episode_state = {
        "before": features_before,
        "after": features_after,
        "summary_before": summary_before,
        "summary_after": summary_after,
        "goal_alignment": goal_alignment,
        "policy_genome": {
            "generation": int(genome.generation),
            "selection_pressure": float(getattr(settings, "ecosystem_experiment_selection_pressure", 1.35)),
            "fitness_score": round(float(genome.fitness_score or 0.0), 4),
            "explore_bias": round(float(genome.explore_bias or 0.0), 4),
        },
        "recommended_next_tick_minutes": int(recommended_tick),
        "heat_score": float(heat.get("score", 0.0)),
        "heat_band": str(heat.get("band", "cool")),
        "project_codename": str(getattr(settings, "project_codename", "Project Resonance")),
        "project_tagline": str(getattr(settings, "project_tagline", "Where data settles into life.")),
    }
    episode_outcome = {
        "outcome_note": outcome_note,
        "counterfactual_utility": round(counterfactual_utility, 4),
        "counterfactual_method": "counterfactual_v2",
        "delta": round(delta, 4),
        "mutation_delta": mutation_delta,
        "risk": {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "safety_gate": bool(safety_gate),
        },
        "nudge_feedback_signal": round(chosen_feedback_effect, 4),
        "explainability": explainability.get(chosen_action, {}),
        "governance": {
            "requires_confirmation": bool(requires_confirmation),
            "proposal_status": proposal_status,
            "reason": (
                "high_impact_gate"
                if requires_confirmation
                else "auto_execute"
            ),
        },
    }
    episode = AutonomyEpisode(
        user_id=user_id,
        mode="autonomous" if mode == "daemon" else "manual",
        status=status,
        governance_status="proposed" if requires_confirmation else "executed",
        requires_human_confirm=bool(requires_confirmation),
        chosen_action=chosen_action,
        rationale=rationale,
        expected_utility=float(round(expected_utility, 4)),
        observed_utility=float(round(observed_utility, 4)),
        confidence=float(round(confidence, 4)),
        n_signals_window=int(features_after["n_signals_window"]),
        coherence=float(round(features_after["coherence"], 4)),
        attention_entropy=float(round(features_after["entropy"], 4)),
        momentum=float(round(features_after["momentum"], 4)),
        novelty_index=float(round(features_after["novelty"], 4)),
        action_candidates_json=json.dumps(candidates, sort_keys=True),
        state_json=json.dumps(episode_state, sort_keys=True),
        outcome_json=json.dumps(episode_outcome, sort_keys=True),
    )
    session.add(episode)
    session.commit()
    session.refresh(episode)

    return {
        "ok": True,
        "episode": {
            "id": int(episode.id or 0),
            "created_at": episode.created_at.isoformat() if episode.created_at else "",
            "mode": str(episode.mode or ""),
            "action": chosen_action,
            "why": rationale,
            "outcome_score": round(float(episode.observed_utility or 0.0), 4),
            "expected_utility": round(float(episode.expected_utility or 0.0), 4),
            "counterfactual_utility": round(counterfactual_utility, 4),
            "counterfactual_method": "counterfactual_v2",
            "delta": round(delta, 4),
            "confidence": round(float(episode.confidence or 0.0), 4),
            "policy_genome": {
                "generation": int(genome.generation),
                "selection_pressure": float(getattr(settings, "ecosystem_experiment_selection_pressure", 1.35)),
                "fitness_score": round(float(genome.fitness_score or 0.0), 4),
                "explore_bias": round(float(genome.explore_bias or 0.0), 4),
            },
            "risk": {"risk_score": round(risk_score, 4), "risk_level": risk_level, "safety_gate": bool(safety_gate)},
            "goal_alignment": goal_alignment,
            "recommended_next_tick_minutes": int(recommended_tick),
            "nudge_feedback_signal": round(
                chosen_feedback_effect, 4
            ),
            "explainability": explainability.get(chosen_action, {}),
            "heat_score": float(heat.get("score", 0.0)),
            "heat_band": str(heat.get("band", "cool")),
            "governance": episode_outcome.get("governance", {}),
            "mutation_delta": mutation_delta,
            "outcome_note": outcome_note,
        },
    }


def run_autonomy_episode(
    session: Session,
    *,
    user_id: int,
    mode: str = "manual",
) -> dict:
    return _run_autonomy_episode(session, user_id=int(user_id), mode=str(mode))


def _experimental_force_evolution(
    session: Session,
    *,
    user_id: int,
    steps: int = 12,
    mutation_rate: float | None = None,
    selection_pressure: float | None = None,
    thermal_noise: float | None = None,
) -> dict:
    """Run a tiny in-memory evolution simulation seeded by recent ecosystem rows.

    This is intentionally lightweight and deterministic enough for UI refreshes.
    It models "signal species" as bodies with:
      - mass (importance/load),
      - charge (signed tendency),
      - velocity (change),
      - energy and fitness.
    """
    rows = session.exec(
        select(EcosystemTimeSeries)
        .where(EcosystemTimeSeries.user_id == int(user_id))
        .order_by(EcosystemTimeSeries.created_at.desc())
        .limit(240)
    ).all()
    history = list(reversed(rows))

    if not history:
        synthetic = EcosystemTimeSeries(
            user_id=int(user_id),
            n_signals_window=12,
            coherence=0.25,
            attention_entropy=0.40,
            force_g=0.30,
            force_em=0.50,
            force_strong=0.80,
            force_weak=0.02,
        )
        history = [synthetic]

    # Seed force constants from observed averages.
    g = max(0.05, float(mean([r.force_g for r in history if r.force_g is not None] or [0.3])))
    em = max(0.05, float(mean([r.force_em for r in history if r.force_em is not None] or [0.5])))
    strong = max(0.05, float(mean([r.force_strong for r in history if r.force_strong is not None] or [0.8])))
    weak = max(0.001, float(mean([r.force_weak for r in history if r.force_weak is not None] or [0.02])))

    last = history[-1]
    base_mass = max(1.0, float(last.n_signals_window or 1))
    base_entropy = max(0.01, float(last.attention_entropy or 0.1))
    base_coherence = max(0.0, min(1.0, float(last.coherence or 0.0)))

    stage_by_age = {0: "infant", 1: "toddler", 2: "adolescent", 3: "adult"}
    species = []
    for idx in range(4):
        species.append(
            {
                "id": f"sp-{idx+1}",
                "mass": max(0.4, base_mass / (idx + 2)),
                "charge": random.uniform(-1.0, 1.0),
                "velocity": random.uniform(-0.05, 0.05),
                "energy": max(0.1, (base_coherence + 0.2) * random.uniform(0.5, 1.5)),
                "stability": max(0.1, 1.0 - min(0.9, base_entropy / 4.0)),
                "fitness": 0.0,
                "age": 0,
                "stage": "infant",
                "mutations": 0,
            }
        )

    steps = max(4, min(int(steps), 1000))
    timeline: list[dict] = []
    default_mutation = 0.08 + base_entropy * 0.07
    mutation_rate = (
        float(getattr(settings, "ecosystem_experiment_mutation_rate", 0.12))
        if mutation_rate is None
        else float(mutation_rate)
    )
    mutation_rate = max(0.01, min(0.90, mutation_rate if mutation_rate > 0 else default_mutation))
    selection_pressure = (
        float(getattr(settings, "ecosystem_experiment_selection_pressure", 1.35))
        if selection_pressure is None
        else float(selection_pressure)
    )
    selection_pressure = max(1.01, min(3.0, selection_pressure))
    thermal_noise = float(thermal_noise) if thermal_noise is not None else 0.08
    thermal_noise = max(0.0, min(1.0, thermal_noise))
    carrying_capacity = max(8.0, base_mass * 1.8)

    for tick in range(steps):
        total_mass = sum(float(sp["mass"]) for sp in species)
        population_pressure = max(0.0, (total_mass - carrying_capacity) / carrying_capacity)

        for sp in species:
            mass = float(sp["mass"])
            charge = float(sp["charge"])
            vel = float(sp["velocity"])
            energy = float(sp["energy"])
            stability = float(sp["stability"])

            # Hybrid force: statistical correlation-like term + physics-like terms.
            grav_pull = g * (mass / max(1.0, total_mass))
            em_push = em * charge * (1.0 - stability)
            strong_bind = strong * stability * 0.35
            weak_decay = weak * (0.4 + population_pressure)
            noise = random.uniform(-0.015, 0.015) * (1.0 + base_entropy + thermal_noise)

            accel = grav_pull + em_push + strong_bind - weak_decay + noise
            vel = max(-0.35, min(0.35, vel + accel))
            mass = max(0.05, mass + vel * 0.15)
            energy = max(0.01, energy + (abs(vel) * 0.3) - (population_pressure * 0.06))
            stability = max(0.02, min(1.2, stability + (strong_bind - weak_decay) * 0.08))

            # Fitness balances energy, stability, and manageable entropy pressure.
            fitness = max(0.0, (energy * 0.45) + (stability * 0.45) - (base_entropy * 0.10))

            # Mutation: occasionally perturb charge/velocity, inspired by entropy.
            if random.random() < mutation_rate:
                charge = max(-1.5, min(1.5, charge + random.uniform(-0.22, 0.22)))
                vel = max(-0.4, min(0.4, vel + random.uniform(-0.05, 0.05)))
                sp["mutations"] = int(sp["mutations"]) + 1

            sp["mass"] = mass
            sp["charge"] = charge
            sp["velocity"] = vel
            sp["energy"] = energy
            sp["stability"] = stability
            sp["fitness"] = fitness
            sp["age"] = int(sp["age"]) + 1
            age_band = min(3, int(sp["age"]) // max(1, steps // 4))
            sp["stage"] = stage_by_age.get(age_band, "adult")

        # Selection + reproduction (simple elitist strategy).
        species.sort(key=lambda x: float(x["fitness"]), reverse=True)
        survivors = species[: max(2, len(species) - 1)]
        strongest = survivors[0]
        weakest = survivors[-1]
        if float(strongest["fitness"]) > float(weakest["fitness"]) * selection_pressure:
            offspring = dict(strongest)
            offspring["id"] = f"sp-{tick+100}"
            offspring["age"] = 0
            offspring["stage"] = "infant"
            offspring["mutations"] = int(offspring.get("mutations", 0))
            offspring["mass"] = max(0.05, float(offspring["mass"]) * random.uniform(0.55, 0.85))
            offspring["charge"] = max(-1.5, min(1.5, float(offspring["charge"]) + random.uniform(-0.15, 0.15)))
            offspring["velocity"] = float(offspring["velocity"]) * 0.5
            offspring["energy"] = max(0.01, float(offspring["energy"]) * random.uniform(0.7, 0.95))
            species = survivors + [offspring]
        else:
            species = survivors

        timeline.append(
            {
                "tick": tick + 1,
                "n_species": len(species),
                "mean_fitness": round(float(mean([float(sp["fitness"]) for sp in species])), 4),
                "mean_mass": round(float(mean([float(sp["mass"]) for sp in species])), 4),
                "mean_energy": round(float(mean([float(sp["energy"]) for sp in species])), 4),
                "mutation_rate": round(float(mutation_rate), 4),
            }
        )

    # Stable ordering for UI and API consumers.
    species.sort(key=lambda x: float(x["fitness"]), reverse=True)
    pub_species = [
        {
            "id": str(sp["id"]),
            "stage": str(sp["stage"]),
            "age": int(sp["age"]),
            "mass": round(float(sp["mass"]), 4),
            "charge": round(float(sp["charge"]), 4),
            "velocity": round(float(sp["velocity"]), 4),
            "energy": round(float(sp["energy"]), 4),
            "stability": round(float(sp["stability"]), 4),
            "fitness": round(float(sp["fitness"]), 4),
            "mutations": int(sp["mutations"]),
        }
        for sp in species[:8]
    ]

    return {
        "ok": True,
        "seed_points": len(history),
        "seed_window_hours": 24,
        "force_constants": {
            "gravity": round(g, 4),
            "electromagnetic": round(em, 4),
            "strong_nuclear": round(strong, 4),
            "weak_nuclear": round(weak, 4),
        },
        "carrying_capacity": round(float(carrying_capacity), 4),
        "mutation_rate": round(float(mutation_rate), 4),
        "steps": int(steps),
        "species": pub_species,
        "timeline": timeline,
        "narrative": (
            "Experimental mode: signals behave like evolving species under combined "
            "statistical and physics-like forces. Track mass, charge, energy, and mutation drift."
        ),
    }


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
        "crystals": _build_crystals(session, user_id),
        "soul": _build_soul(session, user_id),
    }


def _build_soul(session, user_id: int) -> dict | None:
    """Compose the digital soul from crystal neural network."""
    try:
        from mycelium_app.crystallization import crystallize
        from mycelium_app.digital_soul import compose_digital_soul, serialize_soul
        from mycelium_app.force_field import compute_force_field
        from mycelium_app.assistant_profile import get_assistant_profile_effective
        import json as _json
        from datetime import datetime as dt, timedelta as td

        since = dt.utcnow() - td(hours=6)
        from mycelium_app.models import SignalLedgerEvent as SLE
        rows = session.exec(
            select(SLE).where(SLE.created_by_user_id == int(user_id), SLE.created_at >= since)
        ).all()
        if not rows:
            return None

        signals = []
        recent_apps = []
        for r in rows:
            try: payload = _json.loads(r.payload_json or "{}")
            except: payload = {}
            surface = payload.get("surface") or payload.get("stimulus") or payload
            app = str(surface.get("app_name", r.signal_type or ""))
            signals.append({
                "signal_type": str(r.signal_type or ""), "app_name": app,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "payload": surface,
            })
            if r.signal_type in ("app_focus", "app_open"):
                recent_apps.append(app.lower())

        ff = compute_force_field(signals, window_hours=6, n_iterations=15)
        crystals = crystallize(ff)
        profile = get_assistant_profile_effective(session, user_id=user_id, project_id=None)

        soul = compose_digital_soul(
            crystals,
            agent_name=str(profile.get("given_name", "Myco")),
            active_signals=recent_apps[-20:],
        )
        return serialize_soul(soul)
    except Exception:
        return None


def _build_crystals(session, user_id: int) -> dict | None:
    """Build crystallized signal complexes from force field."""
    try:
        from mycelium_app.crystallization import crystallize, serialize_crystallization
        from mycelium_app.force_field import compute_force_field
        import json as _json
        from datetime import datetime as dt, timedelta as td

        since = dt.utcnow() - td(hours=6)
        from mycelium_app.models import SignalLedgerEvent as SLE
        rows = session.exec(
            select(SLE).where(SLE.created_by_user_id == int(user_id), SLE.created_at >= since)
        ).all()
        if not rows:
            return None

        signals = []
        for r in rows:
            try: payload = _json.loads(r.payload_json or "{}")
            except: payload = {}
            surface = payload.get("surface") or payload.get("stimulus") or payload
            signals.append({
                "signal_type": str(r.signal_type or ""),
                "app_name": str(surface.get("app_name", r.signal_type or "")),
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "payload": surface,
            })

        ff = compute_force_field(signals, window_hours=6, n_iterations=15)
        result = crystallize(ff)
        return serialize_crystallization(result)
    except Exception:
        return None


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


@router.get("/soul")
def get_soul(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get the digital soul state — the emergent consciousness."""
    result = _build_soul(session, int(current_user.id or 0))
    if not result:
        return {"ok": True, "soul": None, "maturity": "nascent"}
    return {"ok": True, "soul": result}


@router.get("/crystals")
def get_crystals(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get all crystallized signal complexes (agent population)."""
    result = _build_crystals(session, int(current_user.id or 0))
    if not result:
        return {"ok": True, "crystals": [], "ecosystem_maturity": "nascent"}
    return {"ok": True, **result}


@router.get("/hive-readiness")
def hive_readiness(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Check if the ecosystem is mature enough to join the Hive."""
    from mycelium_app.hive_readiness import check_hive_readiness
    return check_hive_readiness(session, user_id=int(current_user.id or 0))


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


@router.get("/evolution")
def ecosystem_evolution(
    steps: int = 24,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Experimental evolution sandbox for force + mass theory."""
    user_id = int(current_user.id or 0)
    return _experimental_force_evolution(session, user_id=user_id, steps=steps)


def run_force_experiment_tick(
    session: Session,
    *,
    user_id: int,
    cycles: int | None = None,
    mutation_rate: float | None = None,
    selection_pressure_ui: float | None = None,
    thermal_noise: float | None = None,
) -> dict:
    """Run one experiment cycle and persist the result for history/latest APIs."""
    settings_mutation_default = float(getattr(settings, "ecosystem_experiment_mutation_rate", 0.12) or 0.12)
    settings_thermal_default = float(getattr(settings, "ecosystem_experiment_thermal_noise", 0.08) or 0.08)
    resolved_cycles = max(4, min(int(cycles or 120), 1000))
    resolved_mutation = float(
        mutation_rate if mutation_rate is not None else settings_mutation_default
    )
    resolved_mutation = max(0.01, min(0.9, resolved_mutation))
    resolved_thermal = float(
        thermal_noise
        if thermal_noise is not None
        else settings_thermal_default
    )
    resolved_thermal = max(0.0, min(1.0, resolved_thermal))
    sel_ui = float(selection_pressure_ui if selection_pressure_ui is not None else 0.45)
    # UI submits 0..1, simulation uses multiplicative threshold 1.01..3.0
    resolved_selection = 1.0 + max(0.01, min(2.0, sel_ui * 2.0))

    raw = _experimental_force_evolution(
        session,
        user_id=user_id,
        steps=resolved_cycles,
        mutation_rate=resolved_mutation,
        selection_pressure=resolved_selection,
        thermal_noise=resolved_thermal,
    )
    species = raw.get("species") or []
    timeline = raw.get("timeline") or []
    best = species[0] if species else {}
    stage_rank = {"infant": 0, "toddler": 1, "adolescent": 2, "adult": 3}
    emergent_stage = max(
        (sp.get("stage", "infant") for sp in species),
        key=lambda s: stage_rank.get(str(s), 0),
        default="infant",
    )

    output = {
        "seed_particles": int(raw.get("seed_points", 0)),
        "n_species": len(species),
        "emergent_stage": str(emergent_stage),
        "novelty_index": round(float(raw.get("mutation_rate", 0.0)) * max(1, len(species)) / 2.0, 4),
        "best_species": {
            "id": str(best.get("id", "—")),
            "fitness": float(best.get("fitness", 0.0) or 0.0),
            "cohesion": float(best.get("stability", 0.0) or 0.0),
            "mutations": int(best.get("mutations", 0) or 0),
        },
        "population_trace": [float(item.get("mean_mass", 0.0) or 0.0) for item in timeline],
        "force_constants": raw.get("force_constants", {}),
        "message": raw.get("narrative", ""),
    }

    try:
        row = EcosystemExperimentTick(
            user_id=int(user_id),
            cycles=resolved_cycles,
            mutation_rate=resolved_mutation,
            selection_pressure=resolved_selection,
            thermal_noise=resolved_thermal,
            seed_particles=int(output["seed_particles"]),
            n_species=int(output["n_species"]),
            emergent_stage=str(output["emergent_stage"]),
            novelty_index=float(output["novelty_index"]),
            output_json=json.dumps(output),
            best_species_json=json.dumps(output.get("best_species", {})),
        )
        session.add(row)
        session.commit()
    except Exception:
        # Experiment persistence should not block response.
        pass
    return {"ok": True, "output": output}


@router.post("/experiment/run")
def run_force_experiment(
    payload: EcosystemExperimentRunRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Run experimental force+mass evolution with caller-supplied knobs."""
    user_id = int(current_user.id or 0)
    try:
        return run_force_experiment_tick(
            session,
            user_id=user_id,
            cycles=int(payload.cycles or 120),
            mutation_rate=float(
                payload.mutation_rate
                if payload.mutation_rate is not None
                else getattr(settings, "ecosystem_experiment_mutation_rate", 0.12)
            ),
            selection_pressure_ui=float(payload.selection_pressure or 0.45),
            thermal_noise=float(
                payload.thermal_noise
                if payload.thermal_noise is not None
                else getattr(settings, "ecosystem_experiment_thermal_noise", 0.08)
            ),
        )
    except Exception as exc:
        # Return actionable error payload to avoid opaque "Experiment failed" UI.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@router.get("/experiment/latest")
def latest_force_experiment(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    row = session.exec(
        select(EcosystemExperimentTick)
        .where(EcosystemExperimentTick.user_id == user_id)
        .order_by(EcosystemExperimentTick.created_at.desc())
        .limit(1)
    ).first()
    if not row:
        return {"ok": True, "has_data": False}
    try:
        output = json.loads(row.output_json or "{}")
    except Exception:
        output = {}
    return {
        "ok": True,
        "has_data": True,
        "tick": {
            "id": int(row.id or 0),
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "cycles": int(row.cycles),
            "mutation_rate": float(row.mutation_rate),
            "selection_pressure": float(row.selection_pressure),
            "thermal_noise": float(row.thermal_noise),
            "seed_particles": int(row.seed_particles),
            "n_species": int(row.n_species),
            "emergent_stage": str(row.emergent_stage),
            "novelty_index": float(row.novelty_index),
        },
        "output": output,
    }


@router.get("/experiment/history")
def force_experiment_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    n = max(1, min(int(limit), 200))
    rows = session.exec(
        select(EcosystemExperimentTick)
        .where(EcosystemExperimentTick.user_id == user_id)
        .order_by(EcosystemExperimentTick.created_at.desc())
        .limit(n)
    ).all()
    history = [
        {
            "id": int(r.id or 0),
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "cycles": int(r.cycles),
            "mutation_rate": float(r.mutation_rate),
            "selection_pressure": float(r.selection_pressure),
            "thermal_noise": float(r.thermal_noise),
            "seed_particles": int(r.seed_particles),
            "n_species": int(r.n_species),
            "emergent_stage": str(r.emergent_stage),
            "novelty_index": float(r.novelty_index),
        }
        for r in rows
    ]
    return {"ok": True, "count": len(history), "history": history}


@router.post("/autonomy/run")
def run_autonomy_api(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    return _run_autonomy_episode(session, user_id=user_id, mode="manual")


@router.get("/autonomy/latest")
def latest_autonomy_episode(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    row = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.user_id == user_id)
        .order_by(AutonomyEpisode.created_at.desc())
        .limit(1)
    ).first()
    if not row:
        return {"ok": True, "has_data": False}

    try:
        state = json.loads(row.state_json or "{}")
    except Exception:
        state = {}
    try:
        outcome = json.loads(row.outcome_json or "{}")
    except Exception:
        outcome = {}

    state_policy = state.get("policy_genome") if isinstance(state, dict) else {}
    if not isinstance(state_policy, dict):
        state_policy = {}

    return {
        "ok": True,
        "has_data": True,
        "episode": {
            "id": int(row.id or 0),
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "mode": str(row.mode or ""),
            "action": str(row.chosen_action or ""),
            "why": str(row.rationale or ""),
            "expected_utility": float(row.expected_utility or 0.0),
            "outcome_score": float(row.observed_utility or 0.0),
            "counterfactual_utility": float(outcome.get("counterfactual_utility", 0.0) or 0.0),
            "delta": float(outcome.get("delta", 0.0) or 0.0),
            "confidence": float(row.confidence or 0.0),
            "generation": int(state_policy.get("generation", 0) or 0),
            "state": state,
            "action_detail": {"chosen_action": str(row.chosen_action or "")},
            "outcome": outcome,
            "policy_genome": {
                "generation": int(state_policy.get("generation", 0) or 0),
                "fitness_score": float(state_policy.get("fitness_score", 0.0) or 0.0),
                "selection_pressure": float(state_policy.get("selection_pressure", 0.0) or 0.0),
                "explore_bias": float(state_policy.get("explore_bias", 0.0) or 0.0),
            },
            "mutation_delta": outcome.get("mutation_delta", {}) if isinstance(outcome, dict) else {},
            "outcome_note": str(outcome.get("outcome_note", "")),
            "risk": (outcome.get("risk", {}) if isinstance(outcome, dict) else {}),
            "goal_alignment": (state.get("goal_alignment", {}) if isinstance(state, dict) else {}),
            "recommended_next_tick_minutes": int(
                (state.get("recommended_next_tick_minutes", getattr(settings, "ecosystem_autonomy_tick_minutes", 10)) if isinstance(state, dict) else getattr(settings, "ecosystem_autonomy_tick_minutes", 10)) or getattr(settings, "ecosystem_autonomy_tick_minutes", 10)
            ),
            "nudge_feedback_signal": float(outcome.get("nudge_feedback_signal", 0.0) or 0.0),
            "explainability": (outcome.get("explainability", {}) if isinstance(outcome, dict) else {}),
            "heat_score": float((state.get("heat_score", 0.0) if isinstance(state, dict) else 0.0) or 0.0),
            "heat_band": str((state.get("heat_band", "cool") if isinstance(state, dict) else "cool") or "cool"),
            "governance": (outcome.get("governance", {}) if isinstance(outcome, dict) else {}),
            "project": {
                "codename": str((state.get("project_codename", getattr(settings, "project_codename", "Project Resonance")) if isinstance(state, dict) else getattr(settings, "project_codename", "Project Resonance"))),
                "tagline": str((state.get("project_tagline", getattr(settings, "project_tagline", "Where data settles into life.")) if isinstance(state, dict) else getattr(settings, "project_tagline", "Where data settles into life."))),
            },
        },
    }


@router.get("/autonomy/history")
def autonomy_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    n = max(1, min(int(limit), 200))
    rows = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.user_id == user_id)
        .order_by(AutonomyEpisode.created_at.desc())
        .limit(n)
    ).all()

    history = []
    for row in rows:
        try:
            outcome = json.loads(row.outcome_json or "{}")
        except Exception:
            outcome = {}
        try:
            state = json.loads(row.state_json or "{}")
        except Exception:
            state = {}
        history.append(
            {
                "id": int(row.id or 0),
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "mode": str(row.mode or ""),
                "action": str(row.chosen_action or ""),
                "expected_utility": float(row.expected_utility or 0.0),
                "outcome_score": float(row.observed_utility or 0.0),
                "counterfactual_utility": float(outcome.get("counterfactual_utility", 0.0) or 0.0),
                "delta": float(outcome.get("delta", 0.0) or 0.0),
                "generation": int((state.get("policy_genome") or {}).get("generation", 0) or 0),
                "confidence": float(row.confidence or 0.0),
            }
        )
    return {"ok": True, "count": len(history), "history": history}


@router.get("/autonomy/goals")
def autonomy_goals(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    goal = _autonomy_get_or_create_goal_state(session, user_id=user_id)
    try:
        week = json.loads(goal.last_7d_json or "{}")
    except Exception:
        week = {}
    return {
        "ok": True,
        "goals": {
            "focus_goal": float(goal.focus_goal or 0.0),
            "recovery_goal": float(goal.recovery_goal or 0.0),
            "novelty_goal": float(goal.novelty_goal or 0.0),
            "consistency_goal": float(goal.consistency_goal or 0.0),
            "focus_progress": float(goal.focus_progress or 0.0),
            "recovery_progress": float(goal.recovery_progress or 0.0),
            "novelty_progress": float(goal.novelty_progress or 0.0),
            "consistency_progress": float(goal.consistency_progress or 0.0),
            "weekly_trace": week,
        },
    }


@router.get("/autonomy/laws")
def autonomy_laws(
    limit: int = 12,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    n = max(1, min(int(limit), 100))
    rows = session.exec(
        select(AutonomyLaw)
        .where(AutonomyLaw.user_id == user_id)
        .order_by(AutonomyLaw.created_at.desc())
        .limit(n)
    ).all()
    laws = []
    for row in rows:
        try:
            law_obj = json.loads(row.law_json or "{}")
            evidence = list((law_obj if isinstance(law_obj, dict) else {}).get("evidence") or [])
        except Exception:
            evidence = []
        laws.append(
            {
                "id": int(row.id or 0),
                "law_name": str(row.law_name or ""),
                "confidence": float(row.confidence or 0.0),
                "support_count": int(row.support_n or 0),
                "latest_evidence": (evidence[-1] if evidence else {}),
                "updated_at": row.created_at.isoformat() if row.created_at else "",
            }
        )
    return {"ok": True, "count": len(laws), "laws": laws}


@router.post("/autonomy/propose")
def propose_autonomy_action(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Two-key governance: create an explicit proposal for high-impact actions."""
    user_id = int(current_user.id or 0)
    action_name = str(payload.get("action") or "").strip().lower()
    rationale = str(payload.get("why") or "Operator proposed action").strip()[:500]
    allowed = {"collect", "learn", "experiment", "rest"}
    if action_name not in allowed:
        return {"ok": False, "error": "invalid_action"}

    proposal = AutonomyPendingAction(
        user_id=user_id,
        action_name=action_name,
        rationale=rationale,
        status="proposed",
        proposal_json=json.dumps(
            {
                "source": "manual_proposal",
                "created_at": datetime.utcnow().isoformat(),
                "project_codename": str(getattr(settings, "project_codename", "Project Resonance")),
            },
            sort_keys=True,
        ),
    )
    session.add(proposal)
    session.commit()
    session.refresh(proposal)
    return {
        "ok": True,
        "proposal": {
            "id": int(proposal.id or 0),
            "action": str(proposal.action_name or ""),
            "status": str(proposal.status or "proposed"),
            "created_at": proposal.created_at.isoformat() if proposal.created_at else "",
        },
    }


@router.post("/autonomy/confirm")
def confirm_autonomy_action(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Two-key governance: confirm a proposed action and execute a real episode."""
    user_id = int(current_user.id or 0)
    proposal_id = int(payload.get("pending_id") or payload.get("proposal_id") or 0)
    approve = bool(payload.get("approve", True))
    row = session.exec(
        select(AutonomyPendingAction)
        .where(AutonomyPendingAction.id == proposal_id, AutonomyPendingAction.user_id == user_id)
        .limit(1)
    ).first()
    if not row:
        return {"ok": False, "error": "proposal_not_found"}
    if str(row.status or "") != "proposed":
        return {"ok": False, "error": "proposal_not_open"}
    if not approve:
        row.status = "rejected"
        row.updated_at = datetime.utcnow()
        session.add(row)
        session.commit()
        return {"ok": True, "pending_id": int(row.id or 0), "status": "rejected"}

    row.status = "approved"
    row.updated_at = datetime.utcnow()
    session.add(row)
    session.commit()
    forced_action = str(row.action_name or "").strip().lower()
    result = _run_autonomy_episode(session, user_id=user_id, mode="manual", forced_action=forced_action)
    try:
        episode_id = int((result.get("episode") or {}).get("id") or 0)
    except Exception:
        episode_id = 0
    row.proposal_json = json.dumps(
        {
            "executed_episode_id": episode_id or None,
            "last_confirmed_at": datetime.utcnow().isoformat(),
        },
        sort_keys=True,
    )
    row.status = "executed" if episode_id else "approved"
    row.updated_at = datetime.utcnow()
    session.add(row)
    session.commit()
    return {
        "ok": True,
        "pending_id": int(row.id or 0),
        "status": str(row.status or "approved"),
        "executed_episode_id": int(episode_id or 0),
        "episode": result.get("episode"),
    }


@router.get("/autonomy/pending")
def list_pending_autonomy_actions(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    n = max(1, min(int(limit), 100))
    rows = session.exec(
        select(AutonomyPendingAction)
        .where(AutonomyPendingAction.user_id == user_id)
        .order_by(AutonomyPendingAction.created_at.desc())
        .limit(n)
    ).all()
    pending = []
    for row in rows:
        try:
            payload = json.loads(row.proposal_json or "{}")
        except Exception:
            payload = {}
        pending.append(
            {
                "id": int(row.id or 0),
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "updated_at": row.updated_at.isoformat() if row.updated_at else "",
                "status": str(row.status or ""),
                "mode": str(row.mode or ""),
                "action": str(row.action_name or ""),
                "why": str(row.rationale or ""),
                "risk_score": float(row.risk_score or 0.0),
                "risk_level": str(row.risk_level or "safe"),
                "expected_utility": float(row.expected_utility or 0.0),
                "heat_score": float(row.heat_score or 0.0),
                "proposal": payload if isinstance(payload, dict) else {},
            }
        )
    return {"ok": True, "count": len(pending), "pending": pending}


@router.get("/autonomy/governance/report")
def autonomy_governance_report(
    window_days: int = 7,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Weekly governance report for heat, risk, and approval dynamics."""
    user_id = int(current_user.id or 0)
    days = max(1, min(int(window_days), 30))
    since = datetime.utcnow() - timedelta(days=days)

    episodes = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.user_id == user_id, AutonomyEpisode.created_at >= since)
        .order_by(AutonomyEpisode.created_at.desc())
        .limit(2000)
    ).all()
    pending_rows = session.exec(
        select(AutonomyPendingAction)
        .where(AutonomyPendingAction.user_id == user_id, AutonomyPendingAction.created_at >= since)
        .order_by(AutonomyPendingAction.created_at.desc())
        .limit(2000)
    ).all()

    heat_vals: list[float] = []
    risk_vals: list[float] = []
    deltas: list[float] = []
    conf_vals: list[float] = []
    high_risk_episodes = 0
    governance_required = 0
    for ep in episodes:
        try:
            state = json.loads(ep.state_json or "{}")
        except Exception:
            state = {}
        try:
            out = json.loads(ep.outcome_json or "{}")
        except Exception:
            out = {}
        heat_vals.append(float((state if isinstance(state, dict) else {}).get("heat_score", 0.0) or 0.0))
        risk = (out if isinstance(out, dict) else {}).get("risk") if isinstance(out, dict) else {}
        if not isinstance(risk, dict):
            risk = {}
        rv = float(risk.get("risk_score", 0.0) or 0.0)
        risk_vals.append(rv)
        if rv >= float(getattr(settings, "ecosystem_autonomy_governance_risk_threshold", 0.72)):
            high_risk_episodes += 1
        gov = (out if isinstance(out, dict) else {}).get("governance") if isinstance(out, dict) else {}
        if isinstance(gov, dict) and bool(gov.get("requires_confirmation", False)):
            governance_required += 1
        deltas.append(float((out if isinstance(out, dict) else {}).get("delta", 0.0) or 0.0))
        conf_vals.append(float(ep.confidence or 0.0))

    proposed = sum(1 for p in pending_rows if str(p.status or "") == "proposed")
    approved = sum(1 for p in pending_rows if str(p.status or "") in {"approved", "executed"})
    rejected = sum(1 for p in pending_rows if str(p.status or "") == "rejected")
    executed = sum(1 for p in pending_rows if str(p.status or "") == "executed")
    denom = max(1, approved + rejected)

    def _avg(vals: list[float]) -> float:
        return round(float(mean(vals)) if vals else 0.0, 4)

    return {
        "ok": True,
        "window_days": days,
        "project": {
            "codename": str(getattr(settings, "project_codename", "Project Resonance")),
            "tagline": str(getattr(settings, "project_tagline", "Where data settles into life.")),
        },
        "episodes": {
            "count": len(episodes),
            "mean_heat": _avg(heat_vals),
            "mean_risk": _avg(risk_vals),
            "mean_delta": _avg(deltas),
            "mean_confidence": _avg(conf_vals),
            "high_risk_count": int(high_risk_episodes),
            "requires_confirmation_count": int(governance_required),
        },
        "approvals": {
            "proposed": int(proposed),
            "approved_or_executed": int(approved),
            "executed": int(executed),
            "rejected": int(rejected),
            "approval_rate": round(float(approved) / float(denom), 4),
        },
    }


@router.post("/autonomy/feedback")
def autonomy_feedback(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    episode_id = int(payload.get("episode_id") or 0)
    decision = str(payload.get("decision") or "neutral").strip().lower()
    if decision == "approve":
        decision = "accepted"
    elif decision == "reject":
        decision = "rejected"
    if decision not in {"accepted", "rejected", "neutral"}:
        decision = "neutral"
    notes = str(payload.get("notes") or "")[:500]

    row = session.exec(
        select(AutonomyEpisode)
        .where(AutonomyEpisode.id == episode_id, AutonomyEpisode.user_id == user_id)
        .limit(1)
    ).first()
    if not row:
        return {"ok": False, "error": "episode_not_found"}

    feedback = AutonomyActionFeedback(
        user_id=user_id,
        episode_id=int(row.id or 0),
        action_name=str(row.chosen_action or ""),
        decision=decision,
        notes=notes,
    )
    session.add(feedback)

    # Closed-loop policy adjustment from user signal.
    genome = _autonomy_get_or_create_genome(session, user_id=user_id)
    try:
        g = json.loads(genome.genome_json or "{}")
    except Exception:
        g = {}
    action_bias = dict(g.get("action_bias") or {})
    action = str(row.chosen_action or "")
    cur = float(action_bias.get(action, 0.2))
    if decision == "accepted":
        cur = min(0.95, cur + 0.06)
    elif decision == "rejected":
        cur = max(0.01, cur - 0.08)
    action_bias[action] = cur
    g["action_bias"] = action_bias
    g["selection_pressure"] = float(g.get("selection_pressure", getattr(settings, "ecosystem_experiment_selection_pressure", 1.35)))
    genome.genome_json = json.dumps(g, sort_keys=True)
    genome.updated_at = datetime.utcnow()
    session.add(genome)
    session.commit()

    return {"ok": True, "episode_id": int(row.id or 0), "decision": decision, "action": action}

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
