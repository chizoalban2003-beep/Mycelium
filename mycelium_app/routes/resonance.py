from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import random
from pathlib import Path
import subprocess
from typing import Any

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import AutonomyEpisode, SignalLedgerEvent, User
from mycelium_app.open_world_simulation import evolve_world_state, world_replay_summary


router = APIRouter(prefix="/api/resonance", tags=["resonance"])

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = ROOT / "raw_data"
AGENT_METABOLISM = ROOT / "agent_metabolism"
BEDROCK = ROOT / "crystallized_substrate"
TRACE_LOG_PATH = ROOT / ".secrets" / "agent_trace_log.jsonl"
UI_CONFIG_PATH = ROOT / "nexus_ui_config.json"
MOCK_STATE_PATH = ROOT / "mock_nexus_state.json"
ENTROPY_LOGS = ROOT / "entropy_logs"
STRESS_TEST_LOG = ENTROPY_LOGS / "stress_tests.md"
RECOVERY_EVENTS_LOG = ENTROPY_LOGS / "recovery_events.md"
ACTUATOR_STATE_PATH = AGENT_METABOLISM / "actuator_state.json"
REDUNDANCY_REGISTRY_PATH = AGENT_METABOLISM / "redundancy_registry.json"
RUNTIME_STRESS_PATH = AGENT_METABOLISM / "runtime_stress_metrics.json"
INGESTION_GUARD_PATH = AGENT_METABOLISM / "ingestion_guard.json"
LEGACY_SNAPSHOT_DIR = RAW_DATA / "legacy_snapshots"
WORLD_STATE_PATH = AGENT_METABOLISM / "open_world_state.json"


def _load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else fallback
    except Exception:
        return fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _heat_band(avg_heat: float) -> str:
    if avg_heat >= 0.75:
        return "ignite"
    if avg_heat >= 0.48:
        return "metabolize"
    if avg_heat >= 0.28:
        return "stabilize"
    return "cool"


def _autonomy_slider(*, avg_heat: float, avg_risk: float, active_count: int, immutable_count: int) -> float:
    heat_component = max(0.0, min(1.0, avg_heat))
    risk_penalty = max(0.0, min(1.0, avg_risk))
    bedrock_ratio = max(0.0, min(1.0, (immutable_count + 1) / max(1.0, active_count + immutable_count + 1)))
    raw = (heat_component * 0.55) + (bedrock_ratio * 0.30) + ((1.0 - risk_penalty) * 0.15)
    return round(max(0.0, min(1.0, raw)), 4)


def _trace_summary() -> dict[str, object]:
    if not TRACE_LOG_PATH.exists():
        return {"events_total": 0, "last_event_type": None, "last_trace_id": None}
    lines = [line.strip() for line in TRACE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return {"events_total": 0, "last_event_type": None, "last_trace_id": None}
    try:
        last = json.loads(lines[-1])
    except Exception:
        last = {}
    return {
        "events_total": len(lines),
        "last_event_type": last.get("event_type"),
        "last_trace_id": last.get("trace_id"),
    }


def _trace_events(limit: int = 25) -> list[dict[str, object]]:
    if not TRACE_LOG_PATH.exists():
        return []
    lines = [line.strip() for line in TRACE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    events: list[dict[str, object]] = []
    for line in lines[-max(1, int(limit)):]:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        entity = row.get("entity") if isinstance(row.get("entity"), dict) else {}
        events.append(
            {
                "trace_id": row.get("trace_id"),
                "cycle_id": row.get("cycle_id"),
                "event_type": row.get("event_type"),
                "occurred_at": row.get("occurred_at"),
                "entity_id": entity.get("id"),
                "entity_name": entity.get("name"),
                "role": entity.get("role"),
            }
        )
    return list(reversed(events))


def _trace_recent(limit: int = 25) -> list[dict[str, object]]:
    if not TRACE_LOG_PATH.exists():
        return []
    lines = [line.strip() for line in TRACE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    out: list[dict[str, object]] = []
    for line in lines[-max(1, min(int(limit), 200)):]:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
        except Exception:
            continue
    return out


def _memory_summary() -> dict[str, object]:
    memory = _load_json(
        AGENT_METABOLISM / "resonance_memory.json",
        {"cycle_count": 0, "last_consolidated_at": None, "consolidated_memories": []},
    )
    memories = list(memory.get("consolidated_memories") or [])
    return {
        "cycle_count": int(memory.get("cycle_count", 0) or 0),
        "last_consolidated_at": memory.get("last_consolidated_at"),
        "consolidated_entries": len(memories),
    }


def _orchestration_profile() -> dict[str, object]:
    return {
        "mission_control": "Use Cursor cloud handoff (&) for long thermal runs.",
        "state_persistence": True,
        "multi_agent_friction": {
            "lava_agent": "Liquid mutation + exploration",
            "frost_agent": "Bedrock verification + hardening",
        },
        "recommended_runtime": "cloud-handoff",
    }


def _secondary_force_state() -> dict[str, object]:
    state = _load_json(
        AGENT_METABOLISM / "secondary_force_state.json",
        {
            "enabled": True,
            "coefficient": 0.1,
            "seed": 42,
            "last_spike_at": None,
            "last_response": "idle",
        },
    )
    coefficient = float(state.get("coefficient", 0.1) or 0.1)
    coefficient = max(0.0, min(1.0, coefficient + random.uniform(-0.025, 0.08)))
    state["coefficient"] = round(coefficient, 4)
    if coefficient >= 0.6:
        state["last_response"] = "critical_refactor"
        state["last_spike_at"] = datetime.utcnow().isoformat() + "Z"
    elif coefficient >= 0.35:
        state["last_response"] = "latency_trim"
    else:
        state["last_response"] = "stable_flow"
    return state


def _append_stress_log(*, baseline_heat: float, secondary_force: float, adaptive_heat: float, response: str) -> None:
    ENTROPY_LOGS.mkdir(parents=True, exist_ok=True)
    if not STRESS_TEST_LOG.exists():
        STRESS_TEST_LOG.write_text(
            "# Adaptive Stress Response Log\n\n"
            "| at | baseline_heat | secondary_force | adaptive_heat | response |\n"
            "| --- | ---: | ---: | ---: | --- |\n",
            encoding="utf-8",
        )
    with STRESS_TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            f"| {datetime.utcnow().isoformat()}Z | {baseline_heat:.4f} | "
            f"{secondary_force:.4f} | {adaptive_heat:.4f} | {response} |\n"
        )


def _append_recovery_log(record: dict[str, Any]) -> None:
    ENTROPY_LOGS.mkdir(parents=True, exist_ok=True)
    if not RECOVERY_EVENTS_LOG.exists():
        RECOVERY_EVENTS_LOG.write_text(
            "# Recovery Events Log\n\n"
            "| at | event_type | stress_level | action | details |\n"
            "| --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
    with RECOVERY_EVENTS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "| {at} | {event_type} | {stress_level} | {action} | {details} |\n".format(
                at=str(record.get("at") or datetime.utcnow().isoformat() + "Z"),
                event_type=str(record.get("event_type") or "actuator_event"),
                stress_level=str(record.get("stress_level") or "unknown"),
                action=str(record.get("action") or "none"),
                details=str(record.get("details") or "").replace("|", "/"),
            )
        )


def _runtime_metrics() -> dict[str, float]:
    payload = _load_json(
        RUNTIME_STRESS_PATH,
        {
            "latency_p95_ms": 240.0,
            "error_rate": 0.04,
            "queue_depth": 18.0,
        },
    )
    latency_norm = max(0.0, min(1.0, float(payload.get("latency_p95_ms", 240.0) or 240.0) / 1000.0))
    error_norm = max(0.0, min(1.0, float(payload.get("error_rate", 0.04) or 0.04)))
    queue_norm = max(0.0, min(1.0, float(payload.get("queue_depth", 18.0) or 18.0) / 120.0))
    return {
        "latency": round(latency_norm, 4),
        "error_rate": round(error_norm, 4),
        "queue_depth": round(queue_norm, 4),
    }


def _blend_secondary_force(*, synthetic: float, runtime: dict[str, float]) -> dict[str, float]:
    alpha, beta, gamma = 0.45, 0.35, 0.20
    latency = float(runtime.get("latency", 0.0) or 0.0)
    errors = float(runtime.get("error_rate", 0.0) or 0.0)
    blended = max(0.0, min(1.0, (alpha * synthetic) + (beta * latency) + (gamma * errors)))
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "synthetic": round(synthetic, 4),
        "latency": round(latency, 4),
        "error_rate": round(errors, 4),
        "value": round(blended, 4),
    }


def _current_actuator_state() -> dict[str, Any]:
    return _load_json(
        ACTUATOR_STATE_PATH,
        {
            "version": 1,
            "stress_consecutive_high": 0,
            "throttle_gaseous_ingestion": False,
            "trigger_emergency_crystallization": False,
            "metabolic_flush": False,
            "last_recovery_at": None,
            "last_action": "idle",
            "last_action_details": "",
            "high_stress_started_at": None,
        },
    )


def _write_ingestion_guard(*, throttled: bool, reason: str, stress_level: str) -> dict[str, Any]:
    guard = {
        "throttled": bool(throttled),
        "reason": str(reason or ""),
        "stress_level": str(stress_level or "unknown"),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(INGESTION_GUARD_PATH, guard)
    return guard


def _dweller_efficiency(dweller: dict[str, Any]) -> float:
    utility = float(dweller.get("utility_signal", 0.0) or 0.0)
    metabolic = float(dweller.get("metabolic_rate", 0.2) or 0.2)
    tax = float(dweller.get("thermodynamic_tax", 0.2) or 0.2)
    return utility / max(0.01, metabolic + tax)


def _append_trace_event(*, event_type: str, entity: dict[str, Any], details: dict[str, Any]) -> None:
    payload = {
        "spec_version": "1.0",
        "trace_id": hashlib.sha1(
            f"{event_type}|{entity.get('id')}|{datetime.utcnow().isoformat()}".encode("utf-8")
        ).hexdigest()[:16],
        "event_type": event_type,
        "occurred_at": datetime.utcnow().isoformat() + "Z",
        "cycle_id": "resonance-actuator",
        "entity": {
            "id": str(entity.get("id") or ""),
            "name": str(entity.get("name") or entity.get("id") or ""),
            "role": str(entity.get("role") or "generalist"),
            "layer": str(entity.get("layer") or "liquid"),
        },
        "details": details,
    }
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _promote_stable_dwellers(
    *,
    active_dwellers: list[dict[str, Any]],
    manifest: dict[str, Any],
    dry_run: bool,
    reason: str,
) -> list[str]:
    ranked = sorted(active_dwellers, key=_dweller_efficiency, reverse=True)[:3]
    immutable = list(manifest.get("immutable_modules") or [])
    known = {str(row.get("id") or "") for row in immutable if isinstance(row, dict)}
    promoted: list[str] = []
    for row in ranked:
        dweller_id = str(row.get("id") or "")
        if not dweller_id or dweller_id in known:
            continue
        promoted.append(dweller_id)
        immutable.append(
            {
                "id": dweller_id,
                "name": str(row.get("name") or dweller_id),
                "role": str(row.get("role") or "generalist"),
                "immutable": True,
                "hardened_after_spike": "actuator-emergency",
                "survival_cycles": int(row.get("survival_cycles", 0) or 0),
                "recorded_at": datetime.utcnow().isoformat() + "Z",
                "reason": reason,
            }
        )
        if not dry_run:
            _append_trace_event(
                event_type="emergency_crystallization",
                entity={
                    "id": dweller_id,
                    "name": str(row.get("name") or dweller_id),
                    "role": str(row.get("role") or "generalist"),
                    "layer": "bedrock",
                },
                details={"reason": reason, "source": "resonance_actuator"},
            )
    if promoted:
        manifest["immutable_modules"] = immutable
        manifest["last_selection_cycle_at"] = datetime.utcnow().isoformat() + "Z"
    return promoted


def _metabolic_flush(
    *,
    registry: dict[str, Any],
    manifest: dict[str, Any],
    dry_run: bool,
) -> dict[str, int]:
    dwellers = list(registry.get("dwellers") or [])
    immutable_ids = {
        str(row.get("id") or "")
        for row in list(manifest.get("immutable_modules") or [])
        if isinstance(row, dict)
    }
    flushed = 0
    for row in dwellers:
        dweller_id = str(row.get("id") or "")
        if dweller_id in immutable_ids:
            row["status"] = "active"
            continue
        if str(row.get("status") or "") == "retired":
            continue
        flushed += 1
        row["status"] = "retired"
        row["retired_reason"] = "metabolic_flush"
        row["retired_at"] = datetime.utcnow().isoformat() + "Z"
        if not dry_run:
            _append_trace_event(
                event_type="metabolic_flush_retire",
                entity={
                    "id": dweller_id,
                    "name": str(row.get("name") or dweller_id),
                    "role": str(row.get("role") or "generalist"),
                    "layer": "liquid",
                },
                details={"reason": "persistent_high_stress"},
            )
    return {"flushed": flushed, "retained": len(dwellers) - flushed}


def _legacy_snapshot(*, feature_row: dict[str, Any], reason: str) -> str:
    LEGACY_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    feature_name = str(feature_row.get("feature") or "feature").replace("/", "-").replace(" ", "_").lower()
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{feature_name}_{ts}.json"
    payload = {
        "feature": feature_row.get("feature"),
        "status": "deprecated",
        "reason": reason,
        "resonance_score": feature_row.get("resonance_score"),
        "below_threshold_cycles": feature_row.get("below_threshold_cycles"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "learned": "Feature cost exceeded value under sustained Resonance pressure.",
    }
    (LEGACY_SNAPSHOT_DIR / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return filename


def _sunset_registry() -> dict[str, Any]:
    baseline = {
        "version": 1,
        "cycles": [],
        "features": [
            {"feature": "legacy-chat-pane", "usage": 0.04, "cost": 0.71, "below_threshold_cycles": 1, "status": "candidate"},
            {"feature": "old-device-inspector", "usage": 0.06, "cost": 0.58, "below_threshold_cycles": 2, "status": "candidate"},
            {"feature": "duplicate-metric-widget", "usage": 0.02, "cost": 0.43, "below_threshold_cycles": 0, "status": "candidate"},
        ],
    }
    return _load_json(REDUNDANCY_REGISTRY_PATH, baseline)


def _run_sunset_protocol(*, dry_run: bool = False) -> dict[str, Any]:
    registry = _sunset_registry()
    features = [row for row in list(registry.get("features") or []) if isinstance(row, dict)]
    deprecated: list[dict[str, Any]] = []
    for row in features:
        usage = max(0.0, min(1.0, float(row.get("usage", 0.0) or 0.0)))
        cost = max(0.01, float(row.get("cost", 0.1) or 0.1))
        score = usage / cost
        row["resonance_score"] = round(score, 4)
        if score < 0.10:
            row["below_threshold_cycles"] = int(row.get("below_threshold_cycles", 0) or 0) + 1
        else:
            row["below_threshold_cycles"] = 0
        if int(row.get("below_threshold_cycles", 0) or 0) >= 2 and str(row.get("status") or "") != "deprecated":
            row["status"] = "deprecated"
            snapshot_file = _legacy_snapshot(feature_row=row, reason="sunset_protocol_low_resonance")
            row["legacy_snapshot"] = snapshot_file
            deprecated.append({"feature": row.get("feature"), "snapshot": snapshot_file})
    report = {
        "ok": True,
        "candidates": features,
        "deprecated": deprecated,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }
    if not dry_run:
        registry["features"] = features
        registry.setdefault("cycles", []).append(
            {
                "at": report["evaluated_at"],
                "deprecated_count": len(deprecated),
                "candidate_count": len(features),
            }
        )
        registry["cycles"] = registry["cycles"][-32:]
        _write_json(REDUNDANCY_REGISTRY_PATH, registry)
    return report


def _slo_health(*, actuation: dict[str, Any], liquid_count: int, bedrock_count: int) -> dict[str, Any]:
    stress_high_count = int(actuation.get("stress_consecutive_high", 0) or 0)
    high_stress_duration_min = stress_high_count * 2
    conversion_ratio = 0.0
    if liquid_count > 0:
        conversion_ratio = bedrock_count / max(1.0, float(liquid_count))
    mutation_yield = int(actuation.get("mutation_yield", 0) or 0)
    status = "healthy"
    if high_stress_duration_min >= 15 or conversion_ratio < 0.05 or mutation_yield < 2:
        status = "degraded"
    return {
        "overall_status": status,
        "metrics": {
            "high_stress_duration_min": high_stress_duration_min,
            "conversion_ratio": round(conversion_ratio, 4),
            "mutation_yield": mutation_yield,
        },
        "targets": {
            "high_stress_duration_min": "< 15",
            "conversion_ratio": "> 0.05",
            "mutation_yield": ">= 2",
        },
    }


def _apply_self_healing_actuation(
    *,
    stress_level: str,
    registry: dict[str, Any],
    manifest: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    state = _current_actuator_state()
    high = stress_level == "high"
    if high:
        state["stress_consecutive_high"] = int(state.get("stress_consecutive_high", 0) or 0) + 1
        if not state.get("high_stress_started_at"):
            state["high_stress_started_at"] = datetime.utcnow().isoformat() + "Z"
    else:
        state["stress_consecutive_high"] = 0
        state["high_stress_started_at"] = None
        state["trigger_emergency_crystallization"] = False
        state["metabolic_flush"] = False

    active = [row for row in list(registry.get("dwellers") or []) if isinstance(row, dict) and str(row.get("status") or "active") != "retired"]
    actions: list[str] = []
    details: dict[str, Any] = {}

    if int(state.get("stress_consecutive_high", 0) or 0) >= 5:
        state["throttle_gaseous_ingestion"] = True
        state["trigger_emergency_crystallization"] = True
        actions.append("throttle_gaseous_ingestion")
        promoted = _promote_stable_dwellers(
            active_dwellers=active,
            manifest=manifest,
            dry_run=dry_run,
            reason="high_stress_5_cycles",
        )
        details["promoted"] = promoted
        actions.append("trigger_emergency_crystallization")
        if int(state.get("stress_consecutive_high", 0) or 0) >= 8:
            state["metabolic_flush"] = True
            flush = _metabolic_flush(registry=registry, manifest=manifest, dry_run=dry_run)
            details["flush"] = flush
            actions.append("metabolic_flush")
    else:
        state["throttle_gaseous_ingestion"] = False
        state["trigger_emergency_crystallization"] = False
        state["metabolic_flush"] = False

    if not actions:
        state["last_action"] = "observe"
        state["last_action_details"] = "No self-healing action required."
    else:
        state["last_action"] = ",".join(actions)
        if len(details) == 1 and "promoted" in details:
            promoted = details.get("promoted") or []
            state["last_action_details"] = f"promoted={len(promoted)}"
        elif len(details) == 2 and "promoted" in details and "flush" in details:
            promoted = details.get("promoted") or []
            flush = details.get("flush") if isinstance(details.get("flush"), dict) else {}
            state["last_action_details"] = (
                f"promoted={len(promoted)},flushed={int(flush.get('flushed', 0) or 0)}"
            )
        else:
            state["last_action_details"] = json.dumps(details, sort_keys=True)
        state["last_recovery_at"] = datetime.utcnow().isoformat() + "Z"
        _append_recovery_log(
            {
                "at": state["last_recovery_at"],
                "event_type": "self_healing",
                "stress_level": stress_level,
                "action": state["last_action"],
                "details": state["last_action_details"],
            }
        )

    guard = _write_ingestion_guard(
        throttled=bool(state.get("throttle_gaseous_ingestion", False)),
        reason=str(state.get("last_action_details") or "none"),
        stress_level=stress_level,
    )
    state["ingestion_guard"] = guard
    state["mutation_yield"] = max(2, len(actions))
    if not dry_run:
        _write_json(ACTUATOR_STATE_PATH, state)
    return state


def _ui_config() -> dict[str, object]:
    fallback = {
        "version": 1,
        "resonance_only": {
            "enabled_by_default": False,
            "shortcut": "Mod+Shift+R",
            "hide_navigation_aux": True,
            "hide_notifications": True,
            "show_live_thermal_dashboard": True,
            "live_dashboard_refresh_ms": 10000,
        },
    }
    payload = _load_json(UI_CONFIG_PATH, fallback)
    if not isinstance(payload.get("resonance_only"), dict):
        payload["resonance_only"] = dict(fallback["resonance_only"])
    return payload


def _demo_state_payload() -> dict[str, object]:
    return _load_json(
        MOCK_STATE_PATH,
        {
            "mode": "molten",
            "headline": "Demo Mode: Molten Flux",
            "story": "Deterministic state for authenticated operator walkthroughs.",
            "layers": {
                "gaseous": {"count": 18},
                "liquid": {"count": 12},
                "bedrock": {"count": 6},
            },
            "heat": {"score": 0.72, "band": "metabolize"},
            "awe": {"autonomy_slider": 0.68, "state": "adapting"},
            "recommendations": [
                "Run dry thermal cycle to validate stress handling.",
                "Observe secondary force drift before full burn.",
            ],
        },
    )


def _build_overview_payload(*, secondary_descriptor: str) -> dict[str, object]:
    return {
        "name": "Project Resonance",
        "category": "Agentic Ecosystem / Living Substrate",
        "vision": "Turn entropy (signals) into adaptive logic and crystallized bedrock truths.",
        "current_mode": "Thermodynamic burn with human governance membrane",
        "operator_flow": [
            "Open /resonance and enable Resonance-only mode (Cmd/Ctrl+Shift+R).",
            "Watch Live Thermal Dashboard for heat and crystallization progress.",
            "Run Burn Safely for guarded dry-run + ignition + post-cycle diff.",
            "Review trace fossils and recommendations before any hardening decisions.",
        ],
        "layers": [
            {
                "name": "Gas (raw_data/)",
                "purpose": "Ingests noisy stimuli and stores dissolved artifacts for reuse.",
                "functions": ["noise intake", "sediment packs", "mutation source"],
            },
            {
                "name": "Liquid (agent_metabolism/)",
                "purpose": "Runs active dwellers under thermal pressure and adaptive stress.",
                "functions": ["selection", "mutation", "secondary-force response"],
            },
            {
                "name": "Bedrock (crystallized_substrate/)",
                "purpose": "Stores hardened low-entropy logic that survived repeated spikes.",
                "functions": ["sedimentation", "immutables", "stability guardrails"],
            },
        ],
        "feature_effects": [
            {
                "feature": "Resonance-only mode",
                "status": "active",
                "effect": "Hides non-framework UI noise and centers operator focus.",
            },
            {
                "feature": "Secondary Force stream",
                "status": "active",
                "effect": f"Injects adaptive stress ({secondary_descriptor}) into liquid heat dynamics.",
            },
            {
                "feature": "Closed-loop actuator dweller",
                "status": "active",
                "effect": "Throttles gaseous ingestion, forces emergency crystallization, and triggers flush under sustained stress.",
            },
            {
                "feature": "Sunset protocol",
                "status": "active",
                "effect": "Auto-deprecates low resonance-score features with legacy snapshots for auditability.",
            },
            {
                "feature": "Deterministic demo state",
                "status": "active",
                "effect": "Provides stable operator walkthrough visuals with JWT-based demo grant metadata.",
            },
        ],
        "future_updates": [
            "Closed-loop self-healing latency trims when secondary force remains high.",
            "Secondary force plugins for real API latency and market-volatility streams.",
            "Automated pruning + summary log shipping to external observability sink.",
            "Policy-based removal of redundant dashboards from operator default routes.",
            "Open-world deterministic simulation with infrastructure emergence and replay-ready ticks.",
        ],
    }


def _dwellers_for_world(registry: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(registry.get("dwellers") or []):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "status": row.get("status", "active"),
                "role": row.get("role", "generalist"),
                "volatility_score": row.get("volatility_score", 0.5),
                "utility_signal": row.get("utility_signal", row.get("fitness", 0.5)),
                "survival_cycles": row.get("survival_cycles", 0),
                "metabolic_rate": row.get("metabolic_rate", 0.25),
            }
        )
    return out


def _signals_for_world(signals: list[SignalLedgerEvent]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in signals[-220:]:
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(getattr(row, "payload_json", "") or "{}")
        except Exception:
            payload = {}
        surface = payload.get("surface") if isinstance(payload.get("surface"), dict) else payload
        out.append(
            {
                "signal_type": str(getattr(row, "signal_type", "") or ""),
                "app_name": str(surface.get("app_name") or surface.get("stimulus") or ""),
                "session_seconds": float(surface.get("session_seconds", 0.0) or 0.0),
            }
        )
    return out


def build_resonance_snapshot(
    *,
    session: Session,
    user_id: int,
    window_minutes: int = 120,
    secondary_force_override: float | None = None,
    secondary_source: str | None = None,
    world_ticks: int | None = None,
) -> dict:
    uid = int(user_id)
    window = max(10, min(int(window_minutes), 24 * 60))
    since = datetime.utcnow() - timedelta(minutes=window)

    signals = session.exec(
        select(SignalLedgerEvent).where(
            SignalLedgerEvent.created_by_user_id == uid,
            SignalLedgerEvent.created_at >= since,
        )
    ).all()
    episodes = session.exec(
        select(AutonomyEpisode).where(
            AutonomyEpisode.user_id == uid,
            AutonomyEpisode.created_at >= since,
        )
    ).all()

    registry = _load_json(AGENT_METABOLISM / "dwellers_registry.json", {"dwellers": [], "last_selection_summary": {}})
    manifest = _load_json(BEDROCK / "bedrock_manifest.json", {"immutable_modules": [], "entries": []})
    noise_register = _load_json(RAW_DATA / "noise_register.json", {"artifacts": []})

    dwellers = list(registry.get("dwellers") or [])
    active = [d for d in dwellers if str(d.get("status", "active")) != "retired"]
    retired = [d for d in dwellers if str(d.get("status", "")) == "retired"]
    immutables = list(manifest.get("immutable_modules") or [])
    if not immutables:
        immutables = [e for e in list(manifest.get("entries") or []) if isinstance(e, dict)]

    avg_heat = 0.0
    if episodes:
        avg_heat = sum(float(getattr(ep, "novelty_index", 0.0) or 0.0) for ep in episodes) / len(episodes)
    avg_risk = 0.0

    layers = {
        "gaseous": {
            "count": len(noise_register.get("artifacts") or []),
            "label": "raw_data",
            "entropy": round(min(1.0, (len(signals) / max(1.0, window / 5.0))), 4),
        },
        "liquid": {
            "count": len(active),
            "label": "agent_metabolism",
            "throughput": len(episodes),
            "coherence": round(max(0.0, min(1.0, 1.0 - avg_heat)), 4),
        },
        "bedrock": {
            "count": len(immutables),
            "label": "crystallized_substrate",
            "stability": round(max(0.0, min(1.0, (len(immutables) + 1) / max(1, len(active) + len(immutables) + 1))), 4),
        },
    }

    nodes: list[dict[str, Any]] = []
    for idx, dweller in enumerate(active[:80]):
        role = str(dweller.get("role") or "generalist")
        vol = float(dweller.get("volatility_score", 0.5) or 0.5)
        utility = float(dweller.get("utility_signal", 0.5) or 0.5)
        cycles = int(dweller.get("survival_cycles", 0) or 0)
        if cycles >= 3:
            layer = "bedrock"
            y = -12.0 - min(9.0, cycles * 0.9)
        elif vol >= 0.7:
            layer = "gaseous"
            y = 12.0 + min(10.0, vol * 9.0)
        else:
            layer = "liquid"
            y = 0.0 + (utility - 0.5) * 8.0
        x = ((idx % 14) - 7) * 3.4
        z = ((idx // 14) - 2) * 3.6
        nodes.append(
            {
                "id": str(dweller.get("id") or f"dweller-{idx}"),
                "name": str(dweller.get("name") or f"Dweller {idx + 1}"),
                "role": role,
                "layer": layer,
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "mass": round(0.8 + utility * 1.8, 4),
                "volatility": round(vol, 4),
                "energy": round(utility / max(0.05, float(dweller.get("metabolic_rate", 0.2) or 0.2)), 4),
            }
        )

    summary = {
        "signals": len(signals),
        "episodes": len(episodes),
        "retired_dwellers": len(retired),
        "avg_heat": round(avg_heat, 4),
        "avg_risk": round(avg_risk, 4),
    }

    recommendation = (
        "Run one thermal cycle to metabolize new noise into liquid candidates."
        if layers["liquid"]["count"] > layers["bedrock"]["count"]
        else "Inject one controlled mutation spike to preserve adaptive pressure."
    )

    secondary_force = _secondary_force_state()
    runtime = _runtime_metrics()
    if secondary_force_override is not None:
        synthetic = round(max(0.0, min(1.0, float(secondary_force_override))), 4)
        secondary_force["coefficient"] = synthetic
        secondary_force["override_source"] = str(secondary_source or "api-override")
    synthetic = float(secondary_force.get("coefficient", 0.0) or 0.0)
    if secondary_force_override is not None:
        blend = {
            "alpha": 1.0,
            "beta": 0.0,
            "gamma": 0.0,
            "synthetic": round(synthetic, 4),
            "latency": round(float(runtime.get("latency", 0.0) or 0.0), 4),
            "error_rate": round(float(runtime.get("error_rate", 0.0) or 0.0), 4),
            "value": round(synthetic, 4),
            "mode": "override",
        }
    else:
        blend = _blend_secondary_force(synthetic=synthetic, runtime=runtime)
    c_s = float(blend["value"])

    secondary_force_level = "low"
    if c_s >= 0.7:
        secondary_force_level = "high"
    elif c_s >= 0.35:
        secondary_force_level = "intermediate"

    secondary_force_message = (
        "External stress is low; dwellers should prioritize efficient growth."
        if secondary_force_level == "low"
        else "External stress is moderate; dwellers should trim latency and rebalance heat."
        if secondary_force_level == "intermediate"
        else "External stress is high; trigger self-healing and protect bedrock paths."
    )
    secondary_force_stress_response = (
        "No emergency response needed."
        if secondary_force_level == "low"
        else "Adaptive trim: reduce liquid-path latency and recycle noisy branches."
        if secondary_force_level == "intermediate"
        else "Self-healing: refactor critical paths and harden collapse-risk boundaries."
    )
    secondary_force["coefficient"] = round(c_s, 4)
    secondary_force["level"] = secondary_force_level
    secondary_force["message"] = secondary_force_message
    secondary_force["stress_response"] = secondary_force_stress_response
    secondary_force["is_stressed"] = secondary_force_level in {"intermediate", "high"}
    secondary_force["blend"] = blend
    secondary_force["runtime"] = runtime

    base_heat_band = _heat_band(avg_heat)
    adaptive_heat = round(max(0.0, min(1.0, (avg_heat * 0.82) + (c_s * 0.18))), 4)
    heat_band = _heat_band(adaptive_heat)
    _append_stress_log(
        baseline_heat=round(avg_heat, 4),
        secondary_force=c_s,
        adaptive_heat=adaptive_heat,
        response=str(secondary_force.get("last_response") or "stable_flow"),
    )

    actuation = _apply_self_healing_actuation(
        stress_level=secondary_force_level,
        registry=registry,
        manifest=manifest,
        dry_run=False,
    )

    if actuation.get("trigger_emergency_crystallization") or actuation.get("metabolic_flush"):
        _write_json(BEDROCK / "bedrock_manifest.json", manifest)
        _write_json(AGENT_METABOLISM / "dwellers_registry.json", registry)
        immutables = list(manifest.get("immutable_modules") or immutables)
        active = [d for d in list(registry.get("dwellers") or []) if str(d.get("status", "active")) != "retired"]
        retired = [d for d in list(registry.get("dwellers") or []) if str(d.get("status", "")) == "retired"]
        layers["liquid"]["count"] = len(active)
        layers["bedrock"]["count"] = len(immutables)

    autonomy_slider = _autonomy_slider(
        avg_heat=adaptive_heat,
        avg_risk=avg_risk,
        active_count=len(active),
        immutable_count=len(immutables),
    )
    awe_state = "alive" if autonomy_slider >= 0.75 else "adapting" if autonomy_slider >= 0.45 else "forming"
    headline = f"State of the Fluid: {heat_band}"
    story = (
        f"{layers['gaseous']['count']} gas artifacts feed {layers['liquid']['count']} liquid dwellers, "
        f"with {layers['bedrock']['count']} bedrock modules stabilizing the cycle."
    )
    liquid_flow_heat = round(
        max(
            0.0,
            min(
                1.0,
                (adaptive_heat * 0.65)
                + (min(1.0, layers["liquid"]["throughput"] / max(1.0, layers["liquid"]["count"] + 2)) * 0.35),
            ),
        ),
        4,
    )
    bedrock_progress = round(
        max(
            0.0,
            min(1.0, layers["bedrock"]["count"] / max(1.0, layers["liquid"]["count"] + layers["bedrock"]["count"])),
        ),
        4,
    )
    prev_world = _load_json(WORLD_STATE_PATH, {"version": 2, "entities": [], "infrastructure": {"nodes": [], "links": []}})
    world_tick_budget = max(1, min(int(world_ticks or (2 if secondary_force_level == "high" else 1)), 24))
    world_state = evolve_world_state(
        existing_state=prev_world,
        dwellers=_dwellers_for_world(registry),
        signals=_signals_for_world(signals),
        seed_key=f"{uid}:{window}:{len(signals)}:{len(active)}",
        secondary_force=c_s,
        adaptive_heat=adaptive_heat,
        ticks=world_tick_budget,
    )
    _write_json(WORLD_STATE_PATH, world_state)
    open_world = {
        "state": world_state,
        "ticks_applied": int(world_tick_budget),
        "phase": str(world_state.get("phase") or "forming"),
        "metrics": world_state.get("metrics") if isinstance(world_state.get("metrics"), dict) else {},
        "infrastructure_counts": {
            "nodes": len(list(((world_state.get("infrastructure") or {}).get("nodes") or []))),
            "links": len(list(((world_state.get("infrastructure") or {}).get("links") or []))),
        },
    }
    world_metrics = world_state.get("metrics") if isinstance(world_state.get("metrics"), dict) else {}
    world_health = {
        "score": float((world_metrics.get("world_health_score", 0.0) or 0.0)),
        "status": str((world_metrics.get("world_health_status") or "forming")),
        "infrastructure_density": float((world_metrics.get("infrastructure_density", 0.0) or 0.0)),
        "event_birth_rate": float((world_metrics.get("event_birth_rate", 0.0) or 0.0)),
        "specialization_balance": float((world_metrics.get("specialization_balance", 0.0) or 0.0)),
    }

    sunset = _run_sunset_protocol(dry_run=False)
    slo = _slo_health(actuation=actuation, liquid_count=layers["liquid"]["count"], bedrock_count=layers["bedrock"]["count"])

    return {
        "ok": True,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "window_minutes": window,
        "layers": layers,
        "summary": summary,
        "nodes": nodes,
        "selection": registry.get("last_selection_summary") or {},
        "bedrock_manifest": {
            "immutable_count": len(immutables),
            "last_selection_cycle_at": manifest.get("last_selection_cycle_at"),
        },
        "architect_recommendation": recommendation,
        "heat": {"score": adaptive_heat, "band": heat_band, "baseline_band": base_heat_band},
        "secondary_force": secondary_force,
        "actuation": actuation,
        "sunset_protocol": sunset,
        "sunset_governance": {
            "candidates": len(list(sunset.get("candidates") or [])),
            "deprecated_total": len(list(sunset.get("deprecated") or [])),
            "evaluated_at": sunset.get("evaluated_at"),
        },
        "slo_health": slo,
        "slo": {
            "health_score": (
                1.0
                if str(slo.get("overall_status") or "") == "healthy"
                else 0.55
                if str(slo.get("overall_status") or "") == "degraded"
                else 0.0
            ),
            "high_stress_duration_minutes": (slo.get("metrics") or {}).get("high_stress_duration_min"),
            "conversion_ratio_weekly": (slo.get("metrics") or {}).get("conversion_ratio"),
            "mutation_yield_cycle": (slo.get("metrics") or {}).get("mutation_yield"),
            "status": slo.get("overall_status"),
        },
        "awe": {
            "autonomy_slider": autonomy_slider,
            "state": awe_state,
            "message": "Composite of thermal persistence, risk membrane, and bedrock sedimentation.",
        },
        "trace": _trace_summary(),
        "trace_explorer": _trace_events(limit=25),
        "memory": _memory_summary(),
        "orchestration": _orchestration_profile(),
        "ui_config": _ui_config(),
        "thermal_dashboard": {
            "liquid_flow": {
                "path": "/liquid_flow",
                "heat_level": liquid_flow_heat,
                "active_dwellers": layers["liquid"]["count"],
                "throughput": layers["liquid"]["throughput"],
                "secondary_force_coefficient": c_s,
                "stress_level": secondary_force_level,
            },
            "bedrock": {
                "path": "/bedrock",
                "crystallization_progress": bedrock_progress,
                "immutable_count": layers["bedrock"]["count"],
                "stability": layers["bedrock"]["stability"],
            },
        },
        "overview": _build_overview_payload(secondary_descriptor=f"{secondary_force_level} ({c_s:.2f})"),
        "headline": headline,
        "story": story,
        "open_world": open_world,
        "world_health": world_health,
        "recommendations": [
            recommendation,
            "Keep 24h thermal selection active and recycle tepid logic into `.noise` artifacts.",
            "Only sediment modules that survive >=3 spikes and lower entropy per compute unit.",
            "Use Run Burn Safely when stress remains high for multiple cycles.",
            "Use `/api/resonance/world-step?ticks=3` during stress spikes to observe infrastructure adaptation in real time.",
            "Use `/api/resonance/world-replay` to compare world-health deltas across tick windows.",
        ],
        "stats": {
            "signals_window": len(signals),
            "active_dwellers": len(active),
            "bedrock_immutables": len(immutables),
            "autonomy_episodes": len(episodes),
        },
    }


@router.get("/state-of-fluid")
@router.get("/state_of_fluid")
def state_of_fluid(
    window_minutes: int = 120,
    secondary_force: float | None = None,
    secondary_source: str | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    return build_resonance_snapshot(
        session=session,
        user_id=int(current_user.id or 0),
        window_minutes=window_minutes,
        secondary_force_override=secondary_force,
        secondary_source=secondary_source,
    )


@router.get("/world-state")
@router.get("/world_state")
def world_state(
    ticks: int = 1,
    secondary_force: float | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    ticks = max(1, min(int(ticks or 1), 24))
    snapshot = build_resonance_snapshot(
        session=session,
        user_id=int(current_user.id or 0),
        window_minutes=180,
        secondary_force_override=secondary_force,
        secondary_source="world-state",
        world_ticks=ticks,
    )
    world = snapshot.get("open_world") if isinstance(snapshot, dict) else {}
    if not isinstance(world, dict):
        world = {}
    ticks_applied = int(world.get("ticks_applied", ticks) or ticks)
    return {"ok": True, "ticks_applied": ticks_applied, "world": world}


@router.post("/world-step")
@router.post("/world_step")
def world_step(
    ticks: int = 4,
    secondary_force: float | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    ticks = max(1, min(int(ticks or 4), 24))
    snapshot = build_resonance_snapshot(
        session=session,
        user_id=int(current_user.id or 0),
        window_minutes=180,
        secondary_force_override=secondary_force,
        secondary_source="world-step",
        world_ticks=ticks,
    )
    open_world = snapshot.get("open_world") if isinstance(snapshot, dict) else {}
    if not isinstance(open_world, dict):
        open_world = {}
    current = open_world.get("state") if isinstance(open_world.get("state"), dict) else {}
    return {
        "ok": True,
        "tick": int(current.get("tick", 0) or 0),
        "phase": str(current.get("phase") or "forming"),
        "ticks_applied": int(open_world.get("ticks_applied", ticks) or ticks),
        "metrics": current.get("metrics") if isinstance(current.get("metrics"), dict) else {},
        "infrastructure": current.get("infrastructure") if isinstance(current.get("infrastructure"), dict) else {"nodes": [], "links": []},
        "recent_events": list(current.get("events") or [])[-8:],
    }


@router.get("/world-infrastructure")
@router.get("/world_infrastructure")
def world_infrastructure(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    snapshot = build_resonance_snapshot(
        session=session,
        user_id=int(current_user.id or 0),
        window_minutes=180,
        secondary_source="world-infrastructure",
    )
    open_world = snapshot.get("open_world") if isinstance(snapshot, dict) else {}
    if not isinstance(open_world, dict):
        open_world = {}
    state = open_world.get("state") if isinstance(open_world.get("state"), dict) else {}
    infra = state.get("infrastructure") if isinstance(state.get("infrastructure"), dict) else {"nodes": [], "links": []}
    return {
        "ok": True,
        "as_of": state.get("as_of"),
        "tick": int(state.get("tick", 0) or 0),
        "phase": str(state.get("phase") or "forming"),
        "nodes": list(infra.get("nodes") or []),
        "links": list(infra.get("links") or []),
    }


@router.get("/world-replay")
@router.get("/world_replay")
def world_replay(
    start_tick: int | None = None,
    end_tick: int | None = None,
    window_ticks: int = 24,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    snapshot = build_resonance_snapshot(
        session=session,
        user_id=int(current_user.id or 0),
        window_minutes=180,
        secondary_source="world-replay",
    )
    open_world = snapshot.get("open_world") if isinstance(snapshot, dict) else {}
    if not isinstance(open_world, dict):
        open_world = {}
    state = open_world.get("state") if isinstance(open_world.get("state"), dict) else {}
    current_tick = int((state or {}).get("tick", 0) or 0)
    replay_window = max(2, min(int(window_ticks or 24), 240))
    resolved_start = start_tick
    resolved_end = end_tick
    if resolved_start is None and resolved_end is None and current_tick > 0:
        resolved_end = current_tick
        resolved_start = max(1, current_tick - replay_window + 1)
    replay = world_replay_summary(
        state=state if isinstance(state, dict) else {},
        start_tick=resolved_start,
        end_tick=resolved_end,
    )
    return {
        "ok": True,
        "phase": str((state or {}).get("phase") or "forming"),
        "current_tick": current_tick,
        "window_ticks": replay_window,
        "resolved_window": {
            "start_tick": int(resolved_start) if resolved_start is not None else None,
            "end_tick": int(resolved_end) if resolved_end is not None else None,
        },
        "world_health": snapshot.get("world_health") if isinstance(snapshot.get("world_health"), dict) else {},
        "replay": replay,
    }


@router.post("/thermal_cycle")
def run_thermal_cycle(
    dry_run: bool = False,
    spike_id: str = "api",
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    script_path = ROOT / "scripts" / "thermal_awakening_cycle.py"
    cmd = ["python3", str(script_path), "--spike-id", str(spike_id)]
    if dry_run:
        cmd.append("--dry-run")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0:
            return {"ok": False, "detail": "thermal cycle failed", "stderr": proc.stderr.strip()}
        payload = json.loads(proc.stdout or "{}")
        if not isinstance(payload, dict):
            payload = {"ok": False, "detail": "invalid thermal cycle output"}
        return payload
    except Exception as exc:
        return {"ok": False, "detail": f"thermal cycle error: {type(exc).__name__}: {exc}"}


@router.post("/run-burn-safely")
def run_burn_safely(
    secondary_force: float | None = None,
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    dry = run_thermal_cycle(dry_run=True, spike_id="burn-safe-dry", current_user=current_user)
    manifest = _load_json(BEDROCK / "bedrock_manifest.json", {"immutable_modules": []})
    bedrock_ok = bool(list(manifest.get("immutable_modules") or [])) or bool(manifest.get("immutable", False))
    if not bedrock_ok:
        return {
            "ok": False,
            "detail": "bedrock integrity check failed",
            "steps": {"dry_run": dry, "bedrock_health": {"ok": False}},
        }
    ignition = run_thermal_cycle(dry_run=False, spike_id="burn-safe-ignite", current_user=current_user)
    summary = {
        "burned": int(ignition.get("retired_count", 0) or 0),
        "forged": int(ignition.get("promoted_count", 0) or 0),
        "secrets_born": int(ignition.get("mutated_count", 0) or 0),
    }
    return {
        "ok": bool(ignition.get("ok", False)),
        "steps": {
            "dry_run": dry,
            "bedrock_health": {"ok": True, "immutable_count": len(list(manifest.get("immutable_modules") or []))},
            "ignition": ignition,
            "revelation": summary,
            "secondary_force_override": secondary_force,
        },
    }


@router.get("/slo-health")
@router.get("/slo_health")
def slo_health(
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    actuation = _current_actuator_state()
    registry = _load_json(AGENT_METABOLISM / "dwellers_registry.json", {"dwellers": []})
    manifest = _load_json(BEDROCK / "bedrock_manifest.json", {"immutable_modules": []})
    liquid = len([r for r in list(registry.get("dwellers") or []) if str(r.get("status", "active")) != "retired"])
    bedrock = len(list(manifest.get("immutable_modules") or []))
    return {"ok": True, "health": _slo_health(actuation=actuation, liquid_count=liquid, bedrock_count=bedrock)}


@router.get("/sunset-report")
@router.get("/sunset_report")
def sunset_report(
    dry_run: bool = True,
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    report = _run_sunset_protocol(dry_run=bool(dry_run))
    return {
        "ok": True,
        "evaluated_at": report.get("evaluated_at"),
        "candidates": report.get("candidates", []),
        "deprecated": report.get("deprecated", []),
    }


@router.get("/trace")
def trace_explorer(
    limit: int = 25,
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    recent = _trace_recent(limit=max(1, min(int(limit), 200)))
    return {"ok": True, "count": len(recent), "recent": recent}


@router.get("/ui-config")
@router.get("/ui_config")
def ui_config(
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    return {"ok": True, "config": _ui_config()}


@router.get("/demo-state")
@router.get("/demo_state")
def demo_state(
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    return {"ok": True, "demo": _demo_state_payload()}


@router.get("/explain")
@router.get("/explain/full")
def explain_resonance(
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    return {
        "ok": True,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "summary": (
            "Myco is an agentic ecosystem with Gas->Liquid->Bedrock flow. "
            "Signals enter as entropy, dwellers metabolize them, and resilient logic crystallizes."
        ),
        "architecture": {
            "gaseous": {"path": "raw_data/", "description": "Noise intake, dissolved modules, and sediment packs."},
            "liquid": {"path": "agent_metabolism/", "description": "Active dwellers, secondary-force adaptation, and thermal selection."},
            "bedrock": {"path": "crystallized_substrate/", "description": "Hardened low-entropy modules and immutable survivors."},
        },
        "current_capabilities": [
            "Collects live signals and runs thermal selection loops.",
            "Tracks trace fossils and memory consolidation.",
            "Supports Resonance-only operator mode and thermal dashboard visibility.",
            "Supports deterministic operator demo state with JWT-grant metadata.",
            "Runs actuator-driven self-healing and sunset governance protocols.",
        ],
        "future_updates": [
            "Secondary force calibration with bounded stress windows.",
            "Expanded pruning automation with scheduled sediment compaction and efficiency audits.",
            "Force-aware adaptive stress policies for autonomous refactor triggers.",
            "Live latency/error adapters from production telemetry providers.",
        ],
        "redundancy_candidates": [
            "Legacy non-Resonance pages that duplicate metrics already shown in Resonance dashboard.",
            "Old one-off monitoring widgets replaced by thermal dashboard + overview panel.",
            "Dormant pathways not mapped to Gas->Liquid->Bedrock lifecycle.",
        ],
    }
