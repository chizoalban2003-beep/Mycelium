from __future__ import annotations

from datetime import datetime, timedelta
import json
import random
from pathlib import Path
import subprocess

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import (
    AutonomyEpisode,
    SignalLedgerEvent,
    User,
)


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


def _load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else fallback
    except Exception:
        return fallback


def _heat_band(avg_heat: float) -> str:
    if avg_heat >= 0.75:
        return "ignite"
    if avg_heat >= 0.48:
        return "metabolize"
    if avg_heat >= 0.28:
        return "stabilize"
    return "cool"


def _autonomy_slider(*, avg_heat: float, avg_risk: float, active_count: int, immutable_count: int) -> float:
    # Higher when system sustains productive heat with manageable risk and stable sedimentation.
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
    last = {}
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
    if not lines:
        return []
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
    log_path = ROOT / ".secrets" / "agent_trace_log.jsonl"
    if not log_path.exists():
        return []
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    out: list[dict[str, object]] = []
    for line in lines[-max(1, min(int(limit), 200)) :]:
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
    # Small bounded jitter to simulate exogenous volatility.
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
            "Run dry thermal cycle first, then full cycle if stress response is healthy.",
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
                "feature": "Weekly pruning protocol",
                "status": "active",
                "effect": "Compacts noise, prunes stale synthetic dwellers, and reinjects efficiency plateaus.",
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
        ],
    }


def build_resonance_snapshot(
    *,
    session: Session,
    user_id: int,
    window_minutes: int = 120,
    secondary_force_override: float | None = None,
    secondary_source: str | None = None,
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

    registry = _load_json(
        AGENT_METABOLISM / "dwellers_registry.json",
        {"dwellers": [], "last_selection_summary": {}},
    )
    manifest = _load_json(
        BEDROCK / "bedrock_manifest.json",
        {"immutable_modules": [], "entries": []},
    )
    noise_register = _load_json(
        RAW_DATA / "noise_register.json",
        {"artifacts": []},
    )

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

    nodes = []
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
    if secondary_force_override is not None:
        secondary_force["coefficient"] = round(max(0.0, min(1.0, float(secondary_force_override))), 4)
        secondary_force["override_source"] = str(secondary_source or "api-override")
    secondary_force_coefficient = max(0.0, min(1.0, float(secondary_force.get("coefficient", 0.0) or 0.0)))
    secondary_force_level = "low"
    if secondary_force_coefficient >= 0.7:
        secondary_force_level = "high"
    elif secondary_force_coefficient >= 0.35:
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
    secondary_force["coefficient"] = round(secondary_force_coefficient, 4)
    secondary_force["level"] = secondary_force_level
    secondary_force["message"] = secondary_force_message
    secondary_force["stress_response"] = secondary_force_stress_response
    secondary_force["is_stressed"] = secondary_force_level in {"intermediate", "high"}
    base_heat_band = _heat_band(avg_heat)
    c_s = secondary_force_coefficient
    adaptive_heat = round(max(0.0, min(1.0, (avg_heat * 0.82) + (c_s * 0.18))), 4)
    heat_band = _heat_band(adaptive_heat)
    _append_stress_log(
        baseline_heat=round(avg_heat, 4),
        secondary_force=c_s,
        adaptive_heat=adaptive_heat,
        response=str(secondary_force.get("last_response") or "stable_flow"),
    )
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
            min(
                1.0,
                layers["bedrock"]["count"] / max(1.0, layers["liquid"]["count"] + layers["bedrock"]["count"]),
            ),
        ),
        4,
    )

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
        "recommendations": [
            recommendation,
            "Keep 24h thermal selection active and recycle tepid logic into `.noise` artifacts.",
            "Only sediment modules that survive >=3 spikes and lower entropy per compute unit.",
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


@router.post("/thermal_cycle")
def run_thermal_cycle(
    dry_run: bool = False,
    spike_id: str = "api",
    current_user: User = Depends(get_current_user),
):
    _ = current_user  # auth gate
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


@router.get("/trace")
def trace_explorer(
    limit: int = 25,
    current_user: User = Depends(get_current_user),
):
    _ = current_user  # auth gate
    recent = _trace_recent(limit=max(1, min(int(limit), 200)))
    return {
        "ok": True,
        "count": len(recent),
        "recent": recent,
    }


@router.get("/ui-config")
@router.get("/ui_config")
def ui_config(
    current_user: User = Depends(get_current_user),
):
    _ = current_user  # auth gate
    return {
        "ok": True,
        "config": _ui_config(),
    }


@router.get("/demo-state")
@router.get("/demo_state")
def demo_state(
    current_user: User = Depends(get_current_user),
):
    _ = current_user  # auth gate
    return {
        "ok": True,
        "demo": _demo_state_payload(),
    }


@router.get("/explain")
@router.get("/explain/full")
def explain_resonance(
    current_user: User = Depends(get_current_user),
):
    _ = current_user  # auth gate
    return {
        "ok": True,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "summary": (
            "Myco is an agentic ecosystem with Gas->Liquid->Bedrock flow. "
            "Signals enter as entropy, dwellers metabolize them, and resilient logic crystallizes."
        ),
        "architecture": {
            "gaseous": {
                "path": "raw_data/",
                "description": "Noise intake, dissolved modules, and sediment packs.",
            },
            "liquid": {
                "path": "agent_metabolism/",
                "description": "Active dwellers, secondary-force adaptation, and thermal selection.",
            },
            "bedrock": {
                "path": "crystallized_substrate/",
                "description": "Hardened low-entropy modules and immutable survivors.",
            },
        },
        "current_capabilities": [
            "Collects live signals and runs thermal selection loops.",
            "Tracks trace fossils and memory consolidation.",
            "Supports Resonance-only operator mode and thermal dashboard visibility.",
            "Supports deterministic operator demo state with JWT-grant metadata.",
        ],
        "future_updates": [
            "Secondary force calibration with bounded stress windows.",
            "Expanded pruning automation with scheduled sediment compaction and efficiency audits.",
            "Force-aware adaptive stress policies for autonomous refactor triggers.",
        ],
        "redundancy_candidates": [
            "Legacy non-Resonance pages that duplicate metrics already shown in Resonance dashboard.",
            "Old one-off monitoring widgets replaced by thermal dashboard + overview panel.",
            "Dormant pathways not mapped to Gas->Liquid->Bedrock lifecycle.",
        ],
    }
