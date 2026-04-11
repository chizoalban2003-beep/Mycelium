from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import (
    AutonomyEpisode,
    AutonomyPendingAction,
    GrowthLedgerEntry,
    NexusNudge,
    SignalLedgerEvent,
    User,
)


router = APIRouter(prefix="/api/resonance", tags=["resonance"])

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = ROOT / "raw_data"
AGENT_METABOLISM = ROOT / "agent_metabolism"
BEDROCK = ROOT / "crystallized_substrate"


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


def build_resonance_snapshot(*, session: Session, user_id: int, window_minutes: int = 120) -> dict:
    uid = int(user_id)
    window = max(10, min(int(window_minutes), 24 * 60))
    since = datetime.utcnow() - timedelta(minutes=window)

    signals = session.exec(
        select(SignalLedgerEvent).where(
            SignalLedgerEvent.created_by_user_id == uid,
            SignalLedgerEvent.created_at >= since,
        )
    ).all()
    growth = session.exec(
        select(GrowthLedgerEntry).where(
            GrowthLedgerEntry.created_by_user_id == uid,
            GrowthLedgerEntry.created_at >= since,
        )
    ).all()
    nudges = session.exec(
        select(NexusNudge).where(
            NexusNudge.created_by_user_id == uid,
            NexusNudge.created_at >= since,
        )
    ).all()
    episodes = session.exec(
        select(AutonomyEpisode).where(
            AutonomyEpisode.user_id == uid,
            AutonomyEpisode.created_at >= since,
        )
    ).all()
    pending = session.exec(
        select(AutonomyPendingAction).where(
            AutonomyPendingAction.user_id == uid,
            AutonomyPendingAction.status.in_(["proposed", "approved"]),
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
    if pending:
        avg_risk = sum(float(getattr(p, "risk_score", 0.0) or 0.0) for p in pending) / len(pending)

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
        "growth": len(growth),
        "nudges": len(nudges),
        "episodes": len(episodes),
        "pending_governance": len(pending),
        "retired_dwellers": len(retired),
        "avg_heat": round(avg_heat, 4),
        "avg_risk": round(avg_risk, 4),
    }

    recommendation = (
        "Run one thermal cycle and inspect mutations before hardening."
        if layers["liquid"]["count"] > layers["bedrock"]["count"]
        else "Bedrock is dominant; inject one controlled mutation spike to preserve adaptability."
    )
    heat_band = _heat_band(avg_heat)
    headline = f"Resonance Nexus is in {heat_band} band"
    story = (
        f"{layers['liquid']['count']} active dwellers and {layers['bedrock']['count']} crystallized modules are "
        f"processing {len(signals)} recent signals."
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
        "guardian_recommendation": recommendation,
        "heat": {"score": round(avg_heat, 4), "band": heat_band},
        "headline": headline,
        "story": story,
        "stats": {
            "signals_24h": len(signals),
            "active_dwellers": len(active),
            "bedrock_immutables": len(immutables),
            "unseen_nudges": len([n for n in nudges if getattr(n, "seen_at", None) is None]),
        },
    }


@router.get("/state-of-fluid")
@router.get("/state_of_fluid")
def state_of_fluid(
    window_minutes: int = 120,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    return build_resonance_snapshot(
        session=session,
        user_id=int(current_user.id or 0),
        window_minutes=window_minutes,
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
