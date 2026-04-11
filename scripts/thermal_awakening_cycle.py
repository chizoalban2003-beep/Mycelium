#!/usr/bin/env python3
"""Thermal modulation + dissipative selection runtime for Project Resonance."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "raw_data"
AGENT_METABOLISM = ROOT / "agent_metabolism"
BEDROCK = ROOT / "crystallized_substrate"
SECRETS = ROOT / ".secrets"

REGISTRY_PATH = AGENT_METABOLISM / "dwellers_registry.json"
NOISE_REGISTER_PATH = RAW_DATA / "noise_register.json"
BEDROCK_MANIFEST_PATH = BEDROCK / "bedrock_manifest.json"
SECRETS_MANIFEST_PATH = SECRETS / "dwellers_manifest.md"

STALE_HOURS = 24


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def _load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else fallback
    except Exception:
        return fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def _dweller_name(dweller: dict[str, Any]) -> str:
    return str(dweller.get("name") or dweller.get("id") or "dweller-unknown")


def _normalize_dweller(dweller: dict[str, Any]) -> dict[str, Any]:
    row = dict(dweller)
    row["id"] = str(row.get("id") or _dweller_name(row))
    row["name"] = _dweller_name(row)
    row["role"] = str(row.get("role") or "generalist")
    row["status"] = str(row.get("status") or "active")
    row["survival_cycles"] = int(row.get("survival_cycles", 0) or 0)
    row["last_perturbed_at"] = str(row.get("last_perturbed_at") or _now_iso())
    row["utility_signal"] = float(row.get("utility_signal", row.get("utility_score", 0.5)) or 0.5)
    row["metabolic_rate"] = float(row.get("metabolic_rate", row.get("metabolic_cost", 0.2)) or 0.2)
    row["volatility_score"] = float(row.get("volatility_score", row.get("volatility", 0.5)) or 0.5)
    row["thermodynamic_tax"] = float(
        row.get(
            "thermodynamic_tax",
            (float(row.get("friction_score", 0.3) or 0.3) * 0.6),
        )
        or 0.2
    )
    row["mutation_seed"] = str(row.get("mutation_seed") or row["id"])
    row["requires_human_approval"] = bool(row.get("requires_human_approval", True))
    row["utility_signal"] = _clip(row["utility_signal"], 0.0, 2.0)
    row["metabolic_rate"] = _clip(row["metabolic_rate"], 0.01, 1.2)
    row["volatility_score"] = _clip(row["volatility_score"], 0.01, 0.99)
    row["thermodynamic_tax"] = _clip(row["thermodynamic_tax"], 0.01, 1.2)
    return row


@dataclass
class DwellerScore:
    row: dict[str, Any]
    score: float


def _dissipative_efficiency(dweller: dict[str, Any]) -> float:
    utility = float(dweller.get("utility_signal", 0.0) or 0.0)
    metabolic = float(dweller.get("metabolic_rate", 0.2) or 0.2)
    tax = float(dweller.get("thermodynamic_tax", 0.2) or 0.2)
    volatility = float(dweller.get("volatility_score", 0.5) or 0.5)
    resilience = 1.0 - (volatility * 0.2)
    return (utility / max(0.01, metabolic + tax)) * resilience


def _to_noise_artifact(dweller: dict[str, Any], spike_id: str, reason: str) -> dict[str, Any]:
    source = f"{dweller.get('id','dweller')}|{spike_id}|{reason}|{_now_iso()}"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    filename = f"{dweller.get('id','dweller')}_{digest}.noise".replace("/", "-")
    target = RAW_DATA / filename
    artifact = {
        "id": dweller.get("id"),
        "name": _dweller_name(dweller),
        "role": dweller.get("role"),
        "status": dweller.get("status"),
        "reason": reason,
        "spike_id": spike_id,
        "dissolved_at": _now_iso(),
        "snapshot": dweller,
    }
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return {
        "artifact": filename,
        "spike_id": spike_id,
        "created_at": _now_iso(),
        "snapshot": dweller,
        "reason": reason,
    }


def _mutate_from_noise(noise: dict[str, Any], ordinal: int, spike_id: str) -> dict[str, Any]:
    seed = str(noise.get("id") or noise.get("name") or f"noise-{ordinal}")
    role = str(noise.get("role") or "forager")
    base_utility = _clip(float(noise.get("utility_signal", 0.58) or 0.58), 0.2, 1.3)
    base_rate = _clip(float(noise.get("metabolic_rate", 0.24) or 0.24), 0.08, 0.8)
    base_tax = _clip(float(noise.get("thermodynamic_tax", 0.22) or 0.22), 0.04, 0.7)
    base_vol = _clip(float(noise.get("volatility_score", 0.55) or 0.55), 0.08, 0.95)

    return {
        "id": f"{seed}-mutation-{ordinal}",
        "name": f"{_dweller_name(noise)}-mutation-{ordinal}",
        "role": role,
        "status": "candidate",
        "origin": "mutation",
        "mutation_of": seed,
        "mutation_source": "raw_data_noise",
        "mutation_spike_id": spike_id,
        "survival_cycles": 0,
        "last_perturbed_at": _now_iso(),
        "requires_human_approval": True,
        "utility_signal": _clip(base_utility + random.uniform(-0.09, 0.13), 0.2, 1.5),
        "metabolic_rate": _clip(base_rate + random.uniform(-0.05, 0.06), 0.05, 0.95),
        "thermodynamic_tax": _clip(base_tax + random.uniform(-0.03, 0.05), 0.03, 0.95),
        "volatility_score": _clip(base_vol + random.uniform(-0.10, 0.14), 0.03, 0.98),
        "mutation_seed": seed,
    }


def run_cycle(*, spike_id: str, dry_run: bool) -> dict[str, Any]:
    registry = _load_json(REGISTRY_PATH, {"schema_version": 1, "dwellers": []})
    dwellers = [_normalize_dweller(row) for row in list(registry.get("dwellers") or [])]
    if not dwellers:
        return {"ok": False, "reason": "no_dwellers"}

    now = _now()
    stale_cutoff = now - timedelta(hours=STALE_HOURS)
    active_rows = [d for d in dwellers if str(d.get("status")) != "retired"]

    stale_rows = []
    for row in active_rows:
        ts = _parse_iso(str(row.get("last_perturbed_at") or ""))
        if ts is not None and ts < stale_cutoff:
            stale_rows.append(row)

    scored = [DwellerScore(row=r, score=_dissipative_efficiency(r)) for r in active_rows if r not in stale_rows]
    scored.sort(key=lambda item: item.score, reverse=True)
    n = len(scored)

    retire_count = max(1, int(round(n * 0.10))) if n else 0
    mutate_count = max(1, int(round(n * 0.10))) if n else 0
    bottom = scored[-retire_count:] if retire_count and scored else []
    top = scored[:mutate_count] if mutate_count and scored else []

    retire_names = {_dweller_name(item.row) for item in bottom}
    retire_names.update({_dweller_name(r) for r in stale_rows})

    next_population = []
    dissolved_artifacts: list[dict[str, Any]] = []
    for row in dwellers:
        current = dict(row)
        current["last_cycle_id"] = spike_id
        current["last_cycle_at"] = _now_iso()
        if _dweller_name(current) in retire_names:
            reason = "stale_24h" if current in stale_rows else "thermodynamic_tax_failure"
            current["status"] = "retired"
            current["retired_at"] = _now_iso()
            current["retired_reason"] = reason
            if not dry_run:
                dissolved_artifacts.append(_to_noise_artifact(current, spike_id, reason))
            next_population.append(current)
        elif current["status"] != "retired":
            current["status"] = "active"
            current["survival_cycles"] = int(current.get("survival_cycles", 0) or 0) + 1
            current["last_perturbed_at"] = _now_iso()
            next_population.append(current)
        else:
            next_population.append(current)

    # Mutation pool: prefer recent noise entries; fallback to top performers.
    noise_register = _load_json(NOISE_REGISTER_PATH, {"version": 1, "artifacts": []})
    noise_artifacts = list(noise_register.get("artifacts") or [])
    mutation_pool = []
    for entry in reversed(noise_artifacts):
        snapshot = entry.get("snapshot") if isinstance(entry, dict) else None
        if not isinstance(snapshot, dict):
            artifact_name = str(entry.get("artifact") or "") if isinstance(entry, dict) else ""
            artifact_path = RAW_DATA / artifact_name if artifact_name else None
            if artifact_path and artifact_path.exists():
                try:
                    noise_obj = json.loads(artifact_path.read_text())
                    snap_candidate = noise_obj.get("snapshot") if isinstance(noise_obj, dict) else None
                    snapshot = snap_candidate if isinstance(snap_candidate, dict) else None
                except Exception:
                    snapshot = None
        if isinstance(snapshot, dict):
            mutation_pool.append(_normalize_dweller(snapshot))
        if len(mutation_pool) >= mutate_count:
            break
    if len(mutation_pool) < mutate_count:
        mutation_pool.extend([item.row for item in top][: max(0, mutate_count - len(mutation_pool))])

    mutations = []
    for idx, source in enumerate(mutation_pool[:mutate_count], start=1):
        mutations.append(_mutate_from_noise(source, idx, spike_id))

    if not dry_run:
        for artifact_entry in dissolved_artifacts:
            noise_register.setdefault("artifacts", []).append(artifact_entry)
        noise_register["last_cycle_id"] = spike_id
        noise_register["last_updated_at"] = _now_iso()
        _write_json(NOISE_REGISTER_PATH, noise_register)

    updated = next_population + mutations
    promoted = []
    for row in updated:
        if int(row.get("survival_cycles", 0) or 0) >= 3 and str(row.get("status")) == "active":
            promoted.append(row)

    summary = {
        "ok": True,
        "dry_run": dry_run,
        "spike_id": spike_id,
        "population_before": len(dwellers),
        "population_after": len(updated),
        "retired_count": len(retire_names),
        "mutated_count": len(mutations),
        "dissolved_noise_count": len(dissolved_artifacts),
        "promoted_count": len(promoted),
        "retired_names": sorted(retire_names),
        "promoted_names": sorted([_dweller_name(r) for r in promoted]),
    }

    if dry_run:
        return summary

    registry["dwellers"] = updated
    registry["last_cycle_id"] = spike_id
    registry["last_selection_summary"] = summary
    registry["last_selection_cycle_at"] = _now_iso()
    _write_json(REGISTRY_PATH, registry)

    manifest = _load_json(
        BEDROCK_MANIFEST_PATH,
        {"version": 1, "immutable_modules": [], "selection_telemetry": {}, "last_selection_cycle_at": None},
    )
    immutables = list(manifest.get("immutable_modules") or [])
    known = {str(item.get("id") or "") for item in immutables if isinstance(item, dict)}
    for row in promoted:
        if row["id"] in known:
            continue
        immutables.append(
            {
                "id": row["id"],
                "name": _dweller_name(row),
                "role": row.get("role"),
                "immutable": True,
                "hardened_after_spike": spike_id,
                "survival_cycles": int(row.get("survival_cycles", 0) or 0),
                "recorded_at": _now_iso(),
            }
        )
    manifest["immutable_modules"] = immutables
    manifest["selection_telemetry"] = {
        "spike_id": spike_id,
        "population_before": len(dwellers),
        "population_after": len(updated),
        "retired_count": len(retire_names),
        "mutated_count": len(mutations),
        "promoted_count": len(promoted),
    }
    manifest["last_selection_cycle_at"] = _now_iso()
    _write_json(BEDROCK_MANIFEST_PATH, manifest)

    if not SECRETS_MANIFEST_PATH.exists():
        SECRETS_MANIFEST_PATH.write_text("# Dwellers Manifest\n\n")
    with SECRETS_MANIFEST_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Cycle {spike_id} @ {_now_iso()}\n")
        handle.write(f"- Retired: {', '.join(sorted(retire_names)) or 'none'}\n")
        handle.write(f"- Mutated: {len(mutations)} candidate dwellers\n")
        handle.write(
            f"- Hardened immutables: {', '.join(sorted([_dweller_name(r) for r in promoted])) or 'none'}\n"
        )
        handle.write("- Secret paths: encrypted/internal, functionality tracked here only.\n")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one thermal awakening cycle.")
    parser.add_argument("--spike-id", type=str, default="manual", help="Energy spike identifier.")
    parser.add_argument("--dry-run", action="store_true", help="Calculate cycle without writing files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(run_cycle(spike_id=str(args.spike_id), dry_run=bool(args.dry_run)), indent=2, sort_keys=True))
