#!/usr/bin/env python3
"""Weekly pruning protocol for Project Resonance.

Tasks:
1) Compact fragmented .noise artifacts into sediment packs.
2) Prune synthetic/test dwellers older than N days without unique DNA signature.
3) Audit long-lived dwellers for efficiency stagnation and re-inject to noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "raw_data"
AGENT_METABOLISM = ROOT / "agent_metabolism"
SECRETS = ROOT / ".secrets"

NOISE_REGISTER_PATH = RAW_DATA / "noise_register.json"
DWELLERS_REGISTRY_PATH = AGENT_METABOLISM / "dwellers_registry.json"
TRACE_LOG_PATH = SECRETS / "agent_trace_log.jsonl"


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
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else fallback
    except Exception:
        return fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_trace(record: dict[str, Any]) -> None:
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _dwellers(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(registry.get("dwellers") or [])
    return [row for row in rows if isinstance(row, dict)]


def _dna_signature(dweller: dict[str, Any]) -> str:
    identity = "|".join(
        [
            str(dweller.get("role") or "generalist"),
            f"{float(dweller.get('utility_signal', 0.0) or 0.0):.3f}",
            f"{float(dweller.get('metabolic_rate', 0.0) or 0.0):.3f}",
            f"{float(dweller.get('thermodynamic_tax', 0.0) or 0.0):.3f}",
            f"{float(dweller.get('volatility_score', 0.0) or 0.0):.3f}",
        ]
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def compact_noise(*, dry_run: bool, pack_size: int) -> dict[str, Any]:
    register = _load_json(NOISE_REGISTER_PATH, {"artifacts": []})
    artifacts = list(register.get("artifacts") or [])
    if len(artifacts) < max(2, pack_size):
        return {"packs_created": 0, "artifacts_compacted": 0}

    packs_created = 0
    compacted = 0
    remainder: list[dict[str, Any]] = []
    chunk: list[dict[str, Any]] = []

    for entry in artifacts:
        if not isinstance(entry, dict):
            continue
        chunk.append(entry)
        if len(chunk) >= pack_size:
            packs_created += 1
            compacted += len(chunk)
            digest = hashlib.sha1(
                "|".join([str(row.get("artifact") or row.get("created_at") or "") for row in chunk]).encode("utf-8")
            ).hexdigest()[:12]
            pack_name = f"sediment_pack_{_now().strftime('%Y%m%d')}_{packs_created}_{digest}.noise"
            pack_payload = {
                "kind": "sediment_pack",
                "created_at": _now_iso(),
                "count": len(chunk),
                "artifacts": chunk,
            }
            if not dry_run:
                (RAW_DATA / pack_name).write_text(
                    json.dumps(pack_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                remainder.append(
                    {
                        "artifact": pack_name,
                        "created_at": _now_iso(),
                        "reason": "noise_compacted_weekly",
                        "snapshot": {"pack_count": len(chunk)},
                    }
                )
            chunk = []

    # Keep leftover items that did not reach pack size.
    remainder.extend(chunk)
    if not dry_run:
        register["artifacts"] = remainder
        register["last_compacted_at"] = _now_iso()
        _write_json(NOISE_REGISTER_PATH, register)

    return {"packs_created": packs_created, "artifacts_compacted": compacted}


def prune_synthetic_dwellers(*, dry_run: bool, days: int) -> dict[str, Any]:
    registry = _load_json(DWELLERS_REGISTRY_PATH, {"dwellers": []})
    dwellers = _dwellers(registry)
    if not dwellers:
        return {"pruned": 0, "survivors": 0}

    cutoff = _now() - timedelta(days=max(1, int(days)))
    signature_counts: dict[str, int] = {}
    for row in dwellers:
        sig = _dna_signature(row)
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    survivors: list[dict[str, Any]] = []
    pruned_rows: list[dict[str, Any]] = []
    for row in dwellers:
        name = str(row.get("name") or row.get("id") or "").lower()
        is_synthetic = ("test" in name) or ("synthetic" in name) or ("candidate" in name)
        ts = _parse_iso(str(row.get("last_perturbed_at") or ""))
        stale = bool(ts and ts < cutoff)
        signature = _dna_signature(row)
        not_unique = signature_counts.get(signature, 0) > 1

        if is_synthetic and stale and not_unique:
            pruned_rows.append(row)
            continue
        survivors.append(row)

    if not dry_run and pruned_rows:
        registry["dwellers"] = survivors
        registry["last_pruned_at"] = _now_iso()
        _write_json(DWELLERS_REGISTRY_PATH, registry)
        for row in pruned_rows:
            _append_trace(
                {
                    "spec_version": "1.0",
                    "trace_id": hashlib.sha1(f"prune|{row.get('id')}|{_now_iso()}".encode("utf-8")).hexdigest()[:16],
                    "event_type": "weekly_synthetic_prune",
                    "occurred_at": _now_iso(),
                    "cycle_id": "weekly-pruning",
                    "entity": {
                        "id": str(row.get("id") or ""),
                        "name": str(row.get("name") or row.get("id") or ""),
                        "role": str(row.get("role") or ""),
                        "layer": "liquid",
                    },
                    "details": {"reason": "synthetic_stale_non_unique_dna"},
                }
            )

    return {"pruned": len(pruned_rows), "survivors": len(survivors)}


def audit_efficiency_and_reinject(*, dry_run: bool, min_survival_spikes: int) -> dict[str, Any]:
    registry = _load_json(DWELLERS_REGISTRY_PATH, {"dwellers": []})
    dwellers = _dwellers(registry)
    if not dwellers:
        return {"reinjected": 0}

    reinjected = 0
    register = _load_json(NOISE_REGISTER_PATH, {"artifacts": []})
    for row in dwellers:
        survival = int(row.get("survival_cycles", 0) or 0)
        if survival < max(1, int(min_survival_spikes)):
            continue
        utility = float(row.get("utility_signal", 0.0) or 0.0)
        metabolic = float(row.get("metabolic_rate", 0.2) or 0.2)
        tax = float(row.get("thermodynamic_tax", 0.2) or 0.2)
        efficiency = utility / max(0.01, metabolic + tax)
        plateau = efficiency < 1.2
        if not plateau:
            continue
        reinjected += 1
        artifact_name = (
            f"reinject_{str(row.get('id') or 'dweller').replace('/','-')}_{_now().strftime('%Y%m%d%H%M%S')}.noise"
        )
        artifact_payload = {
            "kind": "efficiency_reinjection",
            "created_at": _now_iso(),
            "reason": "survived_spikes_without_efficiency_gain",
            "snapshot": row,
        }
        if not dry_run:
            (RAW_DATA / artifact_name).write_text(
                json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            register.setdefault("artifacts", []).append(
                {
                    "artifact": artifact_name,
                    "created_at": _now_iso(),
                    "reason": "efficiency_audit_reinject",
                    "snapshot": row,
                }
            )
            _append_trace(
                {
                    "spec_version": "1.0",
                    "trace_id": hashlib.sha1(
                        f"reinject|{row.get('id')}|{_now_iso()}".encode("utf-8")
                    ).hexdigest()[:16],
                    "event_type": "weekly_efficiency_reinject",
                    "occurred_at": _now_iso(),
                    "cycle_id": "weekly-pruning",
                    "entity": {
                        "id": str(row.get("id") or ""),
                        "name": str(row.get("name") or row.get("id") or ""),
                        "role": str(row.get("role") or ""),
                        "layer": "liquid",
                    },
                    "details": {
                        "efficiency": round(efficiency, 4),
                        "min_survival_spikes": int(min_survival_spikes),
                    },
                }
            )

    if not dry_run:
        register["last_efficiency_audit_at"] = _now_iso()
        _write_json(NOISE_REGISTER_PATH, register)
    return {"reinjected": reinjected}


def run_protocol(*, dry_run: bool, sediment_pack_size: int, prune_days: int, min_survival_spikes: int) -> dict[str, Any]:
    compact = compact_noise(dry_run=dry_run, pack_size=sediment_pack_size)
    prune = prune_synthetic_dwellers(dry_run=dry_run, days=prune_days)
    audit = audit_efficiency_and_reinject(dry_run=dry_run, min_survival_spikes=min_survival_spikes)
    return {
        "ok": True,
        "dry_run": dry_run,
        "compaction": compact,
        "synthetic_prune": prune,
        "efficiency_audit": audit,
        "ran_at": _now_iso(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weekly Resonance pruning protocol.")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing files.")
    parser.add_argument("--sediment-pack-size", type=int, default=6, help="Number of noise artifacts per sediment pack.")
    parser.add_argument("--prune-days", type=int, default=7, help="Synthetic dwellers stale threshold in days.")
    parser.add_argument(
        "--min-survival-spikes",
        type=int,
        default=3,
        help="Minimum survival cycles for efficiency plateau audit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_protocol(
        dry_run=bool(args.dry_run),
        sediment_pack_size=max(2, int(args.sediment_pack_size)),
        prune_days=max(1, int(args.prune_days)),
        min_survival_spikes=max(1, int(args.min_survival_spikes)),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
