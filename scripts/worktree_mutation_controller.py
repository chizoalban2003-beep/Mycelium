#!/usr/bin/env python3
"""Zero-downtime sedimentation orchestrator using git worktrees.

Creates isolated mutation branches/worktrees, runs thermal cycle mutations,
and leaves each mutation in a separate branch for evaluation/merge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
WORKTREE_ROOT = ROOT / ".worktrees"
REGISTRY_PATH = ROOT / "agent_metabolism" / "dwellers_registry.json"
TRACE_LOG = ROOT / ".secrets" / "agent_trace_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else fallback
    except Exception:
        return fallback


def _append_trace(event: dict[str, Any]) -> None:
    TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _candidate_dwellers(limit: int) -> list[dict[str, Any]]:
    registry = _load_json(REGISTRY_PATH, {"dwellers": []})
    dwellers = list(registry.get("dwellers") or [])
    candidates = [d for d in dwellers if str(d.get("status", "")).lower() == "candidate"]
    candidates.sort(
        key=lambda d: float(d.get("utility_signal", 0.0) or 0.0)
        / max(0.01, float(d.get("metabolic_rate", 0.2) or 0.2)),
        reverse=True,
    )
    return candidates[: max(0, int(limit))]


def _ensure_clean_main(base_branch: str) -> None:
    _run(["git", "checkout", base_branch], cwd=ROOT)
    _run(["git", "pull", "origin", base_branch], cwd=ROOT)


def orchestrate(*, base_branch: str, max_candidates: int, namespace: str) -> dict[str, Any]:
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_clean_main(base_branch)

    candidates = _candidate_dwellers(max_candidates)
    created: list[dict[str, Any]] = []

    for idx, dweller in enumerate(candidates, start=1):
        dweller_id = str(dweller.get("id") or f"candidate-{idx}")
        branch_name = f"cursor/{namespace}-{dweller_id[:24]}-{idx}-ea4f"
        wt_dir = WORKTREE_ROOT / branch_name.replace("/", "__")

        # Remove stale worktree path if git already cleaned ref.
        if wt_dir.exists():
            _run(["git", "worktree", "remove", str(wt_dir), "--force"], cwd=ROOT)

        add = _run(["git", "worktree", "add", "-b", branch_name, str(wt_dir), base_branch], cwd=ROOT)
        if add.returncode != 0:
            created.append(
                {
                    "dweller_id": dweller_id,
                    "branch": branch_name,
                    "status": "failed",
                    "error": (add.stderr or add.stdout).strip(),
                }
            )
            continue

        spike_id = f"worktree-{dweller_id}-{idx}"
        cycle = _run(
            ["python3", "scripts/thermal_awakening_cycle.py", "--spike-id", spike_id, "--dry-run"],
            cwd=wt_dir,
        )
        cycle_payload: dict[str, Any]
        try:
            cycle_payload = json.loads(cycle.stdout or "{}")
        except Exception:
            cycle_payload = {"ok": False, "detail": "invalid_cycle_output"}

        created.append(
            {
                "dweller_id": dweller_id,
                "branch": branch_name,
                "worktree": str(wt_dir),
                "status": "created",
                "cycle_summary": cycle_payload,
            }
        )
        _append_trace(
            {
                "spec_version": "1.0",
                "trace_id": f"wt-{dweller_id}-{idx}-{int(datetime.now(timezone.utc).timestamp())}",
                "event_type": "worktree_mutation_spawned",
                "occurred_at": _now_iso(),
                "cycle_id": spike_id,
                "entity": {
                    "id": dweller_id,
                    "name": str(dweller.get("name") or dweller_id),
                    "role": str(dweller.get("role") or ""),
                    "layer": "liquid",
                },
                "details": {
                    "branch": branch_name,
                    "worktree": str(wt_dir),
                    "base_branch": base_branch,
                },
            }
        )

    return {
        "ok": True,
        "base_branch": base_branch,
        "namespace": namespace,
        "created_count": len([c for c in created if c.get("status") == "created"]),
        "attempted_count": len(created),
        "worktrees": created,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn mutation worktrees for candidate dwellers.")
    parser.add_argument("--base-branch", default="main", help="Base branch for mutation worktrees.")
    parser.add_argument("--max-candidates", type=int, default=3, help="Maximum candidate dwellers to spawn.")
    parser.add_argument(
        "--namespace",
        default="mutation",
        help="Namespace prefix used in generated branch names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview candidate selection without creating branches/worktrees.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if bool(args.dry_run):
        preview = {
            "ok": True,
            "dry_run": True,
            "base_branch": str(args.base_branch),
            "namespace": str(args.namespace or "mutation"),
            "candidates": _candidate_dwellers(max(0, int(args.max_candidates))),
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        raise SystemExit(0)
    result = orchestrate(
        base_branch=str(args.base_branch),
        max_candidates=max(0, int(args.max_candidates)),
        namespace=str(args.namespace or "mutation").strip().replace(" ", "-"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
