from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlmodel import Session

from mycelium_app.models import MetricSnapshot
from mycelium_app.physics_predictor import PhysicsPlane, PredictorError, run_physics_prediction
from mycelium_app.settings import settings


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_digest(path: str) -> str:
    """Best-effort dataset digest without hashing the full file."""

    p = Path(path)
    try:
        st = p.stat()
        head = b""
        try:
            with p.open("rb") as f:
                head = f.read(64 * 1024)
        except Exception:
            head = b""
        h = hashlib.sha256()
        h.update(str(p.resolve()).encode("utf-8"))
        h.update(str(int(st.st_size)).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        h.update(head)
        return h.hexdigest()
    except Exception:
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShadowResult:
    ok: bool
    metric_name: str | None = None
    baseline_value: float | None = None
    trial_value: float | None = None
    improvement_frac: float | None = None
    target_kind: str | None = None
    notes: str | None = None


def _metric_from_result(res: Any) -> tuple[str | None, float | None, str | None]:
    try:
        m = res.metrics
    except Exception:
        return None, None, None

    tk = getattr(m, "target_kind", None)
    if tk == "categorical":
        v = getattr(m, "accuracy", None)
        return "accuracy", (None if v is None else float(v)), str(tk)

    # numeric/datetime => regression-style metrics
    v = getattr(m, "mae", None)
    if v is not None:
        return "mae", float(v), str(tk)
    v2 = getattr(m, "rmse", None)
    if v2 is not None:
        return "rmse", float(v2), str(tk)
    return None, None, str(tk) if tk is not None else None


def run_validation_shadow(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    target_col: str,
    baseline_kwargs: dict[str, Any],
    trial_kwargs: dict[str, Any],
    wisdom_digest: str,
) -> ShadowResult:
    """Run a mini-benchmark on a configured local dataset and store MetricSnapshots.

    This only runs if `settings.nexus_validation_shadow_enabled` and required
    settings are present.
    """

    if not bool(getattr(settings, "nexus_validation_shadow_enabled", False)):
        return ShadowResult(ok=False, notes="disabled")

    dataset_path = str(getattr(settings, "nexus_validation_shadow_dataset_path", "") or "").strip()
    if not dataset_path:
        return ShadowResult(ok=False, notes="dataset_path_not_set")

    target_col = (target_col or str(getattr(settings, "nexus_validation_shadow_target_col", "") or "")).strip()
    if not target_col:
        return ShadowResult(ok=False, notes="target_col_not_set")

    max_rows = max(200, min(int(getattr(settings, "nexus_validation_shadow_max_rows", 5000)), 200_000))
    train_fraction = float(getattr(settings, "nexus_validation_shadow_train_fraction", 0.8))
    seed = int(getattr(settings, "nexus_validation_shadow_random_seed", 42))
    n_cycles = max(3, min(int(getattr(settings, "nexus_validation_shadow_n_cycles", 12)), 60))

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return ShadowResult(ok=False, notes=f"read_csv_failed:{type(e).__name__}")

    if int(df.shape[0]) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    d_dig = dataset_digest(dataset_path)

    # Force a few knobs so both runs are comparable.
    common = {
        "train_fraction": train_fraction,
        "random_seed": seed,
        "n_cycles": n_cycles,
        "plane": PhysicsPlane.solid,
    }

    def _run(kwargs: dict[str, Any]) -> Any:
        k = dict(common)
        k.update(kwargs or {})
        # Ensure plane is correct enum
        try:
            if isinstance(k.get("plane"), str):
                k["plane"] = PhysicsPlane(str(k["plane"]))
        except Exception:
            k["plane"] = PhysicsPlane.solid
        return run_physics_prediction(df, target_col=target_col, **k)

    try:
        base_res = _run(baseline_kwargs)
        trial_res = _run(trial_kwargs)
    except PredictorError as e:
        return ShadowResult(ok=False, notes=f"predictor_error:{e}")
    except Exception as e:
        return ShadowResult(ok=False, notes=f"run_failed:{type(e).__name__}")

    metric_name, base_value, tk = _metric_from_result(base_res)
    metric_name2, trial_value, tk2 = _metric_from_result(trial_res)

    if metric_name is None or metric_name2 is None or metric_name != metric_name2:
        return ShadowResult(ok=False, notes="metric_unavailable")

    # Improvement fraction: positive means better.
    improvement: float | None
    if metric_name == "accuracy":
        if base_value is None or base_value == 0.0 or trial_value is None:
            improvement = None
        else:
            improvement = (trial_value - base_value) / abs(base_value)
    else:
        # error metrics: lower is better
        if base_value is None or base_value == 0.0 or trial_value is None:
            improvement = None
        else:
            improvement = (base_value - trial_value) / abs(base_value)

    # Persist snapshots
    def _store(phase: str, value: float, kwargs_obj: dict[str, Any]) -> None:
        row = MetricSnapshot(
            created_by_user_id=int(user_id),
            project_id=int(project_id) if project_id is not None else None,
            dataset_digest=str(d_dig),
            wisdom_digest=str(wisdom_digest),
            phase=str(phase),
            target_col=str(target_col),
            target_kind=str(tk2 or tk or ""),
            metric_name=str(metric_name),
            metric_value=float(value),
            kwargs_json=_dumps(kwargs_obj),
            notes=f"n_cycles={n_cycles};max_rows={max_rows}",
        )
        session.add(row)

    if base_value is not None and trial_value is not None:
        _store("baseline", float(base_value), dict(baseline_kwargs or {}))
        _store("trial", float(trial_value), dict(trial_kwargs or {}))
        session.commit()

    return ShadowResult(
        ok=True,
        metric_name=metric_name,
        baseline_value=base_value,
        trial_value=trial_value,
        improvement_frac=improvement,
        target_kind=str(tk2 or tk or ""),
        notes=None,
    )
