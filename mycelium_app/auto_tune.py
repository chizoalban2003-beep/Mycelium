"""Auto-tuning engine for force field constants.

After each learning cycle, measures prediction accuracy and nudges the
four force constants (G, K_E, K_S, K_W) toward values that minimize MAE.

The constants evolve with the data — they're not fixed physics, they're
discovered physics. The ecosystem finds its own laws.

Algorithm:
    1. Run prediction with current constants → baseline MAE
    2. For each constant, try +delta and -delta perturbation
    3. Run prediction with perturbed constants → trial MAE
    4. If trial MAE < baseline MAE, adopt the perturbation
    5. Decay delta over time (simulated annealing)
    6. Persist tuned constants to DB for next cycle
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from mycelium_app.force_field import (
    _G, _K_E, _K_S, _K_W,
    compute_force_field,
    ForceFieldState,
)
from mycelium_app.unified_field import field_to_predictor_kwargs


@dataclass
class TunedConstants:
    """The four force constants, tunable."""
    G: float = _G
    K_E: float = _K_E
    K_S: float = _K_S
    K_W: float = _K_W
    generation: int = 0
    last_mae: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def _apply_constants(G: float, K_E: float, K_S: float, K_W: float) -> None:
    """Temporarily override the module-level force constants."""
    import mycelium_app.force_field as ff
    ff._G = max(0.01, G)
    ff._K_E = max(0.01, K_E)
    ff._K_S = max(0.01, K_S)
    ff._K_W = max(0.001, K_W)


def _restore_constants(tc: TunedConstants) -> None:
    """Restore constants from a TunedConstants instance."""
    _apply_constants(tc.G, tc.K_E, tc.K_S, tc.K_W)


def _run_prediction_mae(
    df: pd.DataFrame,
    signals: list[dict[str, Any]],
    target_col: str,
    window_hours: float = 6.0,
) -> float | None:
    """Run force field + unified bridge + physics prediction, return MAE."""
    from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction

    try:
        ff = compute_force_field(signals, window_hours=window_hours, n_iterations=10)
        kwargs = field_to_predictor_kwargs(ff, base_kwargs={
            "target_col": target_col,
            "train_fraction": 0.7,
            "random_seed": 42,
            "top_k_weights": min(10, max(1, df.shape[1] - 1)),
        })
        # Clamp cycles to avoid timeout on small data
        kwargs["n_cycles"] = min(kwargs.get("n_cycles", 15), 20)
        pred = run_physics_prediction(df, **kwargs)
        if pred and pred.metrics and pred.metrics.mae is not None:
            return float(pred.metrics.mae)
    except Exception:
        pass

    # Fallback: direct prediction without force field
    try:
        pred = run_physics_prediction(
            df, target_col=target_col, plane=PhysicsPlane.liquid,
            train_fraction=0.7, random_seed=42, n_cycles=10,
            top_k_weights=min(10, max(1, df.shape[1] - 1)),
        )
        if pred and pred.metrics and pred.metrics.mae is not None:
            return float(pred.metrics.mae)
    except Exception:
        pass

    return None


def auto_tune_constants(
    df: pd.DataFrame,
    signals: list[dict[str, Any]],
    target_col: str,
    current: TunedConstants | None = None,
    *,
    delta: float = 0.05,
    min_delta: float = 0.005,
    window_hours: float = 6.0,
) -> TunedConstants:
    """Run one auto-tuning cycle.

    Perturbs each constant by ±delta, keeps the direction that reduces MAE.
    Delta decays with generation (simulated annealing).

    Parameters
    ----------
    df : ecosystem DataFrame
    signals : raw signal list for force field
    target_col : prediction target
    current : current constants (None = use defaults)
    delta : perturbation magnitude (fraction of current value)
    min_delta : minimum delta before stopping
    window_hours : for force field computation

    Returns
    -------
    TunedConstants with updated values and history.
    """
    if current is None:
        current = TunedConstants()

    # Apply current constants
    _restore_constants(current)

    # Anneal delta based on generation
    effective_delta = max(min_delta, delta * (0.9 ** current.generation))

    # Baseline MAE with current constants
    baseline_mae = _run_prediction_mae(df, signals, target_col, window_hours)
    if baseline_mae is None:
        current.history.append({
            "generation": current.generation,
            "action": "skip",
            "reason": "baseline prediction failed",
            "ts": datetime.utcnow().isoformat(),
        })
        return current

    current.last_mae = baseline_mae
    improved = False

    constants = [
        ("G", current.G),
        ("K_E", current.K_E),
        ("K_S", current.K_S),
        ("K_W", current.K_W),
    ]

    for name, value in constants:
        best_value = value
        best_mae = baseline_mae

        for direction in [1, -1]:
            trial_value = value * (1 + direction * effective_delta)
            trial_value = max(0.001, trial_value)

            # Apply trial
            trial_constants = TunedConstants(
                G=current.G, K_E=current.K_E,
                K_S=current.K_S, K_W=current.K_W,
            )
            setattr(trial_constants, name, trial_value)
            _apply_constants(trial_constants.G, trial_constants.K_E,
                           trial_constants.K_S, trial_constants.K_W)

            trial_mae = _run_prediction_mae(df, signals, target_col, window_hours)
            if trial_mae is not None and trial_mae < best_mae:
                best_mae = trial_mae
                best_value = trial_value

        if best_value != value:
            setattr(current, name, best_value)
            improved = True

    # Restore final constants
    _restore_constants(current)

    current.generation += 1
    current.history.append({
        "generation": current.generation,
        "action": "tuned" if improved else "stable",
        "baseline_mae": round(baseline_mae, 6),
        "final_mae": round(current.last_mae or baseline_mae, 6),
        "delta": round(effective_delta, 6),
        "constants": {
            "G": round(current.G, 6),
            "K_E": round(current.K_E, 6),
            "K_S": round(current.K_S, 6),
            "K_W": round(current.K_W, 6),
        },
        "ts": datetime.utcnow().isoformat(),
    })

    # Update last_mae to the best we found
    _restore_constants(current)
    final_mae = _run_prediction_mae(df, signals, target_col, window_hours)
    if final_mae is not None:
        current.last_mae = final_mae

    return current


def save_tuned_constants(tc: TunedConstants, *, user_id: int) -> None:
    """Persist tuned constants to the database."""
    from sqlmodel import Session
    from mycelium_app.db import engine
    from mycelium_app.models import ForceFieldSnapshot

    try:
        with Session(engine) as session:
            snapshot = ForceFieldSnapshot(
                user_id=user_id,
                n_particles=0,
                n_bonds=0,
                total_energy=0,
                mean_coherence=0,
                agent_stage="tuning",
                agent_coherence=0,
                agent_crystallized=False,
                agent_bound_particles=0,
                attention_entropy=0,
                dominant_force="auto_tune",
                field_json=json.dumps({
                    "type": "tuned_constants",
                    "G": tc.G, "K_E": tc.K_E, "K_S": tc.K_S, "K_W": tc.K_W,
                    "generation": tc.generation,
                    "last_mae": tc.last_mae,
                    "history": tc.history[-5:],
                }, separators=(",", ":")),
            )
            session.add(snapshot)
            session.commit()
    except Exception:
        pass


def load_tuned_constants(*, user_id: int) -> TunedConstants | None:
    """Load previously tuned constants from the database."""
    from sqlmodel import Session, select
    from mycelium_app.db import engine
    from mycelium_app.models import ForceFieldSnapshot

    try:
        with Session(engine) as session:
            row = session.exec(
                select(ForceFieldSnapshot)
                .where(
                    ForceFieldSnapshot.user_id == user_id,
                    ForceFieldSnapshot.dominant_force == "auto_tune",
                )
                .order_by(ForceFieldSnapshot.created_at.desc())
                .limit(1)
            ).first()
            if row and row.field_json:
                data = json.loads(row.field_json)
                if data.get("type") == "tuned_constants":
                    tc = TunedConstants(
                        G=float(data.get("G", _G)),
                        K_E=float(data.get("K_E", _K_E)),
                        K_S=float(data.get("K_S", _K_S)),
                        K_W=float(data.get("K_W", _K_W)),
                        generation=int(data.get("generation", 0)),
                        last_mae=data.get("last_mae"),
                    )
                    return tc
    except Exception:
        pass
    return None
