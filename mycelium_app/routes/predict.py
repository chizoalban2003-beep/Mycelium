from __future__ import annotations

import io

from fastapi import APIRouter, Depends, File, Form, UploadFile

import pandas as pd

from mycelium_app.deps import get_current_user
from mycelium_app.models import User
from mycelium_app.physics_predictor import PhysicsPlane, PredictorError, run_physics_prediction


router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/electrophoresis")
async def electrophoresis_predict(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    plane: str = Form(PhysicsPlane.solid.value),
    top_k: int = Form(30),
    train_ratio: float = Form(0.8),
    random_seed: int = Form(42),
    no_split: bool = Form(False),
    max_rows: int = Form(5000),
    current_user: User = Depends(get_current_user),
):
    """Run the Mycelium electrophoresis engine and return JSON outputs for realtime polling clients."""

    _ = current_user
    try:
        plane_enum = PhysicsPlane(plane)
    except Exception:
        plane_enum = PhysicsPlane.solid

    try:
        raw = await file.read()
        if not raw:
            raise PredictorError("Empty upload")

        df = pd.read_csv(io.BytesIO(raw), nrows=max(1, min(int(max_rows), 200_000)))
        top_k = max(1, min(int(top_k), 200))

        if no_split:
            train_ratio = 1.0
        else:
            try:
                train_ratio = float(train_ratio)
            except Exception:
                train_ratio = 0.8
            train_ratio = max(0.05, min(0.95, train_ratio))

        pred = run_physics_prediction(
            df,
            target_col=target_col,
            plane=plane_enum,
            train_fraction=float(train_ratio),
            random_seed=int(random_seed),
            top_k_weights=top_k,
        )

        return {
            "ok": True,
            "target": pred.target,
            "target_kind": pred.target_kind,
            "plane": pred.plane.value,
            "weights": [
                {
                    "feature": w.feature,
                    "weight": round(float(w.weight), 6),
                    "method": w.method,
                    "kind": w.feature_kind,
                    "signed": w.signed,
                }
                for w in pred.weights
            ],
            "migration_map": [
                {
                    "feature": m.feature,
                    "kind": m.feature_kind,
                    "method": m.method,
                    "charge": round(float(m.charge), 6),
                    "entropy": round(float(m.entropy), 6),
                    "variance": round(float(m.variance), 6),
                    "standard_error": round(float(m.standard_error), 6),
                    "kl_divergence": round(float(m.kl_divergence), 6),
                    "density": round(float(m.density), 6),
                    "viscosity": round(float(m.viscosity), 6),
                    "terminal_velocity": round(float(m.terminal_velocity), 6),
                    "arrival_speed": round(float(m.arrival_speed), 6),
                    "direction": m.direction,
                    "state": m.state,
                }
                for m in pred.migration_map
            ],
            "bonding_map": [
                {
                    "feature_a": b.feature_a,
                    "feature_b": b.feature_b,
                    "affinity": round(float(b.affinity), 6),
                    "bonding_factor": round(float(b.bonding_factor), 6),
                }
                for b in pred.bonding_map
            ],
            "iteration_gains": [
                {
                    "cycle": int(it.cycle),
                    "test_accuracy": round(float(it.test_accuracy), 6),
                    "lift_over_baseline": round(float(it.lift_over_baseline), 6),
                }
                for it in pred.iteration_gains
            ],
            "equilibrium_zones": [
                {
                    "zone_id": int(ez.zone_id),
                    "features": ez.features,
                    "avg_pI": round(float(ez.avg_pI), 6),
                    "avg_momentum": round(float(ez.avg_momentum), 6),
                    "strength": round(float(ez.strength), 6),
                }
                for ez in pred.equilibrium_zones
            ],
            "metrics": {
                "target_kind": pred.metrics.target_kind,
                "n_rows": pred.metrics.n_rows,
                "n_train": pred.metrics.n_train,
                "n_test": pred.metrics.n_test,
                "train_fraction": round(float(pred.metrics.train_fraction), 4),
                "random_seed": int(pred.metrics.random_seed),
                "n_features_used": pred.metrics.n_features_used,
                "mae": None if pred.metrics.mae is None else round(float(pred.metrics.mae), 6),
                "rmse": None if pred.metrics.rmse is None else round(float(pred.metrics.rmse), 6),
                "accuracy": None if pred.metrics.accuracy is None else round(float(pred.metrics.accuracy), 6),
                "baseline_accuracy": None
                if pred.metrics.baseline_accuracy is None
                else round(float(pred.metrics.baseline_accuracy), 6),
                "best_cycle": pred.metrics.best_cycle,
                "best_lift": None if pred.metrics.best_lift is None else round(float(pred.metrics.best_lift), 6),
            },
            "preview": pred.preview_rows,
        }

    except PredictorError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Failed to run predictor: {type(e).__name__}: {e}"}
