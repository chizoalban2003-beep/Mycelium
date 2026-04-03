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
    cascade_enabled: bool = Form(True),
    competitive_inhibition: bool = Form(True),
    thermal_noise: bool = Form(False),
    n_cycles: int = Form(30),
    stage2_cycles: int = Form(2),
    stage2_trigger_cycle: int = Form(50),
    stage2_shatter_complexes: bool = Form(True),
    low_confidence_mode: str = Form("none"),
    low_confidence_threshold: float = Form(0.0),
    low_confidence_entropy_threshold: float = Form(0.0),
    low_confidence_smear_metric: str = Form("entropy"),
    low_confidence_combine_rule: str = Form("or"),
    low_confidence_auto_conf_quantile: float = Form(0.20),
    low_confidence_auto_smear_quantile: float = Form(0.80),
    low_confidence_require_ionized: bool = Form(False),
    low_confidence_ionization_pvalue: float = Form(0.05),
    low_confidence_ionization_z_min: float = Form(0.25),
    low_confidence_confirmatory_enabled: bool = Form(False),
    low_confidence_confirmatory_conf_min: float = Form(0.50),
    low_confidence_confirmatory_conf_max: float = Form(0.90),
    low_confidence_confirmatory_consensus_threshold: float = Form(0.60),
    low_confidence_confirmatory_min_ion_hits: int = Form(0),
    low_confidence_secondary_enabled: bool = Form(False),
    low_confidence_secondary_cycles: int = Form(0),
    low_confidence_secondary_viscosity_multiplier: float = Form(0.75),
    low_confidence_secondary_viscosity_anneal: bool = Form(False),
    low_confidence_secondary_viscosity_multiplier_start: float | None = Form(None),
    low_confidence_secondary_inhibition_multiplier: float = Form(0.85),
    low_confidence_secondary_shear_multiplier: float = Form(1.10),
    low_confidence_secondary_relax_ionization_gate: bool = Form(True),
    low_confidence_secondary_ionization_z_min: float = Form(0.10),
    low_confidence_secondary_relaxed_ion_conf_min: float = Form(0.55),
    low_confidence_secondary_use_spearman: bool = Form(True),
    low_confidence_secondary_spearman_min_abs: float = Form(0.015),
    low_confidence_secondary_spearman_margin: float = Form(0.010),
    low_confidence_secondary_promote_min_zone_votes: int = Form(3),
    low_confidence_secondary_promote_z_min: float = Form(0.50),
    low_confidence_secondary_promote_conf_min: float = Form(0.42),
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
            n_cycles=int(n_cycles),
            stage2_cycles=int(stage2_cycles),
            stage2_trigger_cycle=int(stage2_trigger_cycle),
            stage2_shatter_complexes=bool(stage2_shatter_complexes),
            cascade_enabled=bool(cascade_enabled),
            competitive_inhibition=bool(competitive_inhibition),
            thermal_noise=bool(thermal_noise),
            low_confidence_mode=str(low_confidence_mode),
            low_confidence_threshold=float(low_confidence_threshold),
            low_confidence_entropy_threshold=float(low_confidence_entropy_threshold),
            low_confidence_smear_metric=str(low_confidence_smear_metric),
            low_confidence_combine_rule=str(low_confidence_combine_rule),
            low_confidence_auto_conf_quantile=float(low_confidence_auto_conf_quantile),
            low_confidence_auto_smear_quantile=float(low_confidence_auto_smear_quantile),
            low_confidence_require_ionized=bool(low_confidence_require_ionized),
            low_confidence_ionization_pvalue=float(low_confidence_ionization_pvalue),
            low_confidence_ionization_z_min=float(low_confidence_ionization_z_min),
            low_confidence_confirmatory_enabled=bool(low_confidence_confirmatory_enabled),
            low_confidence_confirmatory_conf_min=float(low_confidence_confirmatory_conf_min),
            low_confidence_confirmatory_conf_max=float(low_confidence_confirmatory_conf_max),
            low_confidence_confirmatory_consensus_threshold=float(low_confidence_confirmatory_consensus_threshold),
            low_confidence_confirmatory_min_ion_hits=int(low_confidence_confirmatory_min_ion_hits),
            low_confidence_secondary_enabled=bool(low_confidence_secondary_enabled),
            low_confidence_secondary_cycles=int(low_confidence_secondary_cycles),
            low_confidence_secondary_viscosity_multiplier=float(low_confidence_secondary_viscosity_multiplier),
            low_confidence_secondary_viscosity_anneal=bool(low_confidence_secondary_viscosity_anneal),
            low_confidence_secondary_viscosity_multiplier_start=low_confidence_secondary_viscosity_multiplier_start,
            low_confidence_secondary_inhibition_multiplier=float(low_confidence_secondary_inhibition_multiplier),
            low_confidence_secondary_shear_multiplier=float(low_confidence_secondary_shear_multiplier),
            low_confidence_secondary_relax_ionization_gate=bool(low_confidence_secondary_relax_ionization_gate),
            low_confidence_secondary_ionization_z_min=float(low_confidence_secondary_ionization_z_min),
            low_confidence_secondary_relaxed_ion_conf_min=float(low_confidence_secondary_relaxed_ion_conf_min),
            low_confidence_secondary_use_spearman=bool(low_confidence_secondary_use_spearman),
            low_confidence_secondary_spearman_min_abs=float(low_confidence_secondary_spearman_min_abs),
            low_confidence_secondary_spearman_margin=float(low_confidence_secondary_spearman_margin),
            low_confidence_secondary_promote_min_zone_votes=int(low_confidence_secondary_promote_min_zone_votes),
            low_confidence_secondary_promote_z_min=float(low_confidence_secondary_promote_z_min),
            low_confidence_secondary_promote_conf_min=float(low_confidence_secondary_promote_conf_min),
        )

        return {
            "ok": True,
            "target": pred.target,
            "target_kind": pred.target_kind,
            "plane": pred.plane.value,
            "diagnostics": getattr(pred, "diagnostics", None),
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
                    "ionization": m.ionization,
                    "normality_p": None if m.normality_p is None else round(float(m.normality_p), 8),
                    "p_value": None if m.p_value is None else round(float(m.p_value), 8),
                    "mass": round(float(m.mass), 6),
                    "stable": bool(m.stable),
                    "complex_id": m.complex_id,
                    "complex_size": m.complex_size,
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
                    "bond_type": getattr(b, "bond_type", "affinity"),
                }
                for b in pred.bonding_map
            ],
            "iteration_gains": [
                {
                    "cycle": int(it.cycle),
                    "test_accuracy": None
                    if it.test_accuracy is None
                    else round(float(it.test_accuracy), 6),
                    "test_mae": None if it.test_mae is None else round(float(it.test_mae), 6),
                    "test_rmse": None if it.test_rmse is None else round(float(it.test_rmse), 6),
                    "lift_over_baseline": None
                    if it.lift_over_baseline is None
                    else round(float(it.lift_over_baseline), 6),
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
                "buffer_ionization": pred.metrics.buffer_ionization,
                "buffer_normality_p": None
                if pred.metrics.buffer_normality_p is None
                else round(float(pred.metrics.buffer_normality_p), 8),
                "mae": None if pred.metrics.mae is None else round(float(pred.metrics.mae), 6),
                "rmse": None if pred.metrics.rmse is None else round(float(pred.metrics.rmse), 6),
                "baseline_mae": None
                if pred.metrics.baseline_mae is None
                else round(float(pred.metrics.baseline_mae), 6),
                "baseline_rmse": None
                if pred.metrics.baseline_rmse is None
                else round(float(pred.metrics.baseline_rmse), 6),
                "accuracy": None if pred.metrics.accuracy is None else round(float(pred.metrics.accuracy), 6),
                "baseline_accuracy": None
                if pred.metrics.baseline_accuracy is None
                else round(float(pred.metrics.baseline_accuracy), 6),
                "best_cycle": pred.metrics.best_cycle,
                "best_lift": None if pred.metrics.best_lift is None else round(float(pred.metrics.best_lift), 6),
                "gel_band_sharpness": None
                if pred.metrics.gel_band_sharpness is None
                else round(float(pred.metrics.gel_band_sharpness), 6),
                "gel_smearing": None
                if pred.metrics.gel_smearing is None
                else round(float(pred.metrics.gel_smearing), 6),
                "gel_ghost_band_rate": None
                if pred.metrics.gel_ghost_band_rate is None
                else round(float(pred.metrics.gel_ghost_band_rate), 6),
                "gel_confidence_mean": None
                if pred.metrics.gel_confidence_mean is None
                else round(float(pred.metrics.gel_confidence_mean), 6),
                "gel_confidence_std": None
                if pred.metrics.gel_confidence_std is None
                else round(float(pred.metrics.gel_confidence_std), 6),
                "abstain_rate": None
                if getattr(pred.metrics, "abstain_rate", None) is None
                else round(float(pred.metrics.abstain_rate), 6),
                "coverage": None
                if getattr(pred.metrics, "coverage", None) is None
                else round(float(pred.metrics.coverage), 6),
                "selective_accuracy": None
                if getattr(pred.metrics, "selective_accuracy", None) is None
                else round(float(pred.metrics.selective_accuracy), 6),
            },
            "preview": pred.preview_rows,
        }

    except PredictorError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Failed to run predictor: {type(e).__name__}: {e}"}
