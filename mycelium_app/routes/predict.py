from __future__ import annotations

import io
import math
import time

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlmodel import Session

import pandas as pd

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import User
from mycelium_app.knowledge_sync import MemoryManager
from mycelium_app.predictor_homeostasis import apply_homeostasis_from_db
from mycelium_app.physics_predictor import PhysicsPlane, PredictorError, infer_target_kind, run_physics_prediction
from mycelium_app.presets import (
    PRODUCTION_CLASSIFICATION_BALANCED_KWARGS,
    PRODUCTION_CLASSIFICATION_BALANCED_PRESET_NAME,
    PRODUCTION_CLASSIFICATION_MAX_ACCURACY_KWARGS,
    PRODUCTION_CLASSIFICATION_MAX_ACCURACY_PRESET_NAME,
    PRODUCTION_CLASSIFICATION_MAX_COVERAGE_KWARGS,
    PRODUCTION_CLASSIFICATION_MAX_COVERAGE_PRESET_NAME,
    PRODUCTION_REGRESSION_KWARGS,
    PRODUCTION_REGRESSION_PRESET_DISPLAY_NAME,
    PRODUCTION_REGRESSION_PRESET_NAME,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/predict", tags=["predict"])


def _r2_from_actual_pred(actual: list[object] | None, predicted: list[object] | None) -> float | None:
    if not actual or not predicted:
        return None
    pairs: list[tuple[float, float]] = []
    for a, b in zip(actual, predicted, strict=False):
        if a is None or b is None:
            continue
        try:
            af = float(a)
            bf = float(b)
        except Exception:
            continue
        if math.isfinite(af) and math.isfinite(bf):
            pairs.append((af, bf))

    if len(pairs) < 2:
        return None

    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    y_bar = sum(y_true) / float(len(y_true))
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=False))
    ss_tot = sum((a - y_bar) ** 2 for a in y_true)
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


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
    low_confidence_secondary_sieve_enabled: bool = Form(False),
    low_confidence_secondary_sieve_cycles: int = Form(2),
    low_confidence_secondary_sieve_reverse_multiplier: float = Form(0.75),
    low_confidence_secondary_sieve_noise_std: float = Form(0.04),
    low_confidence_secondary_sieve_instability_min: float = Form(0.65),
    low_confidence_secondary_sieve_conf_delta_max: float = Form(0.002),
    low_confidence_secondary_sieve_update_norm_max: float = Form(0.003),
    classification_goal: str = Form("balanced"),
    cleaning_enabled: bool = Form(True),
    cleaning_outlier_strategy: str = Form("winsorize"),
    cleaning_outlier_fold: float = Form(1.5),
    cleaning_outlier_q_low: float = Form(0.005),
    cleaning_outlier_q_high: float = Form(0.995),
    cleaning_arbitrary_min: float | None = Form(None),
    cleaning_arbitrary_max: float | None = Form(None),
    max_rows: int = Form(5000),
    use_ledger: bool = Form(False),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Run the Proofgrid electrophoresis engine and return JSON outputs for realtime polling clients."""

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

        base_kwargs: dict[str, object] = {
            "target_col": target_col,
            "plane": plane_enum,
            "train_fraction": float(train_ratio),
            "random_seed": int(random_seed),
            "top_k_weights": top_k,
            "n_cycles": int(n_cycles),
            "stage2_cycles": int(stage2_cycles),
            "stage2_trigger_cycle": int(stage2_trigger_cycle),
            "stage2_shatter_complexes": bool(stage2_shatter_complexes),
            "cascade_enabled": bool(cascade_enabled),
            "competitive_inhibition": bool(competitive_inhibition),
            "thermal_noise": bool(thermal_noise),
            "low_confidence_mode": str(low_confidence_mode),
            "low_confidence_threshold": float(low_confidence_threshold),
            "low_confidence_entropy_threshold": float(low_confidence_entropy_threshold),
            "low_confidence_smear_metric": str(low_confidence_smear_metric),
            "low_confidence_combine_rule": str(low_confidence_combine_rule),
            "low_confidence_auto_conf_quantile": float(low_confidence_auto_conf_quantile),
            "low_confidence_auto_smear_quantile": float(low_confidence_auto_smear_quantile),
            "low_confidence_require_ionized": bool(low_confidence_require_ionized),
            "low_confidence_ionization_pvalue": float(low_confidence_ionization_pvalue),
            "low_confidence_ionization_z_min": float(low_confidence_ionization_z_min),
            "low_confidence_confirmatory_enabled": bool(low_confidence_confirmatory_enabled),
            "low_confidence_confirmatory_conf_min": float(low_confidence_confirmatory_conf_min),
            "low_confidence_confirmatory_conf_max": float(low_confidence_confirmatory_conf_max),
            "low_confidence_confirmatory_consensus_threshold": float(low_confidence_confirmatory_consensus_threshold),
            "low_confidence_confirmatory_min_ion_hits": int(low_confidence_confirmatory_min_ion_hits),
            "low_confidence_secondary_enabled": bool(low_confidence_secondary_enabled),
            "low_confidence_secondary_cycles": int(low_confidence_secondary_cycles),
            "low_confidence_secondary_viscosity_multiplier": float(low_confidence_secondary_viscosity_multiplier),
            "low_confidence_secondary_viscosity_anneal": bool(low_confidence_secondary_viscosity_anneal),
            "low_confidence_secondary_viscosity_multiplier_start": low_confidence_secondary_viscosity_multiplier_start,
            "low_confidence_secondary_inhibition_multiplier": float(low_confidence_secondary_inhibition_multiplier),
            "low_confidence_secondary_shear_multiplier": float(low_confidence_secondary_shear_multiplier),
            "low_confidence_secondary_relax_ionization_gate": bool(low_confidence_secondary_relax_ionization_gate),
            "low_confidence_secondary_ionization_z_min": float(low_confidence_secondary_ionization_z_min),
            "low_confidence_secondary_relaxed_ion_conf_min": float(low_confidence_secondary_relaxed_ion_conf_min),
            "low_confidence_secondary_use_spearman": bool(low_confidence_secondary_use_spearman),
            "low_confidence_secondary_spearman_min_abs": float(low_confidence_secondary_spearman_min_abs),
            "low_confidence_secondary_spearman_margin": float(low_confidence_secondary_spearman_margin),
            "low_confidence_secondary_promote_min_zone_votes": int(low_confidence_secondary_promote_min_zone_votes),
            "low_confidence_secondary_promote_z_min": float(low_confidence_secondary_promote_z_min),
            "low_confidence_secondary_promote_conf_min": float(low_confidence_secondary_promote_conf_min),
            "low_confidence_secondary_sieve_enabled": bool(low_confidence_secondary_sieve_enabled),
            "low_confidence_secondary_sieve_cycles": int(low_confidence_secondary_sieve_cycles),
            "low_confidence_secondary_sieve_reverse_multiplier": float(low_confidence_secondary_sieve_reverse_multiplier),
            "low_confidence_secondary_sieve_noise_std": float(low_confidence_secondary_sieve_noise_std),
            "low_confidence_secondary_sieve_instability_min": float(low_confidence_secondary_sieve_instability_min),
            "low_confidence_secondary_sieve_conf_delta_max": float(low_confidence_secondary_sieve_conf_delta_max),
            "low_confidence_secondary_sieve_update_norm_max": float(low_confidence_secondary_sieve_update_norm_max),
            "cleaning_enabled": bool(cleaning_enabled),
            "cleaning_outlier_strategy": str(cleaning_outlier_strategy),
            "cleaning_outlier_fold": float(cleaning_outlier_fold),
            "cleaning_outlier_q_low": float(cleaning_outlier_q_low),
            "cleaning_outlier_q_high": float(cleaning_outlier_q_high),
            "cleaning_arbitrary_min": cleaning_arbitrary_min,
            "cleaning_arbitrary_max": cleaning_arbitrary_max,
        }

        preset_applied: str | None = None
        ledger_info: dict[str, object] | None = None
        try:
            tk = infer_target_kind(df[target_col])
        except Exception:
            tk = "numeric"

        if tk == "categorical" and settings.predictor_lock_production_classification_preset:
            goal = str(classification_goal or "balanced").strip().lower()
            if goal in ("max_accuracy", "accuracy", "precise"):
                base_kwargs.update(PRODUCTION_CLASSIFICATION_MAX_ACCURACY_KWARGS)
                preset_applied = PRODUCTION_CLASSIFICATION_MAX_ACCURACY_PRESET_NAME
            elif goal in ("max_coverage", "coverage", "broad"):
                base_kwargs.update(PRODUCTION_CLASSIFICATION_MAX_COVERAGE_KWARGS)
                preset_applied = PRODUCTION_CLASSIFICATION_MAX_COVERAGE_PRESET_NAME
            else:
                base_kwargs.update(PRODUCTION_CLASSIFICATION_BALANCED_KWARGS)
                preset_applied = PRODUCTION_CLASSIFICATION_BALANCED_PRESET_NAME
        elif settings.predictor_lock_production_regression_preset and tk in ("numeric", "datetime"):
            # Override base kwargs with locked production regression settings.
            merged = dict(base_kwargs)
            merged.update(PRODUCTION_REGRESSION_KWARGS)
            # Convert plane string to enum for predictor.
            merged["plane"] = PhysicsPlane(str(PRODUCTION_REGRESSION_KWARGS.get("plane", "gas")))
            base_kwargs = merged
            preset_applied = PRODUCTION_REGRESSION_PRESET_NAME

        mm = MemoryManager(
            enabled=bool(settings.predictor_physics_ledger_enabled),
            recall_enabled=bool(settings.predictor_physics_ledger_recall_enabled),
            store_enabled=bool(settings.predictor_physics_ledger_store_enabled),
            allow_override_locked_presets=bool(settings.predictor_physics_ledger_allow_override_locked_presets),
            max_candidates=int(settings.predictor_physics_ledger_max_candidates),
            min_jaccard=float(settings.predictor_physics_ledger_min_jaccard),
            min_r2_to_store=float(settings.predictor_physics_ledger_min_r2_to_store),
            min_accuracy_to_store=float(settings.predictor_physics_ledger_min_accuracy_to_store),
            min_gel_confidence_mean_to_store=float(settings.predictor_physics_ledger_min_gel_confidence_mean_to_store),
        )

        if bool(use_ledger):
            recalled, decision, _entry = mm.recall(
                session,
                user_id=int(current_user.id or 0),
                df=df,
                target_col=str(target_col),
                target_kind=str(tk),
                locked_preset_applied=(preset_applied is not None),
            )
            if recalled:
                merged = dict(base_kwargs)
                merged.update(recalled)
                try:
                    if not isinstance(merged.get("plane"), PhysicsPlane):
                        merged["plane"] = PhysicsPlane(str(merged.get("plane")))
                except Exception:
                    pass
                base_kwargs = merged
            if mm.enabled:
                ledger_info = {
                    "enabled": True,
                    "recalled": bool(decision.recalled),
                    "entry_id": decision.recalled_entry_id,
                    "jaccard": decision.jaccard,
                    "score_metric": decision.score_metric,
                    "score_value": decision.score_value,
                }

        # Homeostasis bridge: single source of truth lives in predictor_homeostasis.py
        base_kwargs, homeostasis_info = apply_homeostasis_from_db(
            session,
            user_id=int(current_user.id or 0),
            base_kwargs=base_kwargs,
        )

        t0 = time.perf_counter()
        pred = run_physics_prediction(df, **base_kwargs)
        elapsed_s = float(time.perf_counter() - t0)

        r2 = None
        if pred.target_kind == "numeric":
            r2 = _r2_from_actual_pred(getattr(pred, "test_actual", None), getattr(pred, "test_predicted", None))

        stored_entry_id, stored_metric, stored_value = mm.maybe_store(
            session,
            user_id=int(current_user.id or 0),
            project_id=None,
            df=df,
            target_col=str(target_col),
            target_kind=str(tk),
            preset_name=preset_applied,
            preset_display=None
            if preset_applied is None
            else (
                PRODUCTION_REGRESSION_PRESET_DISPLAY_NAME
                if preset_applied == PRODUCTION_REGRESSION_PRESET_NAME
                else preset_applied
            ),
            applied_kwargs=dict(base_kwargs),
            r2=r2,
            accuracy=(None if pred.metrics.accuracy is None else float(pred.metrics.accuracy)),
            gel_confidence_mean=(
                None
                if pred.metrics.gel_confidence_mean is None
                else float(pred.metrics.gel_confidence_mean)
            ),
        )

        if ledger_info is None and mm.enabled:
            ledger_info = {"enabled": True, "recalled": False}
        if ledger_info is not None and stored_entry_id is not None:
            ledger_info["stored_entry_id"] = int(stored_entry_id)
            ledger_info["stored_score_metric"] = stored_metric
            ledger_info["stored_score_value"] = stored_value

        return {
            "ok": True,
            "production_preset": preset_applied,
            "production_preset_display": None
            if preset_applied is None
            else (
                PRODUCTION_REGRESSION_PRESET_DISPLAY_NAME
                if preset_applied == PRODUCTION_REGRESSION_PRESET_NAME
                else preset_applied
            ),
            "target": pred.target,
            "target_kind": pred.target_kind,
            "plane": pred.plane.value,
            "diagnostics": getattr(pred, "diagnostics", None),
            "homeostasis": homeostasis_info,
            "ledger": ledger_info,
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
                "r2": None if r2 is None else round(float(r2), 6),
                "baseline_mae": None
                if pred.metrics.baseline_mae is None
                else round(float(pred.metrics.baseline_mae), 6),
                "baseline_rmse": None
                if pred.metrics.baseline_rmse is None
                else round(float(pred.metrics.baseline_rmse), 6),
                "elapsed_s": round(float(elapsed_s), 6),
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
