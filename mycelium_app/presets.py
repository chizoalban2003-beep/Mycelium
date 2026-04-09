from __future__ import annotations

from typing import Any


PRODUCTION_REGRESSION_PRESET_NAME = "v4.7_soft_multibuffer_20260405"
PRODUCTION_REGRESSION_PRESET_DISPLAY_NAME = "Myco v4.7 Liquid-Crystal (Soft Multi-Buffer)"

# Locked production hyperparameters for numeric regression.
# These were tuned on an internal tabular regression benchmark, but are intended as a strong
# general-purpose default for numeric targets.
PRODUCTION_REGRESSION_KWARGS: dict[str, Any] = {
    # Deep Freeze backbone (gas plane, 100 cycles)
    "plane": "gas",
    "n_cycles": 100,
    "shear_alpha": 1.60,
    "cycle_learning_rate": 0.25,
    "cycle_learning_rate_schedule": "exp_decay",
    "cycle_learning_rate_exp_decay": 0.995,
    "cycle_learning_rate_min_multiplier": 0.02,
    # Deep Freeze stage2 + scavenger settings (from sweep harness)
    "cascade_enabled": True,
    "competitive_inhibition": True,
    "thermal_noise": False,
    "stage2_cycles": 2,
    "stage2_trigger_cycle": 50,
    "stage2_shatter_complexes": True,
    "inhibition_strength": 0.7,
    "scavenger_cycles": 1,
    # Buffer shift
    "target_induced_viscosity_enabled": True,
    "target_induced_viscosity_gain": 0.60,
    "target_induced_viscosity_min_multiplier": 0.70,
    "target_induced_viscosity_max_multiplier": 1.00,
    # Field-Effect coupling (v4.5)
    "field_effect_enabled": True,
    "field_effect_alpha": 0.25,
    "field_effect_start_cycle": 40,
    "field_effect_use_abs_corr": True,
    "field_effect_coupling": "linear",
    "field_effect_alpha_exp_decay": 1.01,
    # Multi-Buffer (v4.6) + soft transitions (v4.7 probe)
    "multibuffer_enabled": True,
    "multibuffer_q_low": 0.20,
    "multibuffer_q_high": 0.80,
    "multibuffer_low_viscosity_multiplier": 1.30,
    "multibuffer_mid_viscosity_multiplier": 1.00,
    "multibuffer_high_viscosity_multiplier": 0.80,
    "multibuffer_low_field_alpha_multiplier": 1.00,
    "multibuffer_mid_field_alpha_multiplier": 1.00,
    "multibuffer_high_field_alpha_multiplier": 1.00,
    "multibuffer_transition_frac": 0.06,
}

_PCR4C_KWARGS: dict[str, Any] = {
    "pcr_enabled": True,
    "pcr_cycles": 4,
    "pcr_pvalue_threshold": 0.05,
    "pcr_tau": 4.0,
    "pcr_gain": 0.55,
    "pcr_strength_cap": 2.5,
    "pcr_amp_cap": 3.5,
    "pcr_require_stable": True,
}


PRODUCTION_CLASSIFICATION_BALANCED_PRESET_NAME = "v5.0_pcr4c_balanced_20260405"
PRODUCTION_CLASSIFICATION_BALANCED_KWARGS: dict[str, Any] = {
    **_PCR4C_KWARGS,
    # Balanced: keep full coverage (no abstain), but surface uncertainty.
    "low_confidence_mode": "flag",
    "low_confidence_threshold": 0.0,
    "low_confidence_entropy_threshold": 0.0,
    "low_confidence_smear_metric": "entropy",
    "low_confidence_combine_rule": "or",
    "low_confidence_auto_conf_quantile": 0.20,
    "low_confidence_auto_smear_quantile": 0.80,
}

PRODUCTION_CLASSIFICATION_MAX_ACCURACY_PRESET_NAME = "v5.0_pcr4c_max_accuracy_20260405"
PRODUCTION_CLASSIFICATION_MAX_ACCURACY_KWARGS: dict[str, Any] = {
    **_PCR4C_KWARGS,
    # Max accuracy: abstain aggressively, then attempt to rescue via confirmatory/secondary passes.
    "low_confidence_mode": "abstain",
    "low_confidence_threshold": 0.0,
    "low_confidence_entropy_threshold": 0.0,
    "low_confidence_smear_metric": "entropy",
    "low_confidence_combine_rule": "or",
    "low_confidence_auto_conf_quantile": 0.60,
    "low_confidence_auto_smear_quantile": 0.65,
    "low_confidence_require_ionized": True,
    "low_confidence_ionization_pvalue": 0.05,
    "low_confidence_ionization_z_min": 0.35,
    "low_confidence_confirmatory_enabled": True,
    "low_confidence_confirmatory_conf_min": 0.55,
    "low_confidence_confirmatory_conf_max": 0.90,
    "low_confidence_confirmatory_consensus_threshold": 0.65,
    "low_confidence_secondary_enabled": True,
    "low_confidence_secondary_cycles": 2,
    "low_confidence_secondary_viscosity_multiplier": 0.80,
    "low_confidence_secondary_use_spearman": True,
}

PRODUCTION_CLASSIFICATION_MAX_COVERAGE_PRESET_NAME = "v5.0_pcr4c_max_coverage_20260405"
PRODUCTION_CLASSIFICATION_MAX_COVERAGE_KWARGS: dict[str, Any] = {
    **_PCR4C_KWARGS,
    # Max coverage: never abstain; minimize flagging.
    "low_confidence_mode": "flag",
    "low_confidence_threshold": 0.0,
    "low_confidence_entropy_threshold": 0.0,
    "low_confidence_smear_metric": "entropy",
    "low_confidence_combine_rule": "or",
    "low_confidence_auto_conf_quantile": 0.10,
    "low_confidence_auto_smear_quantile": 0.90,
    "low_confidence_require_ionized": False,
}
