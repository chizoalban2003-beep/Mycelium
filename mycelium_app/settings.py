from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Proofgrid"
    system_motto: str = "Grow with Data."
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 60 * 24 * 7
    database_url: str = "sqlite:///storage/mycelium.db"
    cookie_name: str = "mycelium_access_token"
    cookie_secure: bool = False

    # When true, numeric targets (regression/datetime) will use the locked production
    # preset hyperparameters unless explicitly overridden in code.
    predictor_lock_production_regression_preset: bool = True

    # When true, categorical targets (classification) will use the locked production
    # preset hyperparameters unless explicitly overridden in code.
    predictor_lock_production_classification_preset: bool = True

    # Optional persistent-memory layer ("Physics Ledger"). Disabled by default.
    predictor_physics_ledger_enabled: bool = False
    predictor_physics_ledger_recall_enabled: bool = False
    predictor_physics_ledger_store_enabled: bool = False
    predictor_physics_ledger_allow_override_locked_presets: bool = False
    predictor_physics_ledger_max_candidates: int = 250
    predictor_physics_ledger_min_jaccard: float = 0.70
    predictor_physics_ledger_min_r2_to_store: float = 0.05
    predictor_physics_ledger_min_accuracy_to_store: float = 0.55
    predictor_physics_ledger_min_gel_confidence_mean_to_store: float = 0.95

    # Proofgrid Nexus (assistant) settings.
    nexus_device_id: str = "local"

    # Nexus Homeostasis ("Body")
    # When enabled, a background loop periodically computes reflection snapshots
    # and can prune low-value memory under resource pressure.
    nexus_homeostasis_enabled: bool = False
    nexus_homeostasis_tick_seconds: int = 30
    nexus_homeostasis_window_days: int = 30
    nexus_homeostasis_agitated_cycles_trigger: int = 10
    nexus_homeostasis_deep_breath_cooldown_minutes: int = 30

    # Resource guardrails
    nexus_homeostasis_min_free_mb: int = 512
    nexus_homeostasis_prune_signal_retention_days: int = 21
    nexus_homeostasis_prune_growth_retention_days: int = 90
    nexus_homeostasis_prune_experience_retention_days: int = 90
    nexus_homeostasis_prune_experience_confidence_lt: float = 0.55

    # Identity backup
    nexus_homeostasis_identity_backup_hours: int = 24

    # Nexus parental defaults (can be overridden per-user via /api/nexus/policy).
    nexus_intro_mode: str = "observe"  # ask | observe
    nexus_observe_hours: int = 24

    # Proofgrid HiveSync (federated learning) MVP toggles.
    hive_enabled: bool = False
    hive_export_enabled_default: bool = False


settings = Settings()
