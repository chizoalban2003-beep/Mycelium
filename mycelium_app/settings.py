from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Myco"
    system_motto: str = "Grow with Data."
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 60 * 24 * 7
    database_url: str = "sqlite:///storage/mycelium.db"
    db_auto_create_tables: bool = True
    db_migration_mode: str = "create_all"  # create_all|migrate
    cookie_name: str = "myco_access_token"
    cookie_secure: bool = False

    # CORS (for SaaS: when hosting API and frontend on different origins)
    # Comma-separated origins, e.g. "https://app.example.com,https://admin.example.com"
    cors_allow_origins_csv: str = ""
    cors_allow_credentials: bool = True

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

    # Mycelium Nexus (assistant) settings.
    nexus_device_id: str = "local"

    # Telemetry assistant (nudges from digital signals).
    nexus_telemetry_assistant_enabled: bool = False
    nexus_telemetry_assistant_tick_seconds: int = 60
    nexus_telemetry_assistant_window_hours: int = 6
    nexus_telemetry_assistant_confidence_threshold: float = 0.85
    nexus_telemetry_assistant_throttle_minutes: int = 120

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
    nexus_mission_log_retention_days: int = 14

    # Identity backup
    nexus_homeostasis_identity_backup_hours: int = 24

    # Nexus parental defaults (can be overridden per-user via /api/nexus/policy).
    nexus_intro_mode: str = "observe"  # ask | observe
    nexus_observe_hours: int = 24

    # Mycelium HiveSync (federated learning) MVP toggles.
    hive_enabled: bool = False
    hive_export_enabled_default: bool = False

    # Global Wisdom broadcast guardrails (ProjectMembrane companion).
    # A recommendation is published only when enough evidence exists.
    hive_wisdom_min_whispers: int = 2
    hive_wisdom_min_devices: int = 3

    # Public-ingest guardrail: throttle whisper imports to reduce abuse risk.
    hive_whisper_import_rate_limit_enabled: bool = True
    hive_whisper_import_rate_limit_window_seconds: int = 60
    hive_whisper_import_rate_limit_max_per_source: int = 60
    hive_whisper_import_rate_limit_max_per_device: int = 20

    # Optional external messaging bridge (Telegram first).
    app_public_base_url: str = ""
    notifications_bridge_enabled: bool = False
    notifications_dispatch_tick_seconds: int = 30
    notifications_dispatch_lookback_hours: int = 24
    notifications_dispatch_max_per_tick: int = 50
    notifications_telegram_bot_token: str = ""
    notifications_telegram_webhook_secret: str = ""

    # Optional SMTP mail delivery for account recovery.
    mail_enabled: bool = False
    mail_from_address: str = "noreply@myco.local"
    mail_smtp_host: str = ""
    mail_smtp_port: int = 587
    mail_smtp_username: str = ""
    mail_smtp_password: str = ""
    mail_smtp_use_tls: bool = True
    mail_smtp_use_ssl: bool = False
    mail_smtp_timeout_seconds: int = 10

    # Hybrid predictor (physics governor + pattern timing score).
    hybrid_predictor_enabled: bool = True
    hybrid_predictor_window_minutes: int = 120
    hybrid_predictor_min_signal_events: int = 10
    hybrid_predictor_governor_min_confidence: float = 0.90

    # Android TWA verification (served at /.well-known/assetlinks.json).
    android_app_package_name: str = "com.myco.companion"
    android_app_sha256_cert_fingerprints_csv: str = ""
    hive_wisdom_consensus_fraction: float = 0.50

    # Optional shared secret to allow headless child devices to ingest into the Parent Hub.
    # When set, child nodes can call Hive import endpoints with header: X-Hive-Token: <token>
    hive_ingest_token: str = ""

    # Hive Health dashboard access control.
    # If set (comma-separated emails), only these accounts can access /api/hive/health and /hive/health.
    hive_health_allowlist_emails_csv: str = ""

    # Validation Shadow (Empirical Nudges)
    # Disabled by default: when enabled and configured with a local CSV, the
    # system will benchmark old vs new Hive wisdom and only claim improvements
    # that are measured.
    nexus_validation_shadow_enabled: bool = False
    nexus_validation_shadow_dataset_path: str = ""
    nexus_validation_shadow_target_col: str = ""
    nexus_validation_shadow_max_rows: int = 5000
    nexus_validation_shadow_train_fraction: float = 0.8
    nexus_validation_shadow_random_seed: int = 42
    nexus_validation_shadow_n_cycles: int = 12
    nexus_validation_shadow_min_improvement_frac: float = 0.02

    # Active Curiosity (Human Ground Truth loop)
    # When enabled, the system will capture a few high-error ("agitated") samples
    # from prediction runs and ask the user for a short explanation or correction.
    nexus_active_curiosity_enabled: bool = False
    nexus_active_curiosity_max_cases_per_run: int = 3
    nexus_active_curiosity_min_abs_error: float = 0.0  # numeric targets only
    nexus_active_curiosity_min_error_quantile: float = 0.97  # 0..1; applied within test set
    nexus_active_curiosity_safe_columns_csv: str = ""  # comma-separated allowlist of columns to show

    # When true, creating new cases will also create a throttled nudge.
    nexus_active_curiosity_nudge_enabled: bool = True
    nexus_active_curiosity_nudge_throttle_minutes: int = 120

    # Ecosystem (living agent) settings.
    ecosystem_collector_enabled: bool = True
    ecosystem_collector_tick_seconds: int = 15
    ecosystem_learning_enabled: bool = True
    ecosystem_learning_tick_seconds: int = 120
    ecosystem_learning_window_hours: int = 6
    ecosystem_learning_bucket_minutes: int = 30
    ecosystem_experiment_enabled: bool = True
    ecosystem_experiment_tick_minutes: int = 30
    ecosystem_experiment_mutation_rate: float = 0.06
    ecosystem_experiment_selection_pressure: float = 1.35
    ecosystem_experiment_thermal_noise: float = 0.08
    ecosystem_autonomy_enabled: bool = True
    ecosystem_autonomy_tick_minutes: int = 10
    ecosystem_autonomy_mutation_rate: float = 0.08
    ecosystem_autonomy_thermal_noise: float = 0.08
    ecosystem_autonomy_selection_threshold: float = 0.02
    ecosystem_autonomy_action_cooldown_minutes: int = 45
    ecosystem_autonomy_min_confidence: float = 0.35
    ecosystem_autonomy_high_risk_threshold: float = 0.72
    ecosystem_autonomy_nudge_feedback_learning_rate: float = 0.08
    ecosystem_autonomy_goal_weekly_refresh_hours: int = 24
    ecosystem_autonomy_memory_consolidation_tick_minutes: int = 180

    @field_validator("app_name", mode="before")
    @classmethod
    def normalize_legacy_app_name(cls, value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "Myco"
        if raw.lower() in ("proofgrid", "mycelium"):
            return "Myco"
        return raw


settings = Settings()
