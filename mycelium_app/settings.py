from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Proofgrid"
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


settings = Settings()
