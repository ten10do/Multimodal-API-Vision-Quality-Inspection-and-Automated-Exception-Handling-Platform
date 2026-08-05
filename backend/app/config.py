from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="IVQC_", extra="ignore")

    app_name: str = "industrial-vision-qc-backend"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5432/industrialvision_dev"
    inference_service_url: str = "http://127.0.0.1:8100"
    inference_timeout_seconds: float = 30.0
    max_upload_bytes: int = 10 * 1024 * 1024

    storage_dir: str = "storage-images"
    default_rule_version: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
