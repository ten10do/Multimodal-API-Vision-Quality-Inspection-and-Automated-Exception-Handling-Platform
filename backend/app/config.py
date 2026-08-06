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

    # ---- Phase 7 industrial integration ----
    # Disabled by default: the development environment runs in
    # NOT_INTEGRATED (the desired command is computed and recorded, the PLC
    # adapter is never invoked, and no field state is claimed). Industrial
    # integration is OPT-IN: set IVQC_PLC_ENABLED=true (and run the PLC
    # simulator/gateway) and IVQC_MES_ENABLED=true to enable the closed loop.
    plc_url: str = "http://127.0.0.1:8501"
    plc_adapter_type: str = "http"  # "http" | "opcua"
    plc_opcua_endpoint: str = "opc.tcp://127.0.0.1:8503"
    plc_timeout_seconds: float = 2.0
    plc_max_retries: int = 2
    plc_enabled: bool = False
    mes_url: str = "http://127.0.0.1:8502"
    mes_timeout_seconds: float = 2.0
    mes_max_retries: int = 2
    mes_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
