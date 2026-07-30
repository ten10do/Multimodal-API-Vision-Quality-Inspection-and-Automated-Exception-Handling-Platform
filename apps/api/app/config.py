from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    app_name: str = "vision-qc-agent"
    log_level: str = "INFO"
    api_cors_origins: list[str] = ["http://localhost:3000"]
    upload_dir: Path = Path("./uploads")
    max_upload_bytes: int = 10 * 1024 * 1024

    database_url: str = "sqlite+aiosqlite:///./vision_qc.db"
    redis_url: str = "redis://localhost:6379/0"
    celery_task_always_eager: bool = True

    ai_mode: str = Field(
        default="mock",
        validation_alias=AliasChoices("AI_MODE", "AI_PROVIDER_MODE"),
    )
    mock_vision_model: str = "mock-vision-v1"
    mock_reasoning_model: str = "mock-reasoning-v1"

    bailian_api_key: str = ""
    bailian_base_url: str = ""
    bailian_model: str = ""
    bailian_timeout_seconds: float = Field(default=30, gt=0)
    bailian_max_retries: int = Field(default=2, ge=0, le=10)

    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    deepseek_model: str = ""
    deepseek_timeout_seconds: float = Field(default=30, gt=0)
    deepseek_max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("ai_mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = value.lower()
        if normalized == "live":
            normalized = "real"
        if normalized not in {"mock", "real"}:
            raise ValueError("AI_MODE must be 'mock' or 'real'")
        return normalized

    def missing_bailian_config(self) -> list[str]:
        required = {
            "BAILIAN_API_KEY": self.bailian_api_key,
            "BAILIAN_BASE_URL": self.bailian_base_url,
            "BAILIAN_MODEL": self.bailian_model,
        }
        return [name for name, value in required.items() if not value]

    def missing_deepseek_config(self) -> list[str]:
        required = {
            "DEEPSEEK_API_KEY": self.deepseek_api_key,
            "DEEPSEEK_BASE_URL": self.deepseek_base_url,
            "DEEPSEEK_MODEL": self.deepseek_model,
        }
        return [name for name, value in required.items() if not value]

    def validate_bailian_config(self) -> None:
        missing = self.missing_bailian_config()
        if missing:
            raise ValueError(f"Bailian configuration is incomplete: {', '.join(missing)}")

    def validate_deepseek_config(self) -> None:
        missing = self.missing_deepseek_config()
        if missing:
            raise ValueError(f"DeepSeek configuration is incomplete: {', '.join(missing)}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
