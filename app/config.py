from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic Interview Scheduler"
    environment: str = "development"
    debug: bool = False
    secret_key: str = Field(default="change-me", min_length=8)
    access_token_expire_minutes: int = 1440

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/interview_saas"
    sync_database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/interview_saas"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    sendgrid_api_key: str | None = None
    email_from: str = "hr@example.com"

    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from_number: str | None = None
    twilio_webhook_base_url: str = "http://localhost:8000"

    google_service_account_file: str | None = None
    default_timezone: str = "UTC"

    max_upload_size_mb: int = 5
    rate_limit_per_minute: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()

