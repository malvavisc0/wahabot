"""Typed settings loaded from environment variables and .env files."""

import sys
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """whabot configuration; real env vars win over .env values."""

    model_config = SettingsConfigDict(
        env_prefix="WHABOT_", env_file=".env", extra="ignore"
    )

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    webhook_hmac_key: str
    waha_url: str
    waha_api_key: str

    llm_api_base: str
    llm_api_key: str
    llm_model: str = "gpt-4o-mini"
    timezone: str = "UTC"

    session: str = "default"

    data_dir: Path = Path("data")

    @property
    def access_config(self) -> Path:
        """Path of this session's access config."""
        return self.data_dir / "sessions" / f"{self.session}.json"

    @property
    def journal_dir(self) -> Path:
        """Directory of the raw event journal."""
        return self.data_dir / "events"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load .env then build the cached Settings instance."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    return Settings()  # pydantic-settings reads os.environ


def setup_logging(settings: Settings) -> None:
    """Configure loguru sinks from the log level."""
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level.upper())
