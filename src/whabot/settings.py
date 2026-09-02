"""Typed settings loaded from environment variables and .env files."""

import logging
import sys
from functools import lru_cache
from pathlib import Path

from loguru import logger
from pydantic import Field
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
    memory_token_limit: int = 8000
    timezone: str = "UTC"

    #: Langfuse credentials (no WHABOT_ prefix — the SDK's conventional
    #: names); empty disables LLM trace export entirely.
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(default="", alias="LANGFUSE_BASE_URL")

    #: Only download and attach image messages when the model supports vision.
    vision: bool = True
    #: Images larger than this many bytes are skipped (vision APIs reject
    #: oversized payloads and RAM spikes ~4/3x via base64).
    max_image_bytes: int = 10 * 1024 * 1024
    #: How many image URLs sniffed from a message's text to download.
    max_url_images: int = 2

    web_search_max_results: int = 5
    web_search_timeout: float = 30.0
    web_search_proxy: str | None = None

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
    """Build the cached Settings instance (pydantic reads .env and os.environ)."""
    return Settings()


class InterceptHandler(logging.Handler):
    """Bridge stdlib logging records into loguru, one format for all logs."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(settings: Settings) -> None:
    """Configure loguru sinks and route stdlib logging (uvicorn) through it."""
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level.upper())
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx"):
        std_logger = logging.getLogger(name)
        std_logger.handlers = []
        std_logger.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
