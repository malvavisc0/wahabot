from pathlib import Path

import pytest

from wahabot.settings import Settings


@pytest.fixture()
def unit_settings(tmp_path: Path) -> Settings:
    return Settings(
        webhook_hmac_key="k",
        waha_url="http://waha.invalid",
        waha_api_key="k",
        llm_api_base="http://llm.invalid",
        llm_api_key="k",
        data_dir=tmp_path,
        triage_gate=False,
        langfuse_public_key="",
        langfuse_secret_key="",
        _env_file=None,
    )
