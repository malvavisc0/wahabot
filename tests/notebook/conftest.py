from pathlib import Path
from typing import cast

import pytest
from llama_index.core.tools import FunctionTool

from tests.notebook.support import (
    SESSION,
)
from wahabot.ai.tools.notebook import (
    notebook_builder,
)
from wahabot.settings import Settings


@pytest.fixture()
def notebook_settings(tmp_path: Path) -> Settings:
    return Settings(
        webhook_hmac_key="fictional-key",
        waha_url="http://waha.invalid",
        waha_api_key="fictional-key",
        llm_api_base="http://llm.invalid",
        llm_api_key="fictional-key",
        langfuse_public_key="",
        langfuse_secret_key="",
        session=SESSION,
        data_dir=tmp_path,
        memory_token_limit=4096,
        _env_file=None,
    )


@pytest.fixture()
def notebook_tool(notebook_settings: Settings) -> FunctionTool:
    return cast(FunctionTool, notebook_builder(notebook_settings))
