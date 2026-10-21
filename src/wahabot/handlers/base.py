"""State and method contract shared by the handler mixins."""

from typing import Any

from wahabot.ai.tools.whatsapp import EscalationChannel
from wahabot.ai.triage import TriageGate
from wahabot.ai.workflow import FunctionCallingAgentWorkflow
from wahabot.core.access import SessionConfig, SessionConfigReloader
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


class HandlerBase:
    """Per-registration state shared by every handler mixin."""

    settings: Settings  # pyright: ignore[reportUninitializedInstanceVariable]
    waha: WahaClient  # pyright: ignore[reportUninitializedInstanceVariable]
    config: SessionConfig  # pyright: ignore[reportUninitializedInstanceVariable]
    config_reloader: SessionConfigReloader  # pyright: ignore[reportUninitializedInstanceVariable]
    escalation_channel: EscalationChannel  # pyright: ignore[reportUninitializedInstanceVariable]
    triage: TriageGate | None  # pyright: ignore[reportUninitializedInstanceVariable]
    started_at: float  # pyright: ignore[reportUninitializedInstanceVariable]
    agent: FunctionCallingAgentWorkflow  # pyright: ignore[reportUninitializedInstanceVariable]

    async def transcribe_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def download_captioned_image(self, *args: Any, **kwargs: Any) -> Any: ...
    async def prepare_message_videos(self, *args: Any, **kwargs: Any) -> Any: ...
    async def accept_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def handle_self_command(self, *args: Any, **kwargs: Any) -> Any: ...
    async def prepare_chat_message(self, *args: Any, **kwargs: Any) -> Any: ...
    async def skip_or_buffer_message(self, *args: Any, **kwargs: Any) -> Any: ...
    def group_message_addressed(self, *args: Any, **kwargs: Any) -> Any: ...
