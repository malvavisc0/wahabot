"""Concrete handler object combining the handler mixins."""

from wahabot.ai.intake import IntakeLedger
from wahabot.ai.tools.whatsapp import EscalationChannel
from wahabot.ai.triage import TriageGate
from wahabot.core.access import SessionConfig, SessionConfigReloader
from wahabot.core.waha import WahaClient
from wahabot.handlers.album_mixin import AlbumMixin
from wahabot.handlers.burst_mixin import BurstMixin
from wahabot.handlers.config_mixin import ConfigMixin
from wahabot.handlers.ingest_mixin import IngestMixin
from wahabot.handlers.single_mixin import SingleMixin
from wahabot.settings import Settings


class Handlers(AlbumMixin, BurstMixin, IngestMixin, SingleMixin, ConfigMixin):
    """One registration's handler state and methods."""

    def __init__(
        self,
        settings: Settings,
        waha: WahaClient,
        config: SessionConfig,
        config_reloader: SessionConfigReloader,
        escalation_channel: EscalationChannel,
        triage: TriageGate | None,
        intake: IntakeLedger,
        started_at: float,
    ) -> None:
        self.settings = settings
        self.waha = waha
        self.config = config
        self.config_reloader = config_reloader
        self.escalation_channel = escalation_channel
        self.triage = triage
        self.intake = intake
        self.started_at = started_at
