"""Concrete handler object combining the handler mixins."""

from wahabot.ai.albums import Albums
from wahabot.ai.bursts import Bursts
from wahabot.ai.intake import IntakeLedger
from wahabot.ai.tools.whatsapp import EscalationChannel
from wahabot.ai.triage import TriageGate
from wahabot.core.access import SessionConfig, SessionConfigReloader
from wahabot.core.delivery.intents import IntentStore
from wahabot.core.scheduler import TicketScheduler
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tickets.translate import TicketTranslator
from wahabot.core.transport import Transport
from wahabot.handlers.album_mixin import AlbumMixin
from wahabot.handlers.burst_mixin import BurstMixin
from wahabot.handlers.config_mixin import ConfigMixin
from wahabot.handlers.ingest_mixin import IngestMixin
from wahabot.handlers.single_mixin import SingleMixin
from wahabot.handlers.ticket_mixin import TicketMixin
from wahabot.settings import Settings


class Handlers(
    AlbumMixin, BurstMixin, IngestMixin, SingleMixin, TicketMixin, ConfigMixin
):
    """One registration's handler state and methods."""

    def __init__(
        self,
        settings: Settings,
        waha: Transport,
        config: SessionConfig,
        config_reloader: SessionConfigReloader,
        escalation_channel: EscalationChannel,
        triage: TriageGate | None,
        intake: IntakeLedger,
        started_at: float,
        intents: IntentStore,
        ticket_store: TicketStore | None = None,
        ticket_translator: TicketTranslator | None = None,
        scheduler: TicketScheduler | None = None,
    ) -> None:
        self.settings = settings
        self.waha = waha
        self.config = config
        self.config_reloader = config_reloader
        self.escalation_channel = escalation_channel
        self.triage = triage
        self.intake = intake
        self.started_at = started_at
        self.intents = intents
        self.ticket_store = ticket_store
        self.ticket_translator = ticket_translator
        self.scheduler = scheduler
        self.transport = waha
        self.bursts = Bursts(
            settings.burst_inactivity_s,
            settings.burst_hold_cap_s,
            self.run_burst,
        )
        self.albums = Albums(self.run_album)
