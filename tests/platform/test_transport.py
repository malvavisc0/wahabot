from pathlib import Path

from tests.conftest import Bot
from tests.harness import RecordingWaha
from wahabot.core.transport.protocol import Transport
from wahabot.core.transport.waha import WahaTransport


def test_recording_client_satisfies_the_transport_protocol() -> None:
    assert isinstance(RecordingWaha(), Transport)


def test_waha_client_satisfies_the_transport_protocol(tmp_path: Path) -> None:
    from wahabot.core.db import Database
    from wahabot.core.delivery.intents import IntentStore

    db = Database(tmp_path / "wahabot.db")
    db.migrate()
    transport = WahaTransport(RecordingWaha(), IntentStore(db.path))

    assert isinstance(transport, Transport)


def test_delivery_and_whatsapp_tools_do_not_import_the_concrete_client() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "wahabot"
    modules = [
        root / "core" / "delivery" / "send.py",
        root / "core" / "delivery" / "intents.py",
        *sorted((root / "ai" / "tools" / "whatsapp").glob("*.py")),
    ]

    offenders = [
        path.name
        for path in modules
        if "from wahabot.core.waha import WahaClient" in path.read_text()
    ]

    assert offenders == []


def test_registration_selects_only_the_waha_transport(bot: Bot) -> None:
    assert type(bot.transport) is WahaTransport
    assert bot.transport.client is bot.waha
    assert isinstance(bot.transport, Transport)
