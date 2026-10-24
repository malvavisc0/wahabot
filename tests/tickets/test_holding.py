import asyncio
from collections.abc import Callable

from tests.harness import RecordingWaha, chat_event
from wahabot.core.models import WahaEvent
from wahabot.core.tickets.store import TicketStore
from wahabot.core.tickets.translate import TicketTranslator
from wahabot.handlers.ticket_mixin import _HOLDING_REPLY, TicketMixin

CUSTOMER = "491555000001@c.us"


def holding_handler(
    store: TicketStore, completer: Callable[[str], str] | None
) -> tuple[TicketMixin, RecordingWaha]:
    handler = TicketMixin()
    handler.ticket_store = store
    handler.ticket_translator = TicketTranslator(completer)
    handler.waha = RecordingWaha()
    return handler, handler.waha


def hold_event(mid: str) -> WahaEvent:
    return WahaEvent.model_validate(chat_event(body="any news?", mid=mid, chat=CUSTOMER))


def test_holding_reply_is_translated_and_cached(store: TicketStore) -> None:
    calls: list[str] = []

    def spanish(prompt: str) -> str:
        calls.append(prompt)
        return "Sigue con el equipo."

    handler, waha = holding_handler(store, spanish)

    asyncio.run(handler.ticket_send_holding([hold_event("H1")], CUSTOMER, "es"))
    asyncio.run(handler.ticket_send_holding([hold_event("H2")], CUSTOMER, "es"))

    assert [entry[2] for entry in waha.sent] == ["Sigue con el equipo."] * 2
    assert len(calls) == 1


def test_holding_reply_falls_back_to_english(store: TicketStore) -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("llm down")

    handler, waha = holding_handler(store, broken)

    asyncio.run(handler.ticket_send_holding([hold_event("H1")], CUSTOMER, "es"))
    asyncio.run(handler.ticket_send_holding([hold_event("H2")], CUSTOMER, ""))
    asyncio.run(handler.ticket_send_holding([hold_event("H3")], CUSTOMER, "en"))

    assert [entry[2] for entry in waha.sent] == [_HOLDING_REPLY] * 3
