from tests.harness import ME_JID, OWN_LID, waha_event
from wahabot.core.models import WahaEvent
from wahabot.core.tickets.models import Contact, Ticket, TicketState, TicketUpdate
from wahabot.core.tickets.router import (
    RouteKind,
    ask_which_ticket,
    route_message,
    whats_open_text,
)
from wahabot.core.tickets.store import TicketStore

CUSTOMER = "491555000001@c.us"
OTHER = "491555000002@c.us"


def operator_event(body: str, *, mid: str = "OP", quote: str = "") -> WahaEvent:
    raw = waha_event(mid)
    payload = raw["payload"]
    payload["id"] = f"true_{ME_JID}_{mid}"
    payload["from"] = ME_JID
    payload["to"] = ME_JID
    payload["fromMe"] = True
    payload["participant"] = OWN_LID
    payload["source"] = "app"
    payload["body"] = body
    if quote:
        payload["replyTo"] = {"id": quote}
    return WahaEvent.model_validate(raw)


def make_ticket(
    store: TicketStore, chat: str, name: str, mid: str
) -> tuple[Contact, Ticket]:
    contact = store.upsert_contact(chat, display_name=name)
    ticket = store.create_ticket(contact.id, chat, summary="issue")
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value))
    return contact, store.ticket(ticket.id)


def test_quote_routes_every_alert_and_forwarded_id(store: TicketStore) -> None:
    _, ticket = make_ticket(store, CUSTOMER, "Smoke Sender", "Q")
    store.append_alert_ids(ticket.id, ["true_alert", "true_forwarded"])

    for quoted in ("true_alert", "true_forwarded"):
        route = route_message(store, operator_event("refunded 10 EUR", quote=quoted))
        assert route.kind == RouteKind.QUOTE_REPLY
        assert route.ticket.number == ticket.number


def test_commands_win_over_quoting_an_alert(store: TicketStore) -> None:
    _contact, ticket = make_ticket(store, CUSTOMER, "Smoke Sender", "QC")
    store.append_alert_ids(ticket.id, ["true_alert"])

    route = route_message(
        store, operator_event(f"close {ticket.number}", quote="true_alert")
    )

    assert route.kind == RouteKind.CLOSE
    assert route.ticket is not None and route.ticket.id == ticket.id


def test_number_and_unique_name_both_route(store: TicketStore) -> None:
    _, ticket = make_ticket(store, CUSTOMER, "Smoke Sender", "N")

    numbered = route_message(store, operator_event(f"{ticket.number}: refunded"))
    named = route_message(store, operator_event("Smoke Sender: refunded"))

    assert numbered.kind == RouteKind.TICKET_REPLY
    assert numbered.ticket.number == ticket.number
    assert named.kind == RouteKind.TICKET_REPLY
    assert named.text == "refunded"


def test_numbered_forms_are_all_accepted(store: TicketStore) -> None:
    _, ticket = make_ticket(store, CUSTOMER, "Smoke Sender", "NF")

    for body in (
        f"{ticket.number}: refunded",
        f"ticket {ticket.number} refunded",
        f"#{ticket.number} refunded",
    ):
        route = route_message(store, operator_event(body))
        assert route.kind == RouteKind.TICKET_REPLY
        assert route.ticket is not None and route.ticket.number == ticket.number


def test_timestamps_and_note_numbers_do_not_route(store: TicketStore) -> None:
    make_ticket(store, CUSTOMER, "Smoke Sender", "FP")

    for body in (
        "2:30 lunch with the team",
        "2026 is going to be a good year",
        "123 Main Street",
        "10.5 kg please",
    ):
        assert route_message(store, operator_event(body)).kind == RouteKind.NONE


def test_unknown_number_asks_instead_of_ignoring(store: TicketStore) -> None:
    make_ticket(store, CUSTOMER, "Smoke Sender", "U")

    route = route_message(store, operator_event("999: refunded"))

    assert route.kind == RouteKind.ASK
    assert route.text == "No ticket #999."


def test_ambiguous_name_asks_with_candidates(store: TicketStore) -> None:
    make_ticket(store, CUSTOMER, "Sam", "A1")
    make_ticket(store, OTHER, "Sam", "A2")

    route = route_message(store, operator_event("Sam: refunded"))

    assert route.kind == RouteKind.ASK
    assert sorted(ticket.number for ticket in route.candidates) == [1, 2]
    assert "Which ticket?" in ask_which_ticket(route.candidates)


def test_commands_route_without_a_bot_name(store: TicketStore) -> None:
    make_ticket(store, CUSTOMER, "Smoke Sender", "C")

    kinds = {
        route_message(store, operator_event("what's open?")).kind,
        route_message(store, operator_event("close 1")).kind,
        route_message(store, operator_event("reopen 1")).kind,
        route_message(store, operator_event("pause")).kind,
        route_message(store, operator_event("resume")).kind,
    }

    assert kinds == {
        RouteKind.WHATS_OPEN,
        RouteKind.CLOSE,
        RouteKind.REOPEN,
        RouteKind.PAUSE,
        RouteKind.RESUME,
    }


def test_pause_and_resume_name_the_contact(store: TicketStore) -> None:
    make_ticket(store, CUSTOMER, "Smoke Sender", "P")

    pause = route_message(store, operator_event("pause Smoke Sender"))
    resume = route_message(store, operator_event("resume Smoke Sender"))

    assert pause.contact is not None and pause.contact.chat_id == CUSTOMER
    assert resume.contact is not None and resume.contact.chat_id == CUSTOMER


def test_unknown_note_is_none(store: TicketStore) -> None:
    assert (
        route_message(store, operator_event("not a ticket note")).kind == RouteKind.NONE
    )


def test_whats_open_lists_newest_first_with_age_and_urgent(
    store: TicketStore,
) -> None:
    _, older = make_ticket(store, CUSTOMER, "Older", "O1")
    _, newer = make_ticket(store, OTHER, "Newer", "O2")
    store.update_ticket(newer.id, TicketUpdate(urgent=True), at=store.now())

    text = whats_open_text(store, now=store.now() + 2 * 3600)
    lines = text.splitlines()[1:]

    assert lines[0].startswith(f"- {newer.number}: Newer")
    assert "URGENT" in lines[0]
    assert lines[1].startswith(f"- {older.number}: Older")


def test_unknown_pause_name_does_not_pause_the_whole_session(
    store: TicketStore,
) -> None:
    make_ticket(store, CUSTOMER, "Smoke Sender", "P1")

    route = route_message(store, operator_event("pause Nobody Here"))

    assert route.kind == RouteKind.ASK
    assert "No contact named" in route.text
    assert store.session_paused_until("default") is None


def test_unique_name_without_an_active_ticket_asks(store: TicketStore) -> None:
    store.upsert_contact(CUSTOMER, display_name="Smoke Sender")

    route = route_message(store, operator_event("Smoke Sender: refunded"))

    assert route.kind == RouteKind.ASK
    assert "No active ticket" in route.text


def test_bare_save_routes_to_the_save_command(store: TicketStore) -> None:
    route = route_message(store, operator_event("save"))
    assert route.kind == RouteKind.SAVE
    assert route.ticket is None


def test_save_with_a_ticket_number_targets_that_ticket(store: TicketStore) -> None:
    _contact, ticket = make_ticket(store, CUSTOMER, "Smoke Sender", "m1")

    route = route_message(store, operator_event(f"save {ticket.number}"))

    assert route.kind == RouteKind.SAVE
    assert route.ticket is not None and route.ticket.id == ticket.id


def test_save_with_an_unknown_ticket_asks(store: TicketStore) -> None:
    route = route_message(store, operator_event("save 999"))

    assert route.kind == RouteKind.ASK
    assert "999" in route.text
