from wahabot.ai.pinned_state import render_pinned_block
from wahabot.core.tickets.models import TicketState, TicketUpdate
from wahabot.core.tickets.pinned import ticket_marker
from wahabot.core.tickets.store import TicketStore


def test_marker_is_one_line_per_state(store: TicketStore) -> None:
    contact = store.upsert_contact("491555000001@c.us", display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, contact.chat_id)

    assert ticket_marker(ticket) == f"Ticket {ticket.number}: open"
    assert ticket_marker(None) == ""


def test_pinned_block_carries_the_marker_but_not_the_record(store: TicketStore) -> None:
    contact = store.upsert_contact("491555000001@c.us", display_name="Smoke Sender")
    ticket = store.create_ticket(contact.id, contact.chat_id, summary="private summary")
    store.update_ticket(ticket.id, TicketUpdate(state=TicketState.NEEDS_YOU.value))

    block = render_pinned_block(
        policies=[],
        chat_rules=[],
        rules="",
        recap="",
        ticket=ticket_marker(store.ticket(ticket.id)),
        token_budget=1000,
    )

    assert "## Ticket" in block
    assert f"Ticket {ticket.number}: needs you" in block
    assert "private summary" not in block
    assert block.count("Ticket ") == 1
