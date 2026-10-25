"""History list and ticket view models for the web back office."""

from dataclasses import dataclass, field
from urllib.parse import quote, urlencode, urlsplit

from wahabot.core.skills.store import SkillStore
from wahabot.core.tickets.models import (
    Attachment,
    Contact,
    DoneBy,
    EventKind,
    Ticket,
    TicketEvent,
    TicketState,
)
from wahabot.core.tickets.store import TicketStore
from wahabot.web.services import format_time
from wahabot.web.services.translation import (
    TRANSLATION_LANGUAGE,
    Translator,
    translate_cached,
)

SHOW_CHOICES = (
    ("", "All tickets"),
    ("open", "Open"),
    ("needs_you", "Needs you"),
    ("waiting", "Waiting for the customer"),
    ("done_assistant", "Done by the assistant"),
    ("done_you", "Done by you"),
    ("closed", "Closed"),
)
INTERNAL_KINDS = {
    EventKind.ALERT.value,
    EventKind.NUDGE.value,
    EventKind.HELD.value,
    EventKind.HOLDING.value,
    EventKind.OPERATOR_REPLY.value,
    EventKind.SKILL_CALL.value,
}
STATE_LABELS = {
    TicketState.OPEN.value: "Open",
    TicketState.WAITING_CUSTOMER.value: "Waiting for the customer",
    TicketState.NEEDS_YOU.value: "Needs you",
    TicketState.DONE.value: "Done",
    TicketState.CLOSED.value: "Closed",
}


@dataclass(frozen=True, slots=True)
class HistoryRow:
    """One ticket row in the list."""

    number: int
    subject: str
    summary: str
    status_label: str
    status_class: str
    person: str
    number_text: str
    reason: str
    solved: str
    time: str
    updated_at: int
    urgent: bool


@dataclass(frozen=True, slots=True)
class HistorySection:
    """One grouped list section."""

    key: str
    title: str
    rows: list[HistoryRow] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class HistoryList:
    """The rendered list's filters, counts and sections."""

    rows: list[HistoryRow]
    sections: list[HistorySection]
    shown: int
    q: str
    show: str
    choices: tuple[tuple[str, str], ...]
    has_filters: bool
    back_url: str


@dataclass(frozen=True, slots=True)
class AttachmentView:
    """One attachment inside its sender's message."""

    id: int
    kind: str
    mime: str
    derived_text: str


@dataclass(frozen=True, slots=True)
class ThreadEntry:
    """One message or internal event in the ticket thread."""

    sender: str
    kind_label: str
    time: str
    body: str
    style: str
    internal: bool
    outgoing: bool
    attachments: list[AttachmentView] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CallView:
    """One skill call shown on a ticket or skill page."""

    skill: str
    method: str
    path: str
    outcome: str
    failed: bool
    started_by: str
    time: str
    duration_ms: int
    error: str
    ticket_number: int | None


@dataclass(frozen=True, slots=True)
class TicketPage:
    """The ticket detail view."""

    ticket: Ticket
    subject: str
    summary: str
    status_label: str
    status_class: str
    next_step: str
    contact_name: str
    contact_number: str
    language: str
    entries: list[ThreadEntry]
    calls: list[CallView]
    toggle_url: str
    toggle_label: str
    back: str


def match_show(ticket: Ticket, show: str) -> bool:
    """Whether one ticket belongs to the selected Show filter."""
    if not show:
        return True
    if show == "open":
        return ticket.state == TicketState.OPEN.value
    if show == "needs_you":
        return ticket.state == TicketState.NEEDS_YOU.value
    if show == "waiting":
        return ticket.state == TicketState.WAITING_CUSTOMER.value
    if show == "closed":
        return ticket.state == TicketState.CLOSED.value
    if show == "done_assistant":
        return ticket.state == TicketState.DONE.value and ticket.done_by == (
            DoneBy.ASSISTANT.value
        )
    if show == "done_you":
        return ticket.state == TicketState.DONE.value and ticket.done_by == (
            DoneBy.OPERATOR.value
        )
    return True


def search_matches(
    store: TicketStore, ticket: Ticket, contact: Contact | None, q: str
) -> bool:
    """Search name, number, ticket number, summary and recorded text."""
    needle = q.strip().casefold()
    if not needle:
        return True
    fields = [str(ticket.number), ticket.summary, ticket.chat_id]
    if contact is not None:
        fields.extend([contact.display, contact.phone, contact.chat_id])
    if any(needle in value.casefold() for value in fields):
        return True
    return any(needle in event.body.casefold() for event in store.events(ticket.id))


def status_label(ticket: Ticket) -> str:
    """The plain status word, including who resolved a done ticket."""
    label = STATE_LABELS.get(ticket.state, ticket.state.title())
    if ticket.state == TicketState.DONE.value and ticket.done_by:
        if ticket.done_by == DoneBy.OPERATOR.value:
            return f"{label} by you"
        return f"{label} by the {ticket.done_by}"
    return label


def status_class(ticket: Ticket) -> str:
    """A semantic class; the word always carries the meaning too."""
    if ticket.urgent or ticket.state == TicketState.NEEDS_YOU.value:
        return "urgent" if ticket.urgent else "needs-you"
    if ticket.state == TicketState.WAITING_CUSTOMER.value:
        return "waiting"
    if ticket.state == TicketState.DONE.value:
        return "done"
    if ticket.state == TicketState.CLOSED.value:
        return "off"
    return "muted"


def latest_reason(events: list[TicketEvent]) -> str:
    """The newest recorded nudge, alert or held note, when one exists."""
    for event in reversed(events):
        if event.kind in INTERNAL_KINDS and event.body.strip():
            return event.body.strip()
    return ""


def customer_summary(events: list[TicketEvent]) -> str:
    """The newest customer message as the row's secondary summary."""
    for event in reversed(events):
        if event.kind == EventKind.CUSTOMER_MESSAGE.value and event.body.strip():
            return " ".join(event.body.split())[:140]
    return ""


def contact_for(store: TicketStore, ticket: Ticket) -> Contact | None:
    """The ticket's contact row."""
    found = store.contact_by_id(ticket.contact_id)
    return found if found is not None else store.contact(ticket.chat_id)


def number_text(contact: Contact | None, ticket: Ticket) -> str:
    """The phone or chat digits without the WhatsApp suffix."""
    raw = contact.phone if contact is not None and contact.phone else ticket.chat_id
    return raw.split("@", 1)[0]


def row_for(store: TicketStore, ticket: Ticket, contact: Contact | None) -> HistoryRow:
    """Build one list row from the stored ticket text."""
    events = store.events(ticket.id)
    solved = ""
    if ticket.state == TicketState.DONE.value and ticket.done_by:
        solved = "you" if ticket.done_by == DoneBy.OPERATOR.value else ticket.done_by
    return HistoryRow(
        number=ticket.number,
        subject=ticket.summary.strip() or f"Ticket {ticket.number}",
        summary=customer_summary(events),
        status_label=status_label(ticket),
        status_class=status_class(ticket),
        person=contact.display if contact is not None else "",
        number_text=number_text(contact, ticket),
        reason=latest_reason(events),
        solved=solved,
        time=format_time(ticket.updated_at),
        updated_at=ticket.updated_at,
        urgent=bool(ticket.urgent),
    )


def filtered_rows(store: TicketStore, q: str, show: str) -> list[HistoryRow]:
    """Every ticket matching the search and Show filter."""
    rows: list[HistoryRow] = []
    for ticket in store.all_tickets():
        contact = contact_for(store, ticket)
        if not match_show(ticket, show):
            continue
        if not search_matches(store, ticket, contact, q):
            continue
        rows.append(row_for(store, ticket, contact))
    return rows


def group_rows(rows: list[HistoryRow]) -> list[HistorySection]:
    """Group into Needs you, Waiting and Resolved without hiding Open."""
    needs = [row for row in rows if row.status_class in {"urgent", "needs-you"}]
    needs.sort(key=lambda row: (not row.urgent, -row.updated_at, -row.number))
    waiting = [
        row
        for row in rows
        if row.status_class in {"waiting", "muted"}
        and row.status_label.startswith(("Open", "Waiting"))
    ]
    waiting.sort(key=lambda row: (-row.updated_at, -row.number))
    resolved = [row for row in rows if row not in needs and row not in waiting]
    resolved.sort(key=lambda row: (-row.updated_at, -row.number))
    return [
        HistorySection("needs-you", "Needs you", needs),
        HistorySection("waiting", "Waiting for the customer", waiting),
        HistorySection("resolved", "Resolved", resolved),
    ]


def back_url(q: str, show: str) -> str:
    """The list address a ticket should return to."""
    params = {key: value for key, value in (("q", q), ("show", show)) if value}
    query = f"?{urlencode(params)}" if params else ""
    return f"/app/history{query}"


def list_view(store: TicketStore, *, q: str = "", show: str = "") -> HistoryList:
    """Filter and group the tickets for the History list."""
    if show not in {value for value, _ in SHOW_CHOICES}:
        show = ""
    rows = filtered_rows(store, q, show)
    return HistoryList(
        rows=rows,
        sections=group_rows(rows),
        shown=len(rows),
        q=q,
        show=show,
        choices=SHOW_CHOICES,
        has_filters=bool(q or show),
        back_url=back_url(q, show),
    )


def safe_back(value: str) -> str:
    """Accept only a same-origin History list target, without a fragment."""
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.path != "/app/history":
        return "/app/history"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"/app/history{query}"


def attachment_views(
    attachments: list[Attachment],
    *,
    lang: str = "originals",
    language: str = "",
    translate: Translator | None = None,
    store: TicketStore | None = None,
) -> list[AttachmentView]:
    """Map attachments inline, translating a voice transcript with the thread."""
    views: list[AttachmentView] = []
    for item in attachments:
        derived = item.derived_text
        if (
            derived
            and lang != "originals"
            and language.strip().lower() not in {"", "en"}
            and translate is not None
            and store is not None
        ):
            derived = translate_cached(store, derived, TRANSLATION_LANGUAGE, translate)
        views.append(
            AttachmentView(
                id=item.id,
                kind=item.kind,
                mime=item.mime,
                derived_text=derived,
            )
        )
    return views


def describe_event(
    event: TicketEvent, contact: Contact | None
) -> tuple[str, str, str, bool, bool, bool]:
    """Return sender, style, label, internal, outgoing and customer flags."""
    internal = event.kind in INTERNAL_KINDS
    customer = event.kind == EventKind.CUSTOMER_MESSAGE.value
    outgoing = event.kind in {EventKind.REPLY.value, EventKind.SENT.value}
    if customer:
        sender = contact.display if contact is not None else "Customer"
        return sender, "customer", "Customer message", internal, outgoing, customer
    if outgoing:
        label = "Outgoing reply" if event.kind == EventKind.REPLY.value else "Relay"
        return "Assistant", "assistant", label, internal, outgoing, customer
    label = event.kind.replace("_", " ").title()
    if internal:
        sender = "Operator" if event.kind == EventKind.OPERATOR_REPLY.value else "System"
        return sender, "internal", label, internal, outgoing, customer
    return event.actor.title(), "internal", label, internal, outgoing, customer


def inline_attachments(
    event: TicketEvent, attachments: list[Attachment]
) -> list[Attachment]:
    """Attachments whose message id belongs to this event."""
    return [
        attachment
        for attachment in attachments
        if attachment.message_id and attachment.message_id in event.message_ids
    ]


def entry_for(
    event: TicketEvent,
    contact: Contact | None,
    attachments: list[Attachment],
    *,
    lang: str,
    language: str,
    translate: Translator | None,
    store: TicketStore,
) -> ThreadEntry:
    """Build one thread entry with its sender, audience and inline media."""
    sender, style, kind_label, internal, outgoing, customer = describe_event(
        event, contact
    )
    body = event.body
    if (
        lang != "originals"
        and (customer or outgoing)
        and language.strip().lower() not in {"", "en"}
    ):
        body = translate_cached(store, body, TRANSLATION_LANGUAGE, translate)
    inline = inline_attachments(event, attachments)
    return ThreadEntry(
        sender=sender,
        kind_label=kind_label,
        time=format_time(event.at),
        body=body,
        style=style,
        internal=internal,
        outgoing=outgoing,
        attachments=attachment_views(
            inline,
            lang=lang,
            language=language,
            translate=translate,
            store=store,
        ),
    )


def next_step(ticket: Ticket) -> str:
    """The one clear next action appropriate to the recorded state."""
    if ticket.state == TicketState.NEEDS_YOU.value:
        return f"Reply to the Ticket {ticket.number} message in WhatsApp."
    if ticket.state == TicketState.WAITING_CUSTOMER.value:
        return "Waiting for the customer."
    if ticket.state == TicketState.OPEN.value:
        return "The assistant is still working on this ticket."
    if ticket.state == TicketState.DONE.value:
        return f"Done by {ticket.done_by or 'the assistant'}."
    return "Closed."


def call_views(
    store: SkillStore, ticket_id: int, numbers: dict[int, int]
) -> list[CallView]:
    """The optional skill calls linked to one ticket."""
    views: list[CallView] = []
    for call in store.calls_for_ticket(ticket_id):
        views.append(
            CallView(
                skill=call.skill,
                method=call.method,
                path=call.path,
                outcome=call.outcome,
                failed=call.outcome != "ok",
                started_by=call.started_by,
                time=format_time(call.created_at),
                duration_ms=call.duration_ms,
                error=call.error,
                ticket_number=numbers.get(call.ticket_id or 0),
            )
        )
    return views


def thread_entries(
    store: TicketStore,
    ticket: Ticket,
    contact: Contact | None,
    attachments: list[Attachment],
    *,
    lang: str,
    translate: Translator | None,
) -> list[ThreadEntry]:
    """The event thread, translated once and in recorded order."""
    language = contact.language if contact is not None else ""
    return [
        entry_for(
            event,
            contact,
            attachments,
            lang=lang,
            language=language,
            translate=translate,
            store=store,
        )
        for event in store.events(ticket.id)
    ]


def add_orphan_entry(
    ticket: Ticket,
    entries: list[ThreadEntry],
    attachments: list[Attachment],
    *,
    lang: str,
    language: str,
    translate: Translator | None,
    store: TicketStore,
) -> None:
    """Show attachments that no event referenced instead of hiding them."""
    attached_ids = {
        attachment.id for entry in entries for attachment in entry.attachments
    }
    orphaned = [item for item in attachments if item.id not in attached_ids]
    if not orphaned:
        return
    entries.append(
        ThreadEntry(
            sender="System",
            kind_label="Attachment",
            time=format_time(ticket.updated_at),
            body="",
            style="internal",
            internal=True,
            outgoing=False,
            attachments=attachment_views(
                orphaned,
                lang=lang,
                language=language,
                translate=translate,
                store=store,
            ),
        )
    )


def toggle_values(ticket: Ticket, lang: str, back: str) -> tuple[str, str]:
    """The language toggle URL and its label."""
    query = "lang=en" if lang == "originals" else "lang=originals"
    url = f"/app/history/{ticket.number}?{query}&back={quote(back, safe='')}"
    label = "Show English translation" if lang == "originals" else "Show originals"
    return url, label


def ticket_page(
    store: TicketStore,
    skill_store: SkillStore,
    ticket: Ticket,
    *,
    lang: str,
    translate: Translator | None,
    back: str,
) -> TicketPage:
    """Build the full ticket detail view."""
    contact = contact_for(store, ticket)
    attachments = store.attachments(ticket.id)
    entries = thread_entries(
        store,
        ticket,
        contact,
        attachments,
        lang=lang,
        translate=translate,
    )
    add_orphan_entry(
        ticket,
        entries,
        attachments,
        lang=lang,
        language=contact.language if contact is not None else "",
        translate=translate,
        store=store,
    )
    calls = call_views(skill_store, ticket.id, {ticket.id: ticket.number})
    toggle_url, toggle_label = toggle_values(ticket, lang, back)
    return TicketPage(
        ticket=ticket,
        subject=ticket.summary.strip() or f"Ticket {ticket.number}",
        summary=ticket.summary,
        status_label=status_label(ticket),
        status_class=status_class(ticket),
        next_step=next_step(ticket),
        contact_name=contact.display if contact is not None else "",
        contact_number=number_text(contact, ticket),
        language=contact.language if contact is not None else "",
        entries=entries,
        calls=calls,
        toggle_url=toggle_url,
        toggle_label=toggle_label,
        back=back,
    )
