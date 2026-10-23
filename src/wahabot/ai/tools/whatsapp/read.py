import time
from typing import Any

from llama_index.core.tools import BaseTool, FunctionTool
from loguru import logger

from wahabot.ai.messages import jid_string
from wahabot.ai.serials import serial_store
from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.outfile import write_json_output
from wahabot.ai.tools.runcontext import (
    FENCE_ERROR,
    RunTarget,
    chat_jid,
    current_target,
    fenced_chat,
    operator_run,
)
from wahabot.ai.tools.schemas import (
    ReadChatSchema,
)
from wahabot.ai.tools.whatsapp.roster import (
    RECENT_CHATS_CAP,
    resolve_in_current_chat,
    search_matches,
    sender_names,
    summarize_chat,
)
from wahabot.ai.tools.whatsapp.targets import resolve_message_token
from wahabot.ai.turn_records import read_record
from wahabot.core.identity import AUTHOR_BOT, AUTHOR_OPERATOR, authors, display_name
from wahabot.core.journal import chat_messages
from wahabot.core.transport import Transport
from wahabot.settings import Settings


def slim_message(message: dict[str, Any], max_body: int = 200) -> dict[str, Any]:
    """The model-relevant fields of a WAHA message, without the noise.

    WAHA messages carry a raw ``_data`` blob (messageSecret,
    reportingToken, engine flags — ~90% of the payload) that is useless
    to the model and inflates every tool result. Slimmed messages keep
    valid JSON and stay small enough for the inline preview. Message
    bodies are capped at *max_body* chars for the inline slice;
    ``body_truncated`` marks a cut body, whose full text lives in the
    spill file :func:`fit_messages` writes — nothing requested is
    silently dropped.
    """
    keys = ("id", "timestamp", "from", "fromMe", "participant", "body", "hasMedia", "ack")
    slimmed = {key: message[key] for key in keys if message.get(key) is not None}
    slimmed.update(sender_fields(message))
    body = slimmed.get("body")
    if isinstance(body, str) and len(body) > max_body:
        slimmed["body"] = body[:max_body] + "…"
        slimmed["body_truncated"] = True
    return slimmed


def sender_fields(message: dict[str, Any]) -> dict[str, str]:
    """Who wrote a fetched message: normalized JID, name, and account author.

    WAHA's history API carries no ``notifyName`` in LID groups, so the
    name comes from the webhook-learned name book. The bot and the
    human operator share the account, so ``fromMe`` alone cannot say
    who wrote a message: ``author`` is the recorded teammate
    (``bot``/``operator``) or ``unknown`` for messages from before the
    record started.
    """
    data = message.get("_data", {})
    sender = jid_string(message.get("participant") or data.get("author"))
    fields: dict[str, str] = {}
    if sender:
        fields["participant"] = sender
    name = str(data.get("notifyName") or "").strip() or display_name(sender)
    if name:
        fields["name"] = name
    if message.get("fromMe"):
        fields["author"] = authors.author(str(message.get("id") or "")) or "unknown"
    return fields


def fit_messages(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    session: str = "",
    chat_id: str = "",
) -> dict[str, Any]:
    """Slim every fetched message inline and spill the raw window to a file.

    Nothing requested is silently dropped: every message appears in
    the inline list (bodies capped by :func:`slim_message`), and the
    full raw window — complete ``_data``, uncapped bodies — goes to the
    spill file, whose metadata rides along as ``file``.
    ``returned``/``message_count`` agree; each message's
    ``body_truncated`` flags cut text whose full form is in the file.
    Empty windows omit count/file; spill failure omits ``file`` (and
    the inline list is still complete).

    With the records grammar, the inline view is the same shape the prompt
    reads (``records`` instead of ``messages``), so the console and the
    model agree and every row carries its ``#m<n>``. The raw spill is
    unchanged.
    """
    kept = [slim_message(message) for message in messages]
    out: dict[str, Any] = {"returned": len(kept)}
    if settings is not None:
        registry = serial_store(str(settings.data_dir)).for_chat(session, chat_id)
        out["records"] = [
            read_record(message, registry, settings.timezone) for message in messages
        ]
    else:
        out["messages"] = kept
    if messages:
        out["message_count"] = len(messages)
        try:
            out["file"] = write_json_output(f"chat_{int(time.time())}", messages)
        except Exception as exc:
            logger.warning("could not spill chat messages to file: {exc}", exc=exc)
    return out


_JOURNAL_AUTHORS_BY_SOURCE: dict[str, str] = {"api": AUTHOR_BOT, "app": AUTHOR_OPERATOR}

_LIST_WINDOW = 20


def journal_author(payload: dict[str, Any], recorded: Any) -> str:
    """Who wrote an account's journal row: live record first, else source.

    ``recorded`` is the author book's entry for the id (``"unknown"``
    when absent): the book is the recorded truth for ids seen live,
    since WAHA history and later quote lookups lose ``source``. The
    journal keeps ``source`` on every event, so it authorizes the whole
    recorded span — including rows from before the book began.
    """
    if recorded not in (None, "", "unknown"):
        return str(recorded)
    return _JOURNAL_AUTHORS_BY_SOURCE.get(str(payload.get("source") or ""), "unknown")


def export_row(payload: dict[str, Any]) -> dict[str, Any]:
    """One slim, identity-annotated row of exported history; full body.

    Same shape as the ``list`` preview but with an uncapped body, so a
    script over the spill file greps what ``list`` would have shown.
    Account rows carry ``author`` (see :func:`journal_author`); bot rows
    drop ``name`` — the shared account's display name belongs to the
    operator, and on a bot row it would read as his words.
    """
    row = {
        key: payload[key]
        for key in ("id", "timestamp", "from", "fromMe", "hasMedia")
        if payload.get(key) is not None
    }
    row.update(sender_fields(payload))
    row["body"] = str(payload.get("body") or "")
    if row.get("fromMe"):
        row["author"] = journal_author(payload, row.get("author"))
        if row["author"] == AUTHOR_BOT:
            row.pop("name", None)
    return row


def export_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The inline export summary: author and member counts plus the spill.

    The envelope states the shared account's presence itself — one
    count per teammate — so figures built from the file cannot "lose"
    either of them; the rows, full bodies included, go to the file,
    not the context.
    """
    if not rows:
        return {}
    account: dict[str, int] = {}
    members: dict[str, int] = {}
    for row in rows:
        if row.get("fromMe"):
            author = str(row.get("author") or "unknown")
            account[author] = account.get(author, 0) + 1
        else:
            member = str(row.get("name") or row.get("participant") or "?")
            members[member] = members.get(member, 0) + 1
    out: dict[str, Any] = {
        "account_messages": account,
        "members": members,
        "span": (rows[0].get("timestamp"), rows[-1].get("timestamp")),
    }
    try:
        out["file"] = write_json_output("chat_export", rows)
    except Exception as exc:
        logger.warning("could not spill exported rows to file: {exc}", exc=exc)
    return out


def read_export_journal(
    settings: Settings, session: str, chat_id: str, limit: int | None
) -> str:
    """The ``export`` mode: the chat's recorded history in one file.

    Month-scale material for figures and registers: every journal row
    of the chat, slimmed, identity-annotated, full bodies — bounded by
    ``limit`` (newest kept) when the call asks for less.
    """
    payloads = chat_messages(settings.journal_dir, session, chat_id)
    recorded = len(payloads)
    if limit is not None and limit < recorded:
        payloads = payloads[-limit:]
    rows = [export_row(payload) for payload in payloads]
    return ok(
        chat=chat_id,
        mode="export",
        returned=len(rows),
        recorded=recorded,
        **export_counts(rows),
    )


def dispatch_export(
    settings: Settings, chat: str | None, target: RunTarget, limit: int | None
) -> str:
    """Fence then export: only operator runs may select another chat."""
    chat_id, fence_error = fenced_chat(chat, target)
    if fence_error:
        return error(fence_error)
    if not chat_id:
        return error("no active conversation context")
    if limit is not None and limit <= 0:
        return error("limit must be positive")
    target.allowed_chats.add(chat_id)
    return read_export_journal(settings, target.session, chat_id, limit)


def resolve_read_chat(
    chat: str | None, target: RunTarget, settings: Settings
) -> tuple[str, str | None]:
    """The fenced chat for a read, resolving a ``#m<n>`` chat argument."""
    if chat and chat.strip().startswith("#m"):
        raw, token_error = resolve_message_token(
            chat,
            session=target.session,
            chat_id="" if operator_run(target) else target.chat_id,
            settings=settings,
            allow_search=operator_run(target),
            allowed_chats=target.allowed_chats,
        )
        if token_error:
            return "", token_error
        chat = chat_jid(raw, target)
    chat_id, fence_error = fenced_chat(chat, target)
    if fence_error:
        return "", fence_error
    if not chat_id:
        return "", "no active conversation context"
    return chat_id, None


def read_chat(waha: Transport, settings: Settings) -> BaseTool:
    """Build the chat-reading tool: list/search/metadata/recent/resolve.

    ``mode`` is required in model calls. List/search inspect a bounded
    recent window; recent is operator-only; resolve searches available
    names without accepting ``chat``. Metadata is best-effort.
    The shell setting controls whether the model can read raw spills.
    """
    deref = (
        "Read the raw spill via run_shell_command and file.path for full metadata."
        if settings.shell_tool
        else "The operator can open file.path for the raw window and metadata."
    )

    def read_chat_fn(
        mode: str = "list",
        chat: str | None = None,
        query: str = "",
        name: str = "",
        limit: int | None = None,
        reason: str = "",
    ) -> str:
        target = current_target()
        if not target.session:
            return error("no active conversation context")
        if mode in ("recent", "resolve"):
            return read_chat_directory(waha, target, mode, chat, name, limit)
        if mode == "export":
            return dispatch_export(settings, chat, target, limit)
        handler = _READ_CHAT_HANDLERS.get(mode)
        if handler is None:
            return error(f"unknown mode: {mode!r}")
        if mode in ("list", "search") and limit is not None and limit <= 0:
            return error("limit must be positive")
        chat_id, chat_error = resolve_read_chat(chat, target, settings)
        if chat_error:
            return error(chat_error)
        target.allowed_chats.add(chat_id)
        return handler(waha, target.session, chat_id, query, name, limit, settings)

    return FunctionTool.from_defaults(
        fn=read_chat_fn,
        fn_schema=ReadChatSchema,
        name="read_chat",
        description=(
            "Read available chat context. list/search return one prompt record "
            "per fetched message (`[[hh:mm speaker #m<n>]] kind :: body`) and "
            "spill the full raw window to a JSON file whose metadata comes "
            "back as file. Rows the registry knows carry their `#m<n>`; older "
            "rows may have none. returned/message_count count fetched/matched "
            "messages, not all history. Never quote unseen content. "
            f"{deref} A search miss applies only to its recent window. "
            "Metadata returns available fields/counts; missing fields are "
            "unknown, names are display labels, and participant_list appears "
            "only for up to 20 known JIDs. resolve returns up to five available "
            "name matches: current roster, or operator's first 200 chats then "
            "first 500 contacts only if no chat matches. Misses are not "
            "exhaustive. export spills the chat's deep recorded history to one "
            "annotated file, teammates and members counted inline."
        ),
    )


def read_chat_directory(
    waha: Transport,
    target: RunTarget,
    mode: str,
    chat: str | None,
    name: str,
    limit: int | None,
) -> str:
    """Handle recent/resolve lookups, which do not accept a chat argument."""
    if chat:
        return error(f"`chat` is not valid for mode={mode}")
    if mode == "recent":
        return read_recent_chats(waha, target.session, target, limit)
    return read_resolve_chat(waha, target.session, target, name)


def read_recent_chats(
    waha: Transport, session: str, target: RunTarget, limit: int | None
) -> str:
    """Newest conversations for operator commands, clamped to at most 30."""
    if not operator_run(target):
        return error(FENCE_ERROR)
    if limit is None:
        limit = RECENT_CHATS_CAP
    if limit <= 0:
        return error("limit must be positive")
    try:
        chats = waha.list_chats(session, limit=min(limit, RECENT_CHATS_CAP))
    except Exception as exc:
        return error(f"could not list chats: {exc}")
    entries = [
        {"id": jid_string(c.get("id", "")), "name": str(c.get("name", ""))}
        for c in chats
        if c.get("id")
    ]
    target.allowed_chats.update(entry["id"] for entry in entries)
    return ok(chats=entries)


def read_resolve_chat(waha: Transport, session: str, target: RunTarget, name: str) -> str:
    """Available-name matching: current roster, or chats then contacts.

    Operator scope is one page of 200 chats, then 500 contacts only
    if no chat matches; any partial chat match suppresses contact lookup.
    At most five exact/substring candidates are returned, without paging.
    """
    name = name.strip()
    if not name:
        return error("name is required")
    if not target.chat_id:
        return error("no active conversation context")
    if not operator_run(target):
        return resolve_in_current_chat(waha, session, target.chat_id, name)
    matches: list[dict[str, Any]] = []
    try:
        chats = waha.list_chats(session)
        matches = search_matches(chats, name)
        if not matches:
            contacts = waha.list_contacts(session)
            matches = search_matches(contacts, name)
    except Exception as exc:
        return error(f"could not search chats: {exc}")
    if not matches:
        return error(
            f"no match for {name!r} in the checked chats/contacts; not exhaustive"
        )
    target.allowed_chats.update(match["id"] for match in matches)
    return ok(name=name, matches=matches)


def read_chat_messages(
    waha: Transport,
    session: str,
    chat_id: str,
    query: str,
    name: str,
    limit: int | None,
    settings: Settings | None = None,
) -> str:
    """The `list` mode: recent messages from a chat, newest first.

    Every nonempty fetched window attempts a raw JSON spill alongside
    the slim inline preview (see :func:`fit_messages`). No paging.
    """
    messages = waha.fetch_chat_messages(session, chat_id, limit=limit or _LIST_WINDOW)
    return ok(
        chat=chat_id,
        **fit_messages(messages, settings=settings, session=session, chat_id=chat_id),
    )


def read_search_messages(
    waha: Transport,
    session: str,
    chat_id: str,
    query: str,
    name: str,
    limit: int | None,
    settings: Settings | None = None,
) -> str:
    """Substring-filter the latest *limit* messages, not the full history.

    All nonempty matched windows attempt a raw spill like ``list``.
    """
    if not query.strip():
        return error("query is required")
    messages = waha.search_messages(session, query, chat_id, limit=limit or _LIST_WINDOW)
    return ok(
        chat=chat_id,
        query=query,
        **fit_messages(messages, settings=settings, session=session, chat_id=chat_id),
    )


def read_chat_metadata(
    waha: Transport,
    session: str,
    chat_id: str,
    query: str,
    name: str,
    limit: int,
    settings: Settings | None = None,
) -> str:
    """Available overview fields and roster summary; names may be missing."""
    overview = waha.get_chat_overview(session, chat_id)
    if not overview:
        return error(f"no metadata found for {chat_id}")
    names = sender_names(waha, session, chat_id)
    return ok(**summarize_chat(chat_id, overview, names))


_READ_CHAT_HANDLERS: dict[str, Any] = {
    "list": read_chat_messages,
    "search": read_search_messages,
    "metadata": read_chat_metadata,
}
