from typing import Any

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.messages import is_blame_text
from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.runcontext import (
    current_target,
    delivered_to_self,
    delivery_message_error,
    fenced_message_id,
    log_action_reason,
    operator_run,
    reaction_guard,
    send_failed_envelope,
    send_guard,
)
from wahabot.ai.tools.schemas import (
    ReactToMessageSchema,
    SendMessageSchema,
    StaySilentSchema,
)
from wahabot.ai.tools.whatsapp.roster import (
    chat_roster,
    dangling_mentions,
    mention_tokens,
    ordered_merge,
    prepare_mentions,
    resolve_explicit_mentions,
    resolve_mentions,
)
from wahabot.ai.tools.whatsapp.targets import (
    prepare_send_target,
    resolve_mention_tokens,
    resolve_message_token,
)
from wahabot.core.presence import clear_typing, typing_pause
from wahabot.core.transport import Transport
from wahabot.settings import Settings


def blame_delivery_error(text: str) -> str | None:
    """The correction-guard error for a delivery text, or None."""
    if not is_blame_text(text):
        return None
    return (
        "correction-guard: do not tell the recipient they were wrong; "
        "apologize plainly or stay silent"
    )


def send_message(waha: Transport, settings: Settings | None = None) -> BaseTool:
    """Build a tool that sends a WhatsApp text message.

    Sends to the current chat by default; an operator-command run may
    pass an explicit ``chat`` (a group or a person's JID) to reach a
    different target — a normal chat run cannot (the fence refuses,
    see :func:`fenced_chat`).

    One successful send per run, shared with media and forwarding.
    Further send calls return an error; reactions have a separate latch.

    ``@<number>`` tokens in *text* that name a roster member become
    real mentions automatically (WhatsApp shows people this way, e.g.
    "Para @111222333444555" — the model copying that shape into its
    reply must still use a numeric token). Explicit ``mentions`` JIDs
    are merged in, but neither a send acknowledgment nor this tool's
    warnings establish whether WhatsApp notified a recipient.
    """

    def send_message_fn(
        chat: str | None = None,
        text: str = "",
        reply_to: str | None = None,
        mentions: list[str] | None = None,
        reason: str = "",
    ) -> str:
        if not text.strip():
            return error("empty message text")
        if blame := blame_delivery_error(text):
            return error(blame)
        target = current_target()
        if refusal := send_guard(target):
            return refusal
        chat_id, reply_to, target_error = prepare_send_target(
            chat, reply_to, target, settings
        )
        if target_error:
            return error(target_error)
        session = target.session
        log_action_reason("send_message", reason, chat=chat_id)
        mentions = resolve_mention_tokens(
            mentions,
            session=session,
            chat_id=chat_id,
            settings=settings,
            allow_search=operator_run(target),
        )
        try:
            sent_id, merged, dangling = deliver_chat_text(
                waha,
                session,
                chat_id,
                text,
                reply_to=reply_to,
                mentions=mentions,
                typing=target.typing,
                prefix=target.prefix,
            )
        except Exception as exc:
            return send_failed_envelope("send_message", chat_id, exc)
        delivered_to_self(chat_id, sent_id)
        target.sent = chat_id
        target.sent_id = sent_id
        return sent_message_envelope(
            chat_id, apply_chat_prefix(text, target.prefix), merged, dangling
        )

    return FunctionTool.from_defaults(
        fn=send_message_fn,
        fn_schema=SendMessageSchema,
        name="send_message",
        description=(
            "Send text to the current chat, or an explicit operator target. "
            "reply_to quotes an exact message by `#m<n>` token or raw id. "
            "Numeric @-tokens are auto-resolved against the roster; "
            "warnings identify unresolved tokens, not notification receipts. "
            "Shares one successful send per run with media and forwarding."
        ),
    )


def sent_message_envelope(
    chat_id: str, text: str, merged: list[str], dangling: list[str]
) -> str:
    """Describe a confirmed text send, retaining any mention-resolution warning."""
    fields: dict[str, Any] = {
        "chat": chat_id,
        "text": text,
        "mentions": merged,
    }
    if merged and not mention_tokens(text):
        fields["warning"] = (
            "mention JIDs were supplied without @<user-part> tokens; "
            "the text was sent, but mention rendering and notification "
            "are not confirmed"
        )
    elif dangling:
        fields["warning"] = (
            f"{' and '.join(f'`@{t}`' for t in dangling)} could not be "
            "auto-resolved against the roster; other explicit or resolved "
            "mentions may still be present. Resolve JIDs before future sends."
        )
    return ok(**fields)


def stay_silent() -> BaseTool:
    """Build the explicit silence tool: choose to not reply at all.

    Models follow a tool call far more reliably than the "reply with
    an empty string" instruction — without this, small models narrate
    their silence ("I'll stay silent here — ...") and the narration is
    sent to the chat as a normal reply. The workflow intercepts this
    call before executing any tools in its batch, logs its reason, and
    stops the run. Calling the function directly only returns ``ok``.
    """

    def stay_silent_fn(reason: str = "") -> str:
        log_action_reason("stay_silent", reason)
        return ok()

    return FunctionTool.from_defaults(
        fn=stay_silent_fn,
        fn_schema=StaySilentSchema,
        name="stay_silent",
        description=(
            "End the run without replying when no response is warranted. "
            "Call alone: any other tools in the same batch are canceled. "
            "Never send text to narrate silence."
        ),
    )


def _reaction_chat_id(target: Any) -> str:
    """The chat a reaction belongs to; operator reactions search freely."""
    return "" if operator_run(target) else str(target.chat_id)


def react_to_message(waha: Transport, settings: Settings | None = None) -> BaseTool:
    """Build a tool that reacts to a WhatsApp message.

    One successful reaction change (including removal) per run.
    Further reaction calls fail; the send latch is independent.
    ``message_id`` may be a ``#m<n>`` token; operator commands resolve it
    across the session when their synthetic context has no chat.
    """

    def react_to_message_fn(message_id: str, reaction: str = "", reason: str = "") -> str:
        target = current_target()
        if refusal := reaction_guard(target):
            return refusal
        session = target.session
        if not session:
            return error("no active conversation context")
        if not message_id:
            return error("message_id is required")
        message_id, token_error = resolve_message_token(
            message_id,
            session=session,
            chat_id="" if operator_run(target) else target.chat_id,
            settings=settings,
            allow_search=operator_run(target),
            allowed_chats=target.allowed_chats,
        )
        if token_error:
            return error(token_error)
        _, id_error = fenced_message_id(message_id, target)
        if id_error:
            return error(id_error)
        if not operator_run(target):
            delivery_error = delivery_message_error(message_id, target)
            if delivery_error:
                return error(delivery_error)
        log_action_reason("react_to_message", reason, message_id=message_id)
        try:
            waha.send_reaction(session, _reaction_chat_id(target), message_id, reaction)
        except Exception as exc:
            return send_failed_envelope("react_to_message", message_id, exc)
        target.reacted = message_id
        return ok(message_id=message_id, reaction=reaction, removed=not reaction)

    return FunctionTool.from_defaults(
        fn=react_to_message_fn,
        fn_schema=ReactToMessageSchema,
        name="react_to_message",
        description=(
            "Add or remove the bot's reaction on an exact `#m<n>` token or raw id. "
            "Use an emoji to add it or an empty reaction to remove it. "
            "One successful change per run, separate from the send allowance."
        ),
    )


def apply_chat_prefix(text: str, prefix: str) -> str:
    """Prepend a chat's text prefix, once, to non-empty chat text.

    Idempotent: text already starting with the prefix — exactly, or
    with the prefix's trailing whitespace dropped, as a model copying
    the prefix from its own prefixed history may write it — is left
    alone, so the prefix is never doubled.
    """
    if not prefix.strip() or not text.strip():
        return text
    lead = text.lstrip()
    if lead.startswith(prefix) or lead.startswith(prefix.rstrip()):
        return text
    return f"{prefix}{text}"


def deliver_chat_text(
    waha: Transport,
    session: str,
    chat_id: str,
    text: str,
    reply_to: str | None = None,
    mentions: list[str] | None = None,
    typing: tuple[float, float] | None = None,
    prefix: str = "",
) -> tuple[str, list[str], list[str]]:
    """Send text with the full delivery treatment.

    Every text delivery — the ``send_message`` tool and the handler's
    final-reply send alike — goes through here, so mention resolution
    is a property of sending, not of which path fired: ``@``-tokens in
    *text* that name roster members become mention JIDs in both.
    Unresolved tokens remain literal text; explicit *mentions* JIDs
    merge in ahead of resolved ones. The send response does not confirm
    mention notification or rendering.

    *prefix* is the chat's ``chat-message-prefix`` rule: applied to the
    outgoing text before mention resolution, so the sent text and the
    stored turn (``visible_message``) carry the same prefix.

    *typing* — a ``(min_s, max_s)`` window from settings — adds the
    human-presence prelude: show "typing…", wait a length-scaled
    random moment (``None`` or ``min <= 0`` skips it). It lives here
    because both delivery paths must type alike; the caller passes the
    settings values, keeping this module settings-free.

    Returns ``(sent_id, merged_mentions, dangling)`` — the id of the
    sent message ("" when the response carried none), the merged JID
    list, and the ``@``-tokens that named no roster member, so a
    caller with a use for any of them (the tool's envelope, the echo
    guard's id) gets them without a second API call.
    """
    text = apply_chat_prefix(text, prefix)
    indicator = (
        typing_pause(waha, session, chat_id, text, typing[0], typing[1])
        if typing is not None
        else False
    )
    try:
        roster = chat_roster(waha, session, chat_id)
        text, roster = prepare_mentions(waha, session, text, roster)
        explicit, unresolved = resolve_explicit_mentions(
            waha, session, chat_id, mentions or [], roster
        )
        merged = ordered_merge(explicit, resolve_mentions(text, roster))
        dangling = dangling_mentions(text, roster)
        sent_id = waha.send_text(
            session, chat_id, text, reply_to=reply_to, mentions=merged or None
        )
    except Exception:
        # The send that would have cleared the indicator never landed;
        # best-effort clear it here, never masking the original error.
        if indicator:
            clear_typing(waha, session, chat_id)
        raise
    return sent_id, merged, [*dangling, *unresolved]
