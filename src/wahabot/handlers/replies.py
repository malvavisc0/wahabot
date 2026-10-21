import asyncio
import random
from collections.abc import Sequence

import httpx
from loguru import logger

from wahabot.ai.context import (
    is_single_emoji,
)
from wahabot.ai.tools.whatsapp import RunTarget, deliver_chat_text
from wahabot.core.audit import save_action
from wahabot.core.echoes import remember_self_echo
from wahabot.core.models import WahaEvent
from wahabot.core.waha import WahaClient
from wahabot.settings import Settings


def log_final_text(chat_id: str, reply: str) -> None:
    """Log a dropped post-delivery final text, when there is one.

    After a delivery tool fired, the run's final text is dropped (the
    reply already went out); it is usually the model narrating what it
    just did, but it can carry real content — log it so nothing the
    model "said" vanishes without a trace.
    """
    if reply and reply.strip():
        logger.info(
            "Dropped post-delivery final text in {chat_id}: {reply}",
            chat_id=chat_id,
            reply=reply[:500],
        )


def journal_send_failure(
    settings: Settings,
    session: str,
    chat_id: str,
    action: str,
    message_id: str,
    exc: Exception,
) -> None:
    """Journal a failed final-reply send.

    The single, burst and album reply paths share this: a send that
    raised left nothing in the chat, so the audit row must say the
    send *failed* — never a ``reply``/``reaction`` record the chat
    never saw (the same standard the durable escalation record set:
    the record follows the confirmed send).
    """
    save_action(
        settings.data_dir,
        session,
        "send_failed",
        chat_id=chat_id,
        action=action,
        message_id=message_id,
        error=str(exc),
    )
    logger.warning(
        "Failed to deliver reply to {chat_id} via {action}: {exc}",
        chat_id=chat_id,
        action=action,
        exc=exc,
    )


async def finish_agent_reply(
    waha: WahaClient,
    session: str,
    chat_id: str,
    message_id: str,
    reply: str,
    delivered: bool,
    settings: Settings,
    prefix: str = "",
    emoji_reaction: bool = False,
    target: RunTarget | None = None,
    message_ids: Sequence[str] = (),
) -> None:
    """Deliver a finished run's reply text, or record why not.

    Shared post-run decision of the single-message, burst and album
    paths: tool-delivered runs log their dropped final text; silence
    stays silent; anything else is quote-replied to *message_id* with
    the typing presence. A lone emoji is not an answer — *emoji_reaction*
    converts it into a reaction on *message_id* with ~50 % probability,
    otherwise it is dropped as silence. The burst path opts in, so a
    burst's lone-emoji answer reacts like a single message's.

    Every branch journals its decision
    — the audit timeline shows *why* a chat saw nothing: delivered via
    tool, silence, or a dropped emoji. Records for actions that send
    (``reply``, ``reaction``) are written only after the send
    succeeded — the same standard the durable escalation record set —
    so a failed send never leaves an audit row claiming the chat was
    answered; its failure is journaled as ``send_failed`` instead.
    """

    async def react_or_silence(emoji: str) -> None:
        """React to a lone emoji, or journal it as silence."""
        if emoji_reaction and message_id and random.random() < 0.5:
            logger.info(
                "Converting lone emoji to reaction on {mid}: {emoji}",
                mid=message_id,
                emoji=emoji,
            )
            try:
                await asyncio.to_thread(waha.send_reaction, session, message_id, emoji)
            except Exception as exc:
                journal_send_failure(
                    settings, session, chat_id, "send_reaction", message_id, exc
                )
                return
            save_action(
                settings.data_dir,
                session,
                "reaction",
                chat_id=chat_id,
                reaction=emoji,
                message_id=message_id,
                message_ids=list(message_ids),
                delivered_via="reaction",
            )
            return
        save_action(
            settings.data_dir,
            session,
            "silence",
            chat_id=chat_id,
            dropped_emoji=emoji,
            message_id=message_id,
            message_ids=list(message_ids),
            delivered_via="silence",
        )
        logger.info(
            "Dropping lone emoji as silence for {chat_id}: {emoji}",
            chat_id=chat_id,
            emoji=emoji,
        )

    if delivered:
        logger.info(
            "Agent decision for {chat_id}: delivered via tool",
            chat_id=chat_id,
        )
        reacted = target.reacted if target is not None else ""
        sent_id = target.sent_id if target is not None else ""
        save_action(
            settings.data_dir,
            session,
            "delivered_via_tool",
            chat_id=chat_id,
            message_id=message_id,
            message_ids=list(message_ids),
            sent_id=sent_id,
            reacted=reacted,
            delivered_via="reaction" if reacted and not sent_id else "tool",
        )
        log_final_text(chat_id, reply)
        return
    if not reply or not reply.strip():
        save_action(
            settings.data_dir,
            session,
            "silence",
            chat_id=chat_id,
            message_id=message_id,
            message_ids=list(message_ids),
            delivered_via="silence",
        )
        logger.info("Agent decision for {chat_id}: stay silent", chat_id=chat_id)
        return
    if is_single_emoji(reply):
        await react_or_silence(reply.strip())
        return
    logger.info("Replying to {chat_id}: {reply}", chat_id=chat_id, reply=reply[:500])
    try:
        sent_id, _, _ = await asyncio.to_thread(
            deliver_chat_text,
            waha,
            session,
            chat_id,
            reply,
            message_id,
            typing=(
                settings.typing_presence_min_s,
                settings.typing_presence_max_s,
            ),
            prefix=prefix,
        )
    except Exception as exc:
        journal_send_failure(settings, session, chat_id, "send_text", message_id, exc)
        return
    save_action(
        settings.data_dir,
        session,
        "reply",
        chat_id=chat_id,
        message_id=message_id,
        message_ids=list(message_ids),
        sent_id=sent_id,
        delivered_via="text",
        reply=reply,
    )


_SELF_REPLY_RETRY_DELAYS_S = (2.0, 5.0, 0.0)


async def send_self_reply(waha: WahaClient, command: WahaEvent, reply: str) -> None:
    """Deliver a self-chat command's final text back to the operator.

    The command ran on a fresh context with no chat of its own; its
    final reply (the part not delivered via a tool) lands in the same
    "message yourself" chat the command came from, quote-replying the
    command message. The sent id is remembered so the echo of our own
    reply — which arrives as a ``fromMe`` event matching the mention
    regex when the reply quotes the trigger — cannot re-trigger the
    command path.
    """
    if not reply or not reply.strip():
        return
    chat_id = str(command.payload.get("reply_chat_id", ""))
    if not chat_id:
        return
    reply_to = str(command.payload.get("reply_to", "")) or None
    sent_id = ""
    for attempt, delay in enumerate(_SELF_REPLY_RETRY_DELAYS_S):
        try:
            sent_id, _, _ = await asyncio.to_thread(
                deliver_chat_text,
                waha,
                command.session,
                chat_id,
                reply,
                reply_to,
                typing=None,  # self-chat: no one is watching for "typing…"
            )
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # WAHA restarting refuses connections for a few seconds; the
            # answer is worth a short wait rather than being lost. Only
            # connection-establishment failures retry: a read/write
            # timeout may have delivered the send already, and a retry
            # would duplicate it.
            if attempt == len(_SELF_REPLY_RETRY_DELAYS_S) - 1:
                raise
            logger.warning(
                "Self-chat reply not delivered ({exc}); retrying in {delay}s",
                exc=exc,
                delay=delay,
            )
            await asyncio.sleep(delay)
    remember_self_echo(sent_id)
