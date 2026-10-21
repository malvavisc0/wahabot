import asyncio

from loguru import logger

from wahabot.core.echoes import remember_self_echo
from wahabot.core.waha import WahaClient
from wahabot.status.state import state


async def notify_operator(waha: WahaClient, session: str, message: str) -> None:
    """Best-effort WhatsApp message to the bot's own account.

    The operator reads it in the "Message yourself" chat. Fails soft:
    the session may be dead — the very thing being reported — so a
    failed send is logged and swallowed; the loud log line is the floor.
    """
    try:
        await send_operator_notify(waha, session, message)
    except Exception as exc:
        logger.warning("Operator notification failed: {exc}", exc=exc)


async def send_operator_notify(waha: WahaClient, session: str, message: str) -> str:
    """Send one operator notification; raises on failure (the retry loop).

    Split from :func:`notify_operator` so the recovery path can
    *retry* a failed recovery notice — the single-shot swallow is right
    for the alert (the session is down, retrying is futile) but loses
    the recovery message when the send races WhatsApp's own reconnect.

    The prefix stays ``wahabot:`` alone — the old
    ``WAHA session '{session}' {message}`` prefix sat in front of
    LLM notices too, telling the operator their WhatsApp session had
    a problem when the LLM provider did. Each caller writes its own
    accurate subject line instead.
    """
    me = state.operator_jid
    if not me:
        return ""
    text = f"wahabot: {message}"
    sent_id = await asyncio.to_thread(waha.send_text, session, me, text)
    # The notification lands in the self-chat; mark it so its echo
    # event is never parsed as an operator command.
    remember_self_echo(sent_id)
    return sent_id
