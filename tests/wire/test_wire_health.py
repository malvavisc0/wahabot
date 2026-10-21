import json
import time
import unittest.mock
from pathlib import Path
from typing import Any

import httpx

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    ME_JID,
    SESSION,
    SecondResponse,
    chat_event,
    tool_call_response,
    waha_event,
)
from tests.wire.support import (
    _wait,
)
from wahabot.core.cli import build_forget_event
from wahabot.core.echoes import remember_self_echo
from wahabot.core.persistence import load_memory, memory_file
from wahabot.core.runs import contexts as handlers_contexts
from wahabot.handlers import seen_recently


def test_recovery_notify_retries_until_sent(bot: Bot) -> None:
    """A recovery notify racing WhatsApp's reconnect retries, not vanish.

    The first two send attempts fail (WhatsApp still settling); the
    third lands. The old single-shot swallow lost exactly this message.
    """
    import wahabot.status.session_health as status_module
    from wahabot.status import session_healthy
    from wahabot.status.session_health import NOTIFY_BACKOFF_S

    status_module.NOTIFY_BACKOFF_S = 0.1  # fast backoff for the test
    real_send_text = bot.waha.send_text
    try:
        bot.waha.session_status = "FAILED"  # poller probes must miss
        bot.post(
            {
                "id": "evt-retry-down",
                "timestamp": int(time.time()),
                "event": "session.status",
                "session": SESSION,
                "me": None,
                "payload": {"name": SESSION, "status": "FAILED"},
            }
        )
        assert _wait(lambda: not session_healthy())
        attempts = {"n": 0}
        real_send_text = bot.waha.send_text

        def flaky_send_text(
            session: str,
            chat_id: str,
            text: str,
            reply_to: str | None = None,
            mentions: list[str] | None = None,
        ) -> str:
            if "WhatsApp session recovered" in text and attempts["n"] < 2:
                attempts["n"] += 1
                raise httpx.HTTPError("WhatsApp still settling")
            return real_send_text(session, chat_id, text, reply_to, mentions)

        bot.waha.send_text = flaky_send_text  # type: ignore[method-assign]
        bot.post(
            {
                "id": "evt-retry-up",
                "timestamp": int(time.time()),
                "event": "session.status",
                "session": SESSION,
                "me": None,
                "payload": {"name": SESSION, "status": "WORKING"},
            }
        )
        assert _wait(lambda: session_healthy())
        assert _wait(
            lambda: any(
                entry[1] == ME_JID and "WhatsApp session recovered" in entry[2]
                for entry in bot.waha.sent
            ),
            timeout=5,
        )
        assert attempts["n"] == 2  # both failures were retried past
    finally:
        status_module.NOTIFY_BACKOFF_S = NOTIFY_BACKOFF_S
        bot.waha.send_text = real_send_text  # type: ignore[method-assign]


def test_llm_outage_notifies_operator_once(bot: Bot) -> None:
    """A dead LLM endpoint flips the health flag and notifies once.

    The operator gets one 🟠 in the self-chat for the whole outage —
    not one per dropped message — and WAHA redeliveries still retry
    (seen marker dropped). Recovery on the first successful run sends
    exactly one 🔵. Both outage shapes covered: connection refused and
    a dead proxy answering 502.
    """
    dead_bot = bot.rebuild(llm_api_base="http://127.0.0.1:1/v1")
    from wahabot.status import llm_healthy

    for i in range(3):  # three messages while the provider is down
        dead_bot.post(waha_event(f"LLMDOWN{i}"))
    assert _wait(lambda: not llm_healthy())
    # exactly one 🟠 notify for the outage, aimed at the self-chat
    assert _wait(
        lambda: any(
            entry[1] == ME_JID and "LLM provider is unreachable" in entry[2]
            for entry in dead_bot.waha.sent
        )
    )
    outage_notes = [
        entry for entry in dead_bot.waha.sent if "LLM provider is unreachable" in entry[2]
    ]
    assert len(outage_notes) == 1
    # the chat itself got nothing — no reply could be produced
    assert all(entry[1] != CHAT_ID for entry in dead_bot.waha.sent)
    # redelivery is still possible: seen markers were dropped
    assert not seen_recently(f"false_{CHAT_ID}_LLMDOWN0")

    live_bot = bot.rebuild()  # provider back up (same session-scoped fake)
    llm = live_bot.stack.llm
    llm.clear()
    live_bot.post(waha_event("LLMUP"))
    assert _wait(lambda: len(live_bot.waha.sent) >= 2)  # reply + 🔵 recovery
    assert any(
        entry[1] == ME_JID and "LLM provider is back" in entry[2]
        for entry in live_bot.waha.sent
    ), live_bot.waha.sent
    assert llm_healthy()
    # steady state: a further successful run sends no extra 🔵
    sent_after_recovery = len(live_bot.waha.sent)
    live_bot.post(waha_event("LLMUP2"))
    assert _wait(lambda: len(live_bot.waha.sent) > sent_after_recovery)
    assert (
        len([entry for entry in live_bot.waha.sent if "LLM provider is back" in entry[2]])
        == 1
    )


def test_llm_timeout_notifies_operator_once(bot: Bot) -> None:
    """A slow generation times out with the timeout message, not the outage.

    The provider is healthy but still generating when the per-request
    budget fires: the operator's 🟠 must say the call timed out (with
    the knob to raise), never "endpoint unreachable — replies
    paused". One notify per slow stretch (the latch), the outage flag
    stays healthy, seen markers drop for redelivery, and a success
    after the slow stretch resets the latch.
    """
    from wahabot.status import llm_healthy

    slow_bot = bot.rebuild(llm_timeout=0.3)
    llm = slow_bot.stack.llm
    llm.hang_seconds = 5.0
    try:
        slow_bot.post(waha_event("LLMSLOW0"))
        assert _wait(
            lambda: any(
                entry[1] == ME_JID and "timed out after 0s" in entry[2]
                for entry in slow_bot.waha.sent
            ),
            timeout=15,
        )
        timeout_notes = [
            entry for entry in slow_bot.waha.sent if "timed out after" in entry[2]
        ]
        assert len(timeout_notes) == 1
        # The notification is the timeout message, never the outage one.
        assert all("unreachable" not in entry[2] for entry in timeout_notes)
        # A timeout is not an outage: the health flag never flipped.
        assert llm_healthy()
        # The chat got no reply — the run died — but redelivery stays
        # possible: the seen marker was dropped.
        assert all(entry[1] != CHAT_ID for entry in slow_bot.waha.sent)
        assert not seen_recently(f"false_{CHAT_ID}_LLMSLOW0")
        # Second slow call while the latch is held: quiet, not a new 🟠.
        slow_bot.post(waha_event("LLMSLOW1"))
        time.sleep(2)
        assert len([e for e in slow_bot.waha.sent if "timed out after" in e[2]]) == 1

        # Provider answers fast again: the reply lands and the next
        # timeout (if one ever happens) would notify afresh.
        llm.hang_seconds = 0.0
        fast_bot = bot.rebuild(llm_timeout=0.3)
        fast_llm = fast_bot.stack.llm
        fast_llm.clear()
        fast_bot.post(waha_event("LLMFAST"))
        assert _wait(lambda: len(fast_bot.waha.sent) >= 1)
        assert any(entry[1] == CHAT_ID for entry in fast_bot.waha.sent)
    finally:
        llm.hang_seconds = 0.0


def test_llm_proxy_502_notifies_operator(bot: Bot) -> None:
    """A dead proxy's 502 classifies as endpoint-down: notify, no spam.

    The connection-refused shape is covered by the outage test; this
    one pins the proxy shape — an HTTP 502 must reach the same 🟠
    branch, never the per-message traceback path.
    """
    llm = bot.stack.llm
    llm.fail_status = 502
    try:
        bot.post(waha_event("LLM502"))
        assert _wait(
            lambda: any(
                entry[1] == ME_JID and "LLM provider is unreachable" in entry[2]
                for entry in bot.waha.sent
            )
        )
        from wahabot.status import llm_healthy as healthy

        assert not healthy()
        assert all(entry[1] != CHAT_ID for entry in bot.waha.sent)
        # 4xx is a bug, not an outage: the traceback path, no notify
        seen_notify = len(
            [e for e in bot.waha.sent if "LLM provider is unreachable" in e[2]]
        )
        llm.fail_status = 400
        bot.post(waha_event("LLM400"))
        assert _wait(lambda: len(llm.requests) >= 2)
        assert (
            len([e for e in bot.waha.sent if "LLM provider is unreachable" in e[2]])
            == seen_notify
        )
    finally:
        llm.fail_status = 0


def test_persistent_memory_roundtrip_wire(bot: Bot) -> None:
    llm = bot.stack.llm
    PC = "5553333333-5553333333@g.us"

    def pc_event(
        body: str, mid: str, from_me: bool = False, chat: str = PC
    ) -> dict[str, Any]:
        return chat_event(body=body, mid=mid, chat=chat, from_me=from_me)

    def persisted_contents() -> list[str]:
        memory = load_memory(bot.settings.data_dir, SESSION, PC)
        return [] if memory is None else [str(m.content) for m in memory.get_all()]

    # Delivered reply persists.
    bot.post(pc_event("kai pc first", "PC1"))
    assert len(bot.waha.sent) == 1
    mem_path = memory_file(bot.settings.data_dir, SESSION, PC)
    assert mem_path.exists()
    envelope = json.loads(mem_path.read_text())
    assert (
        envelope["version"] == 1
        and envelope["session"] == SESSION
        and envelope["chat_id"] == PC
    )
    assert "smoke reply one" in persisted_contents()

    # Restore-on-miss.
    from wahabot.core.runs import contexts as ctxs

    ctxs.pop((SESSION, PC), None)
    assert (SESSION, PC) not in ctxs
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(pc_event("kai pc second", "PC2"))
    assert (SESSION, PC) in ctxs
    assert len(llm.requests) >= 1
    first_request_users = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("pc first" in text for text in first_request_users)

    # fromMe fold persists.
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(pc_event("operator typed pc", "PCOWN", from_me=True))
    assert len(bot.waha.sent) == 0
    assert any(
        not c.startswith("[[") and "<operator>" in c and "operator typed pc" in c
        for c in persisted_contents()
    )

    # Reaction fold persists.
    llm.requests.clear()
    bot.post(
        {
            "id": "evt-smoke-pc-reaction",
            "timestamp": int(time.time()),
            "event": "message.reaction",
            "session": SESSION,
            "me": None,
            "payload": {
                "id": "evt-smoke-pc-reaction-payload",
                "from": PC,
                "participant": "491555000001@c.us",
                "fromMe": False,
                "reaction": {"text": "👍", "messageId": f"true_{PC}_SMOKEREPLY"},
            },
        }
    )
    assert len(llm.requests) == 0
    assert any(
        "reacted" in c and 'to: you "smoke reply one"' in c for c in persisted_contents()
    )


def test_persistence_corrupt_and_forget(bot: Bot) -> None:
    llm = bot.stack.llm
    PC = "5553333333-5553333333@g.us"

    def pc_event(body: str, mid: str, chat: str = PC) -> dict[str, Any]:
        return chat_event(body=body, mid=mid, chat=chat)

    llm.override = None
    bot.post(pc_event("kai pc first", "PC1"))
    mem_path = memory_file(bot.settings.data_dir, SESSION, PC)
    llm.requests.clear()
    bot.waha.sent.clear()
    ctxs = handlers_contexts
    ctxs.pop((SESSION, PC), None)
    mem_path.write_text("{ truncated json")
    bot.post(pc_event("kai pc third", "PC3"))
    assert len(bot.waha.sent) == 1
    assert Path(str(mem_path) + ".bad").exists()
    assert mem_path.exists()

    # forget: context dropped, file deleted, next turn blank.
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(build_forget_event(SESSION, PC))
    assert (SESSION, PC) not in ctxs
    assert not mem_path.exists()
    llm.requests.clear()
    bot.waha.sent.clear()
    bot.post(pc_event("kai pc fourth", "PC4"))
    assert len(bot.waha.sent) == 1
    post_forget_users = [
        str(m.get("content", ""))
        for request in llm.requests
        for m in request["messages"]
        if m.get("role") == "user"
    ]
    assert not any("pc first" in t or "pc second" in t for t in post_forget_users)


def test_memory_persist_disabled(bot: Bot) -> None:
    NPC = "8887777777-8887777777@g.us"
    no_persist_bot = bot.rebuild(memory_persist=False)

    def npc_event(body: str, chat: str = NPC) -> dict[str, Any]:
        ev = waha_event("NPC1")
        ev["id"] = "evt-smoke-npc-1"
        ev["payload"]["id"] = f"false_{chat}_NPC1"
        ev["payload"]["_data"]["id"] = {"_serialized": f"false_{chat}_NPC1"}
        ev["payload"]["from"] = chat
        ev["payload"]["body"] = body
        return ev

    llm = no_persist_bot.stack.llm
    with unittest.mock.patch(
        "wahabot.settings.get_settings", return_value=no_persist_bot.settings
    ):
        no_persist_bot.post(npc_event("kai npc turn"))
    assert _wait(lambda: len(llm.requests) >= 1)
    assert not memory_file(no_persist_bot.settings.data_dir, SESSION, NPC).exists()


def test_self_chat_command(bot: Bot) -> None:
    llm = bot.stack.llm
    llm.clear()

    def self_event(body: str, mid: str) -> dict[str, Any]:
        ev = waha_event(mid)
        ev["id"] = f"evt-smoke-self-{mid}"
        ev["payload"]["id"] = f"true_{ME_JID}_{mid}"
        ev["payload"]["_data"]["id"] = {"_serialized": f"true_{ME_JID}_{mid}"}
        ev["payload"]["from"] = ME_JID
        ev["payload"]["fromMe"] = True
        ev["payload"]["body"] = body
        return ev

    bot.post(self_event("kai do the thing", "SELF1"))
    assert _wait(lambda: len(llm.requests) >= 1)
    self_turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("[operator command] do the thing" in t for t in self_turns)
    assert len(llm.requests[0]["messages"]) <= 3
    # The scripted send_message omits `chat`: an operator run has no
    # default chat, so the tool refuses instead of posting to the
    # synthetic "operator" id (which WAHA answered with a 500), and the
    # final text reaches the self-chat.
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert all(sent[1] != "operator" for sent in bot.waha.sent)
    assert bot.waha.sent == [(SESSION, ME_JID, "smoke final answer", None)]

    # Research-only answer: final text quote-replied to the self-chat.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = SecondResponse
    bot.post(self_event("kai just answer me", "SELF3"))
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert bot.waha.sent == [(SESSION, ME_JID, "smoke final answer", None)]

    # Non-matching fromMe in the self-chat wakes nothing.
    llm.clear()
    bot.waha.sent.clear()
    bot.post(self_event("just a note to myself", "SELF2"))
    time.sleep(0.5)
    assert len(llm.requests) == 0
    assert (SESSION, ME_JID) not in handlers_contexts

    # Echo-tracked self-chat message never re-triggers.
    llm.clear()
    echo_mid = f"true_{ME_JID}_SELFECHO"
    remember_self_echo(echo_mid)
    bot.post(self_event("kai run again", "SELFECHO"))
    time.sleep(0.5)
    assert len(llm.requests) == 0

    # LID-linked self-chat: WAHA reports from=<phone>@c.us, to=<lid>@lid.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = None
    lid_event = self_event("kai lid linked command", "SELF4")
    lid_event["payload"]["to"] = "491555000000@lid"
    bot.post(lid_event)
    assert _wait(lambda: len(llm.requests) >= 1)
    lid_turns = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("[operator command] lid linked command" in t for t in lid_turns)

    # A command that delivers via a tool still acknowledges in the
    # self-chat — the console is never silent about where it went.
    llm.clear()
    bot.waha.sent.clear()
    llm.override = tool_call_response(
        "send_message",
        {"text": "smoke reply one", "chat": CHAT_ID},
        call_id="call_cmd_self5",
        response_id="chatcmpl-smoke-cmd-self5",
        created=1788525841,
    )
    bot.post(self_event(f"kai send the plan to the group {CHAT_ID}", "SELF5"))
    assert _wait(lambda: len(bot.waha.sent) >= 2)
    assert (SESSION, CHAT_ID, "smoke reply one", None) in bot.waha.sent
    notice = [s for s in bot.waha.sent if s[1] == ME_JID]
    assert len(notice) == 1
    assert notice[0][2] == "Delivered to smoke group."  # the name, not the JID
