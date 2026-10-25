import json
from pathlib import Path
from typing import Any

from tests.conftest import Bot
from tests.harness import ME_JID, tool_call_response, waha_event
from tests.skills.conftest import (
    SECRET_VALUE,
    SENDER,
    TEAM_GROUP,
    TEST_KEY,
    write_skill,
)
from tests.wire.support import _audit_entries, _wait
from wahabot.commands import build_command_event
from wahabot.core.skills.loader import import_skill
from wahabot.core.skills.secrets import SecretStore
from wahabot.core.skills.store import SkillStore
from wahabot.core.tickets.models import TicketUpdate

FINAL_TEXT = {
    "id": "chatcmpl-skills-final",
    "object": "chat.completion",
    "created": 1788525832,
    "model": "smoke-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "I found one record."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


def customer_event(mid: str, body: str) -> dict[str, Any]:
    event = waha_event(mid)
    payload = event["payload"]
    payload["id"] = f"false_{SENDER}_{mid}"
    payload["_data"]["id"] = {"_serialized": f"false_{SENDER}_{mid}"}
    payload["from"] = SENDER
    payload["participant"] = SENDER
    payload["body"] = body
    payload["_data"]["notifyName"] = "Smoke Sender"
    return event


def operator_event(mid: str, body: str, *, quote: str = "") -> dict[str, Any]:
    event = waha_event(mid)
    payload = event["payload"]
    payload["id"] = f"true_{ME_JID}_{mid}"
    payload["_data"]["id"] = {"_serialized": f"true_{ME_JID}_{mid}"}
    payload["from"] = ME_JID
    payload["to"] = ME_JID
    payload["fromMe"] = True
    payload["body"] = body
    payload["source"] = "app"
    if quote:
        payload["replyTo"] = {"id": quote}
    return event


def tool_selector(name: str, arguments: dict[str, Any]):
    def select(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        entries = messages if isinstance(messages, list) else []
        if any(str(message.get("role")) == "tool" for message in entries):
            return FINAL_TEXT
        return tool_call_response(name, arguments, call_id=f"call-{name}")

    return select


def install_skill(
    bot: Bot,
    tmp_path: Path,
    base_url: str,
    *,
    name: str = "customer-records",
    customer_chats: str = "read",
    writes_need_operator: bool = True,
) -> SkillStore:
    store = SkillStore(bot.database.path)
    secrets = SecretStore(bot.database.path, bot.settings.secret_key or TEST_KEY)
    secrets.put("SMOKE_KEY", SECRET_VALUE)
    folder = write_skill(
        tmp_path,
        name,
        base_url,
        customer_chats=customer_chats,
        writes_need_operator=writes_need_operator,
    )
    import_skill(folder, bot.settings.data_dir / "skills", store)
    store.set_enabled(name, True)
    return store


def test_a_customer_read_lookup_is_logged_on_the_ticket_and_audit(
    bot: Bot, tmp_path: Path, fake_rest
) -> None:
    install_skill(bot, tmp_path, fake_rest.base)
    running = bot.restart()
    running.stack.llm.clear()
    running.stack.llm.selector = tool_selector(
        "http_request",
        {
            "skill": "customer-records",
            "method": "GET",
            "path": "/records",
            "query": {"phone": "{sender.phone}"},
        },
    )

    running.post(customer_event("SKILLREAD", "look up my record"))

    assert _wait(lambda: any(entry[1] == SENDER for entry in running.waha.sent))
    ticket = running.handlers.ticket_store.chat_tickets(SENDER, limit=1)[0]
    calls = SkillStore(running.database.path).calls("customer-records")
    assert calls and calls[0].ticket_id == ticket.id and calls[0].outcome == "ok"
    assert any(
        event.kind == "skill_call"
        for event in running.handlers.ticket_store.events(ticket.id)
    )
    assert _audit_entries(running, "skill_call", SENDER)
    assert all(
        SECRET_VALUE not in json.dumps(request) for request in running.stack.llm.requests
    )
    assert all(SECRET_VALUE not in entry[2] for entry in running.waha.sent)


def test_a_customer_write_waits_for_yes_and_runs_once(
    bot: Bot, tmp_path: Path, fake_rest
) -> None:
    install_skill(bot, tmp_path, fake_rest.base, customer_chats="read_write")
    running = bot.restart()
    running.stack.llm.clear()
    running.stack.llm.selector = tool_selector(
        "http_request",
        {
            "skill": "customer-records",
            "method": "POST",
            "path": "/leads",
            "body": {"name": "Fictional Smoke", "phone": "{sender.phone}"},
        },
    )

    running.post(customer_event("SKILLWRITE", "create a lead for me"))

    assert _wait(lambda: any("Reply yes." in entry[2] for entry in running.waha.sent))
    assert fake_rest.lead_count() == 0
    confirmation = next(entry for entry in running.waha.sent if "Reply yes." in entry[2])
    confirmation_id = f"true_{ME_JID}_SENT{running.waha.sent.index(confirmation) + 1}"
    store = SkillStore(running.database.path)
    pending = store.latest_pending()
    assert pending is not None and pending.state == "pending"

    running.post(operator_event("SKILLYES", "yes", quote=confirmation_id))

    assert _wait(lambda: fake_rest.lead_count() == 1)
    assert _wait(
        lambda: SkillStore(running.database.path).request(pending.id).state == "done"
    )
    running.post(operator_event("SKILLYES2", "yes", quote=confirmation_id))
    assert fake_rest.lead_count() == 1


def test_an_operator_requested_write_executes_directly(
    bot: Bot, tmp_path: Path, fake_rest
) -> None:
    install_skill(bot, tmp_path, fake_rest.base, customer_chats="read_write")
    running = bot.restart()
    running.stack.llm.clear()
    running.stack.llm.selector = tool_selector(
        "http_request",
        {
            "skill": "customer-records",
            "method": "POST",
            "path": "/leads",
            "body": {"name": "Fictional Lead"},
        },
    )

    running.post(build_command_event(running.settings.session, "create the lead"))

    assert _wait(lambda: fake_rest.lead_count() == 1)


def test_a_team_group_write_still_needs_the_operator_yes(
    bot: Bot, tmp_path: Path, fake_rest
) -> None:
    install_skill(
        bot,
        tmp_path,
        fake_rest.base,
        customer_chats="read_write",
        writes_need_operator=False,
    )
    running = bot.restart()
    config_path = running.settings.access_config
    config = json.loads(config_path.read_text())
    config["team_groups"] = [TEAM_GROUP]
    config_path.write_text(json.dumps(config))
    running.stack.llm.clear()
    running.stack.llm.selector = tool_selector(
        "http_request",
        {
            "skill": "customer-records",
            "method": "POST",
            "path": "/leads",
            "body": {"name": "Team Lead"},
        },
    )
    event = customer_event("TEAMWRITE", "kai create a lead")
    event["payload"]["id"] = f"false_{TEAM_GROUP}_TEAMWRITE"
    event["payload"]["_data"]["id"] = {"_serialized": f"false_{TEAM_GROUP}_TEAMWRITE"}
    event["payload"]["from"] = TEAM_GROUP
    event["payload"]["participant"] = SENDER
    event["payload"]["body"] = "kai create a lead"

    running.post(event)

    assert _wait(lambda: any("Reply yes." in entry[2] for entry in running.waha.sent))
    assert fake_rest.lead_count() == 0


def test_a_done_ticket_reaches_the_event_endpoint(
    bot: Bot, tmp_path: Path, fake_rest
) -> None:
    store = SkillStore(bot.database.path)
    secrets = SecretStore(bot.database.path, bot.settings.secret_key or TEST_KEY)
    secrets.put("SIGNING_KEY", "fictional-signing-secret")
    folder = write_skill(
        tmp_path,
        "send-to-url",
        fake_rest.base,
        auth="{type: none}",
        customer_chats="none",
        sign_events=True,
        signing_secret="SIGNING_KEY",
        events={"done": True},
    )
    import_skill(folder, bot.settings.data_dir / "skills", store)
    store.set_enabled("send-to-url", True)
    running = bot.restart()
    store = SkillStore(running.database.path)
    store.set_enabled("send-to-url", True)
    contact = running.handlers.ticket_store.upsert_contact(SENDER, phone=SENDER)
    ticket = running.handlers.ticket_store.create_ticket(contact.id, SENDER)

    running.handlers.ticket_store.update_ticket(
        ticket.id, TicketUpdate(state="done", done_by="assistant")
    )

    assert _wait(lambda: len(fake_rest.hooks) == 1)
    assert fake_rest.hooks[0]["signature"].startswith("sha256=")


def test_what_can_you_do_lists_active_skills_without_urls_or_secrets(
    bot: Bot, tmp_path: Path, fake_rest
) -> None:
    install_skill(bot, tmp_path, fake_rest.base)
    running = bot.restart()

    running.post(operator_event("CAPABILITIES", "what can you do?"))

    assert _wait(lambda: any(entry[1] == ME_JID for entry in running.waha.sent))
    reply = running.waha.sent[-1][2]
    assert "customer-records" in reply
    assert fake_rest.base not in reply
    assert SECRET_VALUE not in reply


def test_no_skill_configured_still_answers_and_opens_a_ticket(bot: Bot) -> None:
    bot.stack.llm.clear()
    bot.stack.llm.selector = tool_selector(
        "send_message", {"text": "plain answer", "reason": "no skill"}
    )

    bot.post(customer_event("NOSKILL", "hello"))

    assert _wait(lambda: any(entry[1] == SENDER for entry in bot.waha.sent))
    assert bot.handlers.ticket_store.chat_tickets(SENDER, limit=1)


def test_link_ticket_user_is_operator_only(bot: Bot, tmp_path: Path) -> None:
    running = bot.restart()
    store = running.handlers.ticket_store
    contact = store.upsert_contact(SENDER, phone=SENDER)
    ticket = store.create_ticket(contact.id, SENDER)
    running.stack.llm.clear()
    running.stack.llm.selector = tool_selector(
        "link_ticket_user", {"user_ref": "fictional-user-88", "number": ticket.number}
    )

    running.post(customer_event("LINKCUSTOMER", "link me"))
    _wait(lambda: any(entry[1] == SENDER for entry in running.waha.sent))
    assert store.ticket(ticket.id).external_user_ref == ""
    assert store.contact(SENDER).external_user_ref == ""

    running.post(build_command_event(running.settings.session, "link this ticket"))
    assert _wait(lambda: store.ticket(ticket.id).external_user_ref == "fictional-user-88")
    assert store.contact(SENDER).external_user_ref == "fictional-user-88"
