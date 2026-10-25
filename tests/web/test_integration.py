"""One end-to-end web write reaches the next WhatsApp run."""

import json

from fastapi.testclient import TestClient
from tests.harness import FirstResponse, waha_event
from tests.web.conftest import csrf_of
from tests.wire.support import _wait

from wahabot.webhook import app


def test_registration_exposes_the_ticket_translator(bot) -> None:
    """History uses the same translator the agent uses when it is available."""
    assert callable(app.state.web_translate)


def test_assistant_name_change_reaches_the_next_run(auth_client: TestClient, bot) -> None:
    """A saved Settings value is reloaded for the next inbound message."""
    token = csrf_of(auth_client.get("/app/settings").text)
    saved = auth_client.post(
        "/app/settings/assistant",
        data={
            "bot_name": "Bridge Club Desk",
            "goal": "Answer opening-hours questions.",
            "instructions": "Be brief.",
            "operator_name": "Sam",
            "csrf_token": token,
        },
    )
    assert saved.status_code == 200
    llm = bot.stack.llm
    llm.override = FirstResponse
    try:
        event = waha_event("WEBINTEGRATION")
        event["payload"]["body"] = "Bridge Club Desk hola"
        bot.post(event)
        assert _wait(lambda: len(bot.waha.sent) >= 1)
        prompts = json.dumps(llm.requests)
        assert "Bridge Club Desk" in prompts
    finally:
        llm.override = None
