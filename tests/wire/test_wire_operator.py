import asyncio
import base64
import time
import unittest.mock
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from tests.conftest import Bot
from tests.harness import (
    CHAT_ID,
    SESSION,
    SMOKE_PNG,
    SecondResponse,
    album_events,
    chat_event,
    data_url_payload,
    image_part_count,
    is_caption_request,
    make_operator_typed,
    smoke_video_bytes,
    video_event,
    waha_event,
)
from tests.wire.support import (
    _audit_entries,
    _wait,
    requires_ffmpeg,
)
from wahabot.ai.history import sanitize_chat_history as _sanitize
from wahabot.ai.intake import IntakeLedger
from wahabot.core.persistence import memory_file
from wahabot.core.runs import contexts as handlers_contexts
from wahabot.reactions import forget_reaction_notes as _forget_notes


def test_operator_mention_wakes_bot_in_dm(bot: Bot) -> None:
    """The operator naming the bot in a DM runs it there, not the self-chat.

    A DM is not the self-chat (``to`` is the partner, not the account),
    so this is a normal wake: the turn renders as the operator's and
    the reply lands in the partner's chat.
    """
    llm = bot.stack.llm
    bot.post(waha_event())
    llm.requests.clear()
    bot.waha.sent.clear()
    partner = "491555000003@c.us"
    dm = make_operator_typed(waha_event("OPDM"), partner)
    dm["payload"]["id"] = f"false_{partner}_OPDM"
    dm["payload"]["_data"]["id"] = {"_serialized": f"false_{partner}_OPDM"}
    dm["payload"]["body"] = "kai, dime algo"
    bot.post(dm)
    assert _wait(lambda: len(llm.requests) >= 1)
    users = [
        str(m.get("content", ""))
        for m in llm.requests[0]["messages"]
        if m.get("role") == "user"
    ]
    turn = next(t for t in users if "dime algo" in t)
    assert "<operator>" in turn and ":: kai, dime algo" in turn
    assert "491555000000" not in turn
    assert _wait(lambda: len(bot.waha.sent) >= 1)
    assert bot.waha.sent[0][1] == partner


def test_fromMe_without_source_is_ignored(bot: Bot) -> None:
    """No ``source`` field: authorship unknown, so nothing folds or wakes."""
    llm = bot.stack.llm
    bot.post(waha_event())
    llm.requests.clear()
    own = make_operator_typed(waha_event("NOSRC"), CHAT_ID)
    del own["payload"]["source"]
    own["payload"]["body"] = "kai are you there"
    bot.post(own)
    time.sleep(0.3)
    assert len(llm.requests) == 0


def test_fromMe_orphan_folds_nothing(bot: Bot) -> None:
    orphan_chat = "9998887777-9998887777@g.us"
    own_orphan = waha_event()
    own_orphan["id"] = "evt-smoke-own-orphan"
    own_orphan["payload"]["id"] = f"true_{orphan_chat}_ORPHAN"
    own_orphan["payload"]["from"] = orphan_chat
    own_orphan["payload"]["fromMe"] = True
    own_orphan["payload"]["body"] = "operator text in a never-seen chat"
    bot.post(own_orphan)
    assert (SESSION, orphan_chat) not in handlers_contexts
    assert not memory_file(bot.settings.data_dir, SESSION, orphan_chat).exists()


def test_history_repair_merges_assistant_turns() -> None:
    from llama_index.core.base.llms.types import ChatMessage

    def U(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.USER, content=c)

    def A(c: str) -> ChatMessage:
        return ChatMessage(role=MessageRole.ASSISTANT, content=c)

    merged = _sanitize(
        [U("[Ana] q"), A("model reply"), A("[operator message] addendum")],
        drop_trailing_user=False,
    )
    assert len(merged) == 2
    assert "model reply" in str(merged[-1].content)
    assert "addendum" in str(merged[-1].content)

    noted = _sanitize(
        [
            U("[Ana] q"),
            A("model reply"),
            U('[reaction 👍 from Ana to your message: "model reply"]'),
            U("[Ana] next"),
        ],
        drop_trailing_user=False,
    )
    assert len(noted) == 3 and "reaction" in str(noted[-1].content)

    trailing = _sanitize(
        [
            U("[Ana] q"),
            A("model reply"),
            ChatMessage(
                role=MessageRole.USER,
                content='[reaction 👍 from Ana to your message: "model reply"]',
                additional_kwargs={"reaction_target_id": "true_C_R0"},
            ),
        ],
        drop_trailing_user=True,
    )
    assert len(trailing) == 3

    repaired = _sanitize(
        [U("[Ana] q"), A("model reply"), U("[Ana] unanswered")],
        drop_trailing_user=True,
    )
    assert len(repaired) == 2

    merged = _sanitize(
        [
            U("[Ana] q"),
            A("model reply"),
            ChatMessage(
                role=MessageRole.USER,
                content='[reaction 👍 from Ana to your message: "model reply"]',
                additional_kwargs={"reaction_target_id": "true_C_R"},
            ),
            U("[Ana] next"),
        ],
        drop_trailing_user=False,
    )
    assert merged[-1].additional_kwargs.get("reaction_target_id") == "true_C_R"


def test_tagged_reaction_removal(bot: Bot) -> None:
    # Establish the chat context first.
    bot.post(waha_event())

    async def tagged_removal() -> tuple[list[str], list[str]]:
        from llama_index.core.base.llms.types import ChatMessage

        memory = await handlers_contexts[(SESSION, CHAT_ID)].store.get("memory")
        merged_list = [
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    '[reaction 👍 from Ana to your message: "model reply"]\n[Ana] next'
                ),
                additional_kwargs={"reaction_target_id": "true_C_R2"},
            )
        ]
        rest = [
            ChatMessage(role=MessageRole.USER, content="[Ana] q"),
            ChatMessage(role=MessageRole.ASSISTANT, content="model reply"),
        ]
        await memory.aset([*rest, *merged_list])
        await _forget_notes(SESSION, CHAT_ID, "true_C_R2", bot.agent, bot.settings)
        after_merge = [str(m.content) for m in await memory.aget_all()]

        await memory.aset(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content="[reaction leading]",
                    additional_kwargs={"reaction_target_id": "true_C_R3"},
                ),
                ChatMessage(role=MessageRole.ASSISTANT, content="later"),
            ]
        )
        await _forget_notes(SESSION, CHAT_ID, "true_C_R3", bot.agent, bot.settings)
        after_leading = [str(m.content) for m in await memory.aget_all()]
        return after_merge, after_leading

    after_merge, after_leading = asyncio.run(tagged_removal())
    assert after_merge == ["[Ana] q", "model reply"]
    assert after_leading == ["[reaction leading]", "later"]


def test_no_send_tool_scaffolding_in_memory(bot: Bot) -> None:
    bot.post(waha_event())

    async def no_send_tool_group() -> bool:
        ctx = handlers_contexts[(SESSION, CHAT_ID)]
        memory = await ctx.store.get("memory")
        messages = await memory.aget_all()
        return not any(
            "send_message" in str(m.additional_kwargs.get("tool_calls", ""))
            for m in messages
        )

    assert asyncio.run(no_send_tool_group())


def test_album_vision(bot: Bot) -> None:
    llm = bot.stack.llm
    for album_event in album_events():
        bot.post(album_event)
    # Wait for the whole album run, not the first caption request: the
    # agent request lands last, and gating on an earlier request made
    # the assertions below race the run.
    assert _wait(lambda: sum(1 for r in llm.requests if not is_caption_request(r)) >= 1)

    caption_requests = [r for r in llm.requests if is_caption_request(r)]
    agent_requests = [r for r in llm.requests if not is_caption_request(r)]
    assert len(caption_requests) == 2
    assert all(image_part_count(r) == 1 for r in caption_requests)
    assert 1 <= len(agent_requests) <= 2
    album_request = agent_requests[0]
    assert image_part_count(album_request) == 2
    if len(agent_requests) > 1:
        assert image_part_count(agent_requests[-1]) == 0

    expected_b64 = base64.b64encode(SMOKE_PNG).decode()
    expected_url = f"data:image/png;base64,{expected_b64}"
    assert all(url == expected_url for url in data_url_payload(album_request))

    user_turns = [m for m in album_request["messages"] if m.get("role") == "user"]
    album_caption = (
        "two smoke-test squares on a white background; "
        "two smoke-test squares on a white background"
    )
    assert any(
        isinstance(m.get("content"), list)
        and any(
            p.get("type") == "text"
            and f"(images show: {album_caption})" in p.get("text", "")
            for p in m["content"]
        )
        for m in user_turns
    )
    assert all(
        "(images show:" not in str(m.get("content", ""))
        for r in caption_requests
        for m in r["messages"]
    )


def test_album_vision_respects_vision_max_images(bot: Bot) -> None:
    capped = bot.rebuild(vision_max_images=1)
    llm = capped.stack.llm
    for album_event in album_events():
        capped.post(album_event)
    assert _wait(lambda: sum(1 for r in llm.requests if not is_caption_request(r)) >= 1)

    caption_requests = [r for r in llm.requests if is_caption_request(r)]
    agent_requests = [r for r in llm.requests if not is_caption_request(r)]
    # The cap applies at collection time: an album image past the cap
    # can never ride the prompt, so it is neither downloaded nor
    # captioned — one caption request, one image on the agent call.
    assert len(caption_requests) == 1
    assert image_part_count(caption_requests[0]) == 1
    assert image_part_count(agent_requests[0]) == 1


def test_album_reply_types_before_sending(bot: Bot) -> None:
    """An album's text reply rides the typing pause like every path.

    The album fallback is the third delivery path; the pause was
    silently missing there while the tool and single-message paths
    had it — an album answer went out machine-fast.
    """
    typing_bot = bot.rebuild(typing_presence_min_s=0.05, typing_presence_max_s=0.1)
    llm = typing_bot.stack.llm
    llm.override = SecondResponse  # plain final-text reply, no tool call
    for album_event in album_events():
        typing_bot.post(album_event)
    assert _wait(lambda: len(typing_bot.waha.sent) >= 1)
    assert typing_bot.waha.typing_calls == [(SESSION, CHAT_ID, True)]
    assert typing_bot.waha.sent[0][2] == "smoke final answer"


def test_burst_assembles_one_turn(bot: Bot) -> None:
    """A same-sender burst becomes ONE agent turn, not one run per line.

    Three rapid messages from one DM sender must produce a single agent
    run whose user turn carries
    all three bodies joined, not three premature runs. The read
    receipt still fires per message at arrival. Every original message
    id is preserved exactly once, so quoting, audit and evidence links
    survive the merge.
    """
    llm = bot.stack.llm
    # Short windows so the test doesn't wait the production 8s; the
    # inactivity reset logic is the same at any scale.
    burst_bot = bot.rebuild(
        burst_enabled=True,
        burst_inactivity_s=0.4,
        burst_hold_cap_s=5.0,
        send_seen=False,
    )
    llm = burst_bot.stack.llm
    chat = "491555000001@c.us"
    llm.requests.clear()
    for i, line in enumerate(["there is a leak", "flat 3B", "near the panel"]):
        burst_bot.post(chat_event(body=line, mid=f"BURST{i}", chat=chat))

    def assembled_turn() -> str | None:
        """The one user turn that carries the whole burst, if it exists.

        The fake LLM is session-scoped, so a background request from an
        earlier test can still land here; the burst's own text is the
        marker that identifies this run. Assembly into a single turn is
        the guarantee: a per-message run would put each line in a
        separate user turn, so the first turn carrying one line would
        not carry the others.
        """
        for request in llm.requests:
            if is_caption_request(request):
                continue
            for message in request.get("messages", []):
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                text = content if isinstance(content, str) else str(content)
                if "there is a leak" in text:
                    return text
        return None

    assert _wait(lambda: assembled_turn() is not None)
    turn_text = assembled_turn()
    assert turn_text is not None
    assert "there is a leak" in turn_text
    assert "flat 3B" in turn_text
    assert "near the panel" in turn_text
    # One record per buffered message, in arrival order, each with its
    # own #m<n> the model can quote or react to.
    assert turn_text.count("[[") >= 3
    assert (
        turn_text.index("there is a leak")
        < turn_text.index("flat 3B")
        < turn_text.index("near the panel")
    )


def test_burst_lone_emoji_becomes_reaction(bot: Bot) -> None:
    """A burst's lone-emoji reply reacts on the anchor, not silence.

    The production trace showed a burst answering a shared reel with
    ``👍`` and the handler logging "Dropping lone emoji as silence";
    the single-message path converts a lone emoji to a ~50 % reaction,
    but the burst path called ``finish_agent_reply`` without
    ``emoji_reaction`` so it always dropped. The burst path now opts in
    — the reaction lands on the burst's last (anchor) message.
    """
    from tests.harness import chat_event as burst_chat_event

    emoji_bot = bot.rebuild(
        burst_enabled=True,
        burst_inactivity_s=0.4,
        burst_hold_cap_s=5.0,
        send_seen=False,
    )
    llm = emoji_bot.stack.llm
    llm.clear()
    emoji_response = {
        "id": "chatcmpl-smoke-emoji",
        "object": "chat.completion",
        "created": 1788525841,
        "model": "smoke-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "👍"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }

    # Answer by request shape, not the shared round counter: a
    # background request from an earlier test can still land in the
    # shared fake LLM, and that would push this run from round 1 to
    # round 2 and swap in the tool-call answer instead of the emoji.
    def always_emoji(_body: dict[str, Any]) -> dict[str, Any]:
        return emoji_response

    llm.selector = always_emoji
    chat = "491555000001@c.us"
    # Force the ~50 % reaction coin flip to the reaction side. The
    # burst flushes in the background up to burst_inactivity_s after
    # the post, so the patch must stay active through the wait — a
    # patch scoped to the post alone would exit before the coin flip.
    with unittest.mock.patch("wahabot.handlers.replies.random.random", return_value=0.0):
        emoji_bot.post(burst_chat_event(body="mira ese reel", mid="EMOJI1", chat=chat))
        assert _wait(lambda: len(emoji_bot.waha.reactions) >= 1)
    # The anchor is the burst's only/last message.
    assert (f"false_{chat}_EMOJI1", "👍") in emoji_bot.waha.reactions
    # No text was sent: the emoji became a reaction, never a message.
    assert emoji_bot.waha.sent == []
    # The audit journals the reaction, not a dropped-emoji silence.
    reactions = _audit_entries(emoji_bot, "reaction", chat)
    assert any(
        e.get("reaction") == "👍" and e.get("message_id") == f"false_{chat}_EMOJI1"
        for e in reactions
    )
    dropped = _audit_entries(emoji_bot, "silence", chat)
    assert not any("dropped_emoji" in e for e in dropped)


def test_burst_disabled_runs_per_message(bot: Bot) -> None:
    """burst_enabled=False keeps the pre-burst behavior: one run each."""
    plain_bot = bot.rebuild(burst_enabled=False)
    llm = plain_bot.stack.llm
    chat = "491555000001@c.us"
    for i in range(2):
        plain_bot.post(chat_event(body=f"line {i}", mid=f"NOBURST{i}", chat=chat))
    assert _wait(lambda: len(llm.requests) >= 2)


@requires_ffmpeg
def test_video_understanding(bot: Bot) -> None:
    llm = bot.stack.llm
    bot.waha.video_bytes = smoke_video_bytes()
    llm.requests.clear()
    with unittest.mock.patch(
        "wahabot.handlers.media.fetch_transcript", return_value="spoken words here"
    ):
        bot.post(video_event())
    assert _wait(lambda: len(llm.requests) >= 1)
    video_caption_reqs = [r for r in llm.requests if is_caption_request(r)]
    video_agent_reqs = [r for r in llm.requests if not is_caption_request(r)]
    assert len(video_caption_reqs) == 1
    if video_caption_reqs:
        assert image_part_count(video_caption_reqs[0]) == bot.settings.video_frames
    assert len(video_agent_reqs) >= 1
    if video_agent_reqs:
        video_turns = [
            m for m in video_agent_reqs[0]["messages"] if m.get("role") == "user"
        ]
        assert any(
            "(video shows: two smoke-test squares on a white background)"
            in str(m.get("content", ""))
            and '[audio: "spoken words here"]' in str(m.get("content", ""))
            for m in video_turns
        )
        assert image_part_count(video_agent_reqs[0]) == bot.settings.video_frames


@requires_ffmpeg
def test_video_frames_are_bounded_by_video_frames_not_the_image_cap(bot: Bot) -> None:
    capped = bot.rebuild(vision_max_images=1)
    llm = capped.stack.llm
    capped.waha.video_bytes = smoke_video_bytes()
    llm.requests.clear()
    with unittest.mock.patch(
        "wahabot.handlers.media.fetch_transcript", return_value="spoken words here"
    ):
        capped.post(video_event(mid="VIDCAP"))
    assert _wait(lambda: len(llm.requests) >= 1)
    caption_requests = [r for r in llm.requests if is_caption_request(r)]
    agent_requests = [r for r in llm.requests if not is_caption_request(r)]
    assert len(caption_requests) == 1
    assert image_part_count(caption_requests[0]) == capped.settings.video_frames
    assert len(agent_requests) >= 1
    assert image_part_count(agent_requests[0]) == capped.settings.video_frames


def test_folded_operator_message_is_not_ledgered(bot: Bot) -> None:
    """A folded operator DM never wakes a run, so it cannot supersede one."""
    llm = bot.stack.llm
    llm.requests.clear()
    partner = "491555000004@c.us"
    dm = make_operator_typed(waha_event("OPFOLD"), partner)
    dm["payload"]["id"] = f"false_{partner}_OPFOLD"
    dm["payload"]["_data"]["id"] = {"_serialized": f"false_{partner}_OPFOLD"}
    dm["payload"]["body"] = "internal note for the team"

    with unittest.mock.patch.object(IntakeLedger, "record") as record:
        bot.post(dm)
        time.sleep(0.3)

    record.assert_not_called()
    assert llm.requests == []
