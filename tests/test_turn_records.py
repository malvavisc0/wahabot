"""Tests for the WAHA-aware record builder (``wahabot.ai.turn_records``)."""

from __future__ import annotations

from wahabot.ai.messages import OPERATOR_PAYLOAD_FLAG
from wahabot.ai.turn_records import build_record, record_kind, speaker_token
from wahabot.core.models import WahaEvent


def _event(*, body: str = "", kind: str = "chat", operator: bool = False) -> WahaEvent:
    payload: dict[str, object] = {"from": "491555000009@g.us", "body": body}
    if operator:
        payload[OPERATOR_PAYLOAD_FLAG] = True
    return WahaEvent(
        id="e1",
        timestamp=1,
        event="message",
        session="smoke",
        me={},
        payload={"_data": {"type": kind}, **payload},
    )


def test_record_kind_marks_a_bare_url_as_link() -> None:
    assert record_kind(_event(body="https://example.com/a")) == "link"
    assert record_kind(_event(body="look https://example.com/a")) == "text"
    assert record_kind(_event(body="hola")) == "text"
    assert record_kind(_event(body="", kind="image")) == "image"
    assert record_kind(_event(body="", kind="ptt")) == "audio"


def test_speaker_token_prefers_operator_then_name_then_ordinal() -> None:
    assert speaker_token(_event(operator=True), name="Ana", ordinal=2) == "<operator>"
    assert speaker_token(_event(), name="Ana") == "Ana"
    assert speaker_token(_event(), ordinal=3) == "<member-3>"
    assert speaker_token(_event()) == "<member-?>"


def test_build_record_assembles_flags_in_grammar_order() -> None:
    line = build_record(
        _event(body="confirma que es la de enero"),
        speaker="Ana",
        serial=108,
        time_label="21:10",
        media_slots=("image#1",),
        addressed=True,
        links=("quote=#m104",),
        payloads=('quoted: "llegó la factura de enero"',),
    )
    head = line.splitlines()[0]
    assert head.startswith("[[21:10 Ana #m108]] text image#1 addressed quote=#m104 :: ")
    assert head.endswith("confirma que es la de enero")
    assert line.splitlines()[1] == '  · quoted: "llegó la factura de enero"'


def test_build_record_overrides_the_body_and_drops_an_empty_tail() -> None:
    assert (
        build_record(
            _event(body="ignored"),
            speaker="<operator>",
            serial=9,
            time_label="09:00",
            body="",
        )
        == "[[09:00 <operator> #m9]] text"
    )
    assert build_record(
        _event(body="ignored"), speaker="Ana", serial=9, time_label="09:00", body="hi"
    ).endswith(":: hi")


def test_build_record_links_the_kind_to_the_event() -> None:
    line = build_record(
        _event(body="https://example.com/reel/x"),
        speaker="Sofi",
        serial=41,
        time_label="09:41",
    )
    assert line == "[[09:41 Sofi #m41]] link :: https://example.com/reel/x"
