from wahabot.ai.observability import chat_trace_attributes, trace_metadata


def test_trace_metadata_hashes_ids_and_keeps_serial_tokens() -> None:
    metadata = trace_metadata(
        message_ids=["false_491555000001@c.us_1"],
        serial_tokens=["#m1", "#m2"],
        outbound_id="true_491555000001@c.us_2",
        reply_to="false_491555000001@c.us_1",
    )

    assert metadata["wahabot.serial_tokens"] == "#m1,#m2"
    assert "491555000001" not in metadata["wahabot.message_ids"]
    assert "491555000001" not in metadata["wahabot.outbound_id"]
    assert "491555000001" not in metadata["wahabot.reply_to"]
    assert metadata["wahabot.message_ids"]
    assert metadata["wahabot.outbound_id"]
    assert metadata["wahabot.reply_to"]


def test_trace_attributes_accept_ids_without_a_waha_client() -> None:
    with chat_trace_attributes(
        "491555000001@c.us",
        message_ids=["false_491555000001@c.us_1"],
        serial_tokens=["#m1"],
        outbound_id="true_491555000001@c.us_2",
        reply_to="false_491555000001@c.us_1",
    ):
        pass
