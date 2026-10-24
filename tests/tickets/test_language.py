from wahabot.core.tickets.language import detect_language


def test_detects_common_customer_languages() -> None:
    assert detect_language("Hola, necesito ayuda con mi pedido") == "es"
    assert detect_language("Bonjour, j'ai besoin d'aide pour ma commande") == "fr"
    assert detect_language("Hallo, ich brauche Hilfe bei meiner Bestellung") == "de"
    assert detect_language("Olá, preciso de ajuda com o meu pedido") == "pt"
    assert detect_language("Ciao, ho bisogno di aiuto con il mio ordine") == "it"
    assert detect_language("Hello, I need help with my order") == "en"


def test_short_or_ambiguous_messages_stay_unknown() -> None:
    assert detect_language("ok") == ""
    assert detect_language("") == ""


def test_strict_detection_and_operator_fallback() -> None:
    from wahabot.core.tickets.language import operator_language_for

    assert detect_language("hola necesito ayuda") == "es"
    assert detect_language("zzz qqq") == ""
    assert operator_language_for("", "es") == "es"
    assert operator_language_for("de", "es") == "de"
    assert operator_language_for("", "") == "en"


def test_save_command_survives_translation() -> None:
    import asyncio
    from types import SimpleNamespace

    from wahabot.core.access import SessionConfig
    from wahabot.core.tickets.replies import ReplyOutcome
    from wahabot.handlers.ticket_mixin import TicketMixin

    def translate(text: str, language: str) -> str:
        assert "Reply save" not in text
        return text.replace(
            "Should I answer this myself next time?",
            "¿Debería responderlo yo la próxima vez?",
        )

    outcome = ReplyOutcome(
        confirmation=(
            "Sent. Ticket 1 done.\nShould I answer this myself next time? Reply save 1."
        ),
        save_ticket=1,
    )
    handler = SimpleNamespace(ticket_translator=translate)
    config = SessionConfig(system_prompt="smoke", operator_language="es")
    text = asyncio.run(TicketMixin.ticket_confirmation(handler, outcome, config))  # type: ignore[arg-type]
    assert "Reply save 1." in text
    assert "¿Debería responderlo yo la próxima vez?" in text
    assert "guardar" not in text
