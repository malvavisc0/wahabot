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
