"""Fact extraction and comparison for operator drafts."""

import re
from types import MappingProxyType

_NUMBER_WORDS = MappingProxyType(
    {
        # English
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "hundreds": 100,
        "thousand": 1000,
        "thousands": 1000,
        # Spanish (un/una/once/dos/mil are omitted: English homographs)
        "cero": 0,
        "uno": 1,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
        "doce": 12,
        "trece": 13,
        "catorce": 14,
        "quince": 15,
        "dieciseis": 16,
        "dieciséis": 16,
        "diecisiete": 17,
        "dieciocho": 18,
        "diecinueve": 19,
        "veinte": 20,
        "treinta": 30,
        "cuarenta": 40,
        "cincuenta": 50,
        "sesenta": 60,
        "setenta": 70,
        "ochenta": 80,
        "noventa": 90,
        "cien": 100,
        "ciento": 100,
        "doscientos": 200,
        "trescientos": 300,
        "cuatrocientos": 400,
        "quinientos": 500,
        "seiscientos": 600,
        "setecientos": 700,
        "ochocientos": 800,
        "novecientos": 900,
        # French (un/une/cent/mille omitted: English homographs)
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "sept": 7,
        "huit": 8,
        "dix": 10,
        "onze": 11,
        "douze": 12,
        "treize": 13,
        "quatorze": 14,
        "quinze": 15,
        "seize": 16,
        "vingt": 20,
        "trente": 30,
        "quarante": 40,
        "cinquante": 50,
        "soixante": 60,
        "soixante-dix": 70,
        "quatre-vingt": 80,
        "quatre-vingts": 80,
        "quatre-vingt-dix": 90,
        # German (elf omitted: English homograph)
        "zwei": 2,
        "drei": 3,
        "vier": 4,
        "fünf": 5,
        "sechs": 6,
        "sieben": 7,
        "acht": 8,
        "neun": 9,
        "zehn": 10,
        "zwölf": 12,
        "dreizehn": 13,
        "vierzehn": 14,
        "fünfzehn": 15,
        "sechzehn": 16,
        "siebzehn": 17,
        "achtzehn": 18,
        "neunzehn": 19,
        "zwanzig": 20,
        "dreißig": 30,
        "vierzig": 40,
        "fünfzig": 50,
        "sechzig": 60,
        "siebzig": 70,
        "achtzig": 80,
        "neunzig": 90,
        "hundert": 100,
        "tausend": 1000,
        # Portuguese (dos/mil omitted: English/unit homographs)
        "dois": 2,
        "três": 3,
        "quatro": 4,
        "oito": 8,
        "dez": 10,
        "doze": 12,
        "treze": 13,
        "catorze": 14,
        "dezesseis": 16,
        "dezessete": 17,
        "dezoito": 18,
        "dezenove": 19,
        "vinte": 20,
        "cem": 100,
        "cento": 100,
        "duzentos": 200,
        "trezentos": 300,
        "quatrocentos": 400,
        "quinhentos": 500,
        "seiscentos": 600,
        "setecentos": 700,
        "oitocentos": 800,
        "novecentos": 900,
        # Italian (due/otto/mille omitted: English homographs and names)
        "tre": 3,
        "quattro": 4,
        "cinque": 5,
        "sette": 7,
        "dieci": 10,
        "undici": 11,
        "dodici": 12,
        "tredici": 13,
        "quattordici": 14,
        "quindici": 15,
        "sedici": 16,
        "diciassette": 17,
        "diciotto": 18,
        "diciannove": 19,
        "venti": 20,
        "trenta": 30,
        "quaranta": 40,
        "cinquanta": 50,
        "sessanta": 60,
        "settanta": 70,
        "ottanta": 80,
        "novanta": 90,
    }
)
_NUMBER_CONNECTORS = frozenset({"and", "y", "et", "und", "e"})
_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
_CURRENCY_SYMBOLS = MappingProxyType({"€": "eur", "$": "usd", "£": "gbp"})

_MONEY = "eur|usd|gbp"
_DURATIONS = "days?|weeks?|months?|years?|hours?|hrs?|minutes?|mins?|seconds?"
_UNITS = f"{_DURATIONS}|{_MONEY}|kg|grams?|g|km|metres?|meters?"
_CRITICAL_PATTERNS = (
    re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{_MONEY})\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:[.,]\d+)?\s*%"),
)
_FACT_PATTERNS = (
    re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{_UNITS})\b", re.IGNORECASE),
    *_CRITICAL_PATTERNS,
    re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b"),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}(?:am|pm)\b"),
    re.compile(r"\b(?:order|ticket|invoice|ref(?:erence)?)\s*#?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:[.,]\d+)?\b"),
)
_NUMBER_RANGE = re.compile(r"(\d+)\s*(?:-|to|until)\s*(\d+)")
_UNIT_CANON = MappingProxyType(
    {
        "euro": "eur",
        "euros": "eur",
        "dollar": "usd",
        "dollars": "usd",
        "dólar": "usd",
        "dólares": "usd",
        "dolar": "usd",
        "dolares": "usd",
        "pound": "gbp",
        "pounds": "gbp",
        "libra": "gbp",
        "libras": "gbp",
        "day": "days",
        "día": "days",
        "días": "days",
        "dia": "days",
        "dias": "days",
        "jour": "days",
        "jours": "days",
        "tag": "days",
        "tage": "days",
        "giorno": "days",
        "giorni": "days",
        "week": "weeks",
        "semana": "weeks",
        "semanas": "weeks",
        "semaine": "weeks",
        "semaines": "weeks",
        "woche": "weeks",
        "wochen": "weeks",
        "month": "months",
        "mes": "months",
        "meses": "months",
        "mois": "months",
        "monat": "months",
        "monate": "months",
        "year": "years",
        "año": "years",
        "años": "years",
        "ano": "years",
        "anos": "years",
        "ans": "years",
        "jahr": "years",
        "jahre": "years",
        "hour": "hours",
        "hora": "hours",
        "horas": "hours",
        "heure": "hours",
        "heures": "hours",
        "stunde": "hours",
        "stunden": "hours",
        "minute": "minutes",
        "minuto": "minutes",
        "minutos": "minutes",
        "minuten": "minutes",
        "second": "seconds",
        "segundo": "seconds",
        "segundos": "seconds",
        "seconde": "seconds",
        "sekunde": "seconds",
        "sekunden": "seconds",
        "hr": "hours",
        "hrs": "hours",
        "min": "minutes",
        "mins": "minutes",
        "gram": "grams",
        "gramo": "grams",
        "gramos": "grams",
        "gramme": "grams",
        "gramm": "grams",
        "metre": "metres",
        "meter": "metres",
        "metro": "metres",
        "metros": "metres",
    }
)


def compose_number(values: list[int]) -> int:
    """Compose a run of number words the way English does."""
    total = 0
    current = 0
    for value in values:
        if value == 100:
            current = (current or 1) * 100
        elif value == 1000:
            total += (current or 1) * 1000
            current = 0
        else:
            current += value
    return total + current


def word_value(word: str) -> int | None:
    """The value of one number word, including hyphenated compounds."""
    whole = _NUMBER_WORDS.get(word)
    if whole is not None:
        return whole
    if "-" not in word:
        return None
    parts = [part for part in word.split("-") if part]
    if len(parts) < 2 or any(part not in _NUMBER_WORDS for part in parts):
        return None
    return compose_number([_NUMBER_WORDS[part] for part in parts])


def replace_number_words(text: str) -> str:
    """Replace every run of number words with its composed digits."""
    matches = list(_WORD_RE.finditer(text))
    pieces: list[str] = []
    last = 0
    index = 0
    while index < len(matches):
        if word_value(matches[index].group()) is None:
            index += 1
            continue
        values: list[int] = []
        cursor = index
        while cursor < len(matches):
            value = word_value(matches[cursor].group())
            if value is not None:
                values.append(value)
                cursor += 1
                continue
            connector = matches[cursor].group().casefold()
            if (
                values
                and values[-1] >= 20
                and connector in _NUMBER_CONNECTORS
                and cursor + 1 < len(matches)
                and word_value(matches[cursor + 1].group()) is not None
            ):
                cursor += 1
                continue
            break
        pieces.append(text[last : matches[index].start()])
        pieces.append(str(compose_number(values)))
        last = matches[cursor - 1].end()
        index = cursor
    pieces.append(text[last:])
    return "".join(pieces)


def canonical_currencies(text: str) -> str:
    """Turn currency symbols into canonical unit words."""

    def after(match: re.Match[str]) -> str:
        return f"{match.group(2)} {_CURRENCY_SYMBOLS[match.group(1)]}"

    def before(match: re.Match[str]) -> str:
        return f"{match.group(1)} {_CURRENCY_SYMBOLS[match.group(2)]}"

    text = re.sub(r"([€$£])\s*(\d+(?:[.,]\d+)?)", after, text)
    return re.sub(r"(\d+(?:[.,]\d+)?)\s*([€$£])", before, text)


def normalize_text(text: str) -> str:
    """Normalize number words, ranges, units and currency for comparison."""
    lowered = text.casefold()
    lowered = replace_number_words(lowered)
    lowered = _NUMBER_RANGE.sub(r"\1-\2", lowered)
    for word, canon in _UNIT_CANON.items():
        lowered = re.sub(rf"\b{re.escape(word)}\b", canon, lowered)
    lowered = re.sub(r"(\d)\s+(am|pm)\b", r"\1\2", lowered)
    lowered = re.sub(r"(\d)\s+%", r"\1%", lowered)
    lowered = re.sub(r"\b(eur|usd|gbp)\s*(\d+(?:[.,]\d+)?)", r"\2 \1", lowered)
    lowered = canonical_currencies(lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def fact_tokens(text: str) -> set[str]:
    """Every amount, date, time, duration and reference in *text*."""
    normalized = normalize_text(text)
    found: set[str] = set()
    for pattern in _FACT_PATTERNS:
        found.update(match.group(0).strip() for match in pattern.finditer(normalized))
    return found


def critical_tokens(text: str) -> set[str]:
    """Amounts and percentages that must survive the rewrite."""
    normalized = normalize_text(text)
    found: set[str] = set()
    for pattern in _CRITICAL_PATTERNS:
        found.update(match.group(0).strip() for match in pattern.finditer(normalized))
    return found


def mismatch_tokens(draft: str, operator_text: str) -> list[str]:
    """Facts in the draft that the operator never wrote."""
    missing = fact_tokens(draft) - fact_tokens(operator_text)
    return sorted(token for token in missing if not subsumed(token, missing))


def omitted_tokens(draft: str, operator_text: str) -> list[str]:
    """Critical facts the operator wrote that the draft dropped."""
    missing = critical_tokens(operator_text) - critical_tokens(draft)
    return sorted(token for token in missing if not subsumed(token, missing))


def subsumed(token: str, tokens: set[str]) -> bool:
    """Whether a broader token in the same set already covers *token*."""
    pattern = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")
    return any(token != other and pattern.search(other) for other in tokens)
