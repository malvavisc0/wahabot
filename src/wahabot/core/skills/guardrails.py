"""Code-enforced skill guardrails: URL lock, address checks and sender scope."""

import ipaddress
import posixpath
import re
import socket
from dataclasses import dataclass
from urllib.parse import SplitResult, unquote, urlencode, urlsplit, urlunsplit

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.models import READ_METHODS, SkillConfig, SkillRuntime
from wahabot.core.tickets.models import Contact, Ticket

_PHONE = re.compile(r"\+\d[\d\s().-]{6,}\d")
_JID = re.compile(r"\b\d+(?:-\d+)?@[a-z.]+", re.IGNORECASE)
_METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254"})


class EndpointRefused(SkillError):
    """A request left the skill's fence or reached a forbidden address."""


@dataclass(frozen=True, slots=True)
class PinnedAddress:
    """The address one request is bound to after DNS resolution."""

    host: str
    port: int
    ip: str


def is_write_method(method: str) -> bool:
    """Reads are GET/HEAD/OPTIONS; everything else is a write."""
    return method.strip().upper() not in READ_METHODS


def join_skill_url(
    config: SkillConfig, path: str, query: dict[str, str] | None = None
) -> str:
    """Build a request URL locked to one configured base path."""
    decoded, embedded = _relative_path(path)
    urls = config.effective_base_urls
    if not urls:
        raise EndpointRefused("the skill has no base URL")
    base = urlsplit(urls[0])
    base_path = posixpath.normpath(base.path or "/").rstrip("/")
    relative = decoded.lstrip("/")
    joined = posixpath.normpath(f"{base_path}/{relative}") if relative else base_path
    _require_under_base(joined, base_path)
    pieces = [
        part for part in (embedded, urlencode(query, doseq=True) if query else "") if part
    ]
    return urlunsplit((base.scheme, base.netloc, joined or "/", "&".join(pieces), ""))


def url_under_base(config: SkillConfig, url: str) -> bool:
    """Whether *url* still points inside one configured base path."""
    candidate = urlsplit(url)
    if candidate.username or candidate.password:
        return False
    path = _safe_request_path(candidate.path)
    if path is None:
        return False
    return any(
        _same_origin_and_path(candidate, base_url, path)
        for base_url in config.effective_base_urls
    )


def _safe_request_path(raw: str) -> str | None:
    decoded = unquote(raw or "/")
    if "\\" in decoded or ".." in decoded.split("/"):
        return None
    return posixpath.normpath(decoded)


def _same_origin_and_path(candidate: SplitResult, base_url: str, path: str) -> bool:
    base = urlsplit(base_url)
    if (candidate.scheme, candidate.hostname, candidate.port) != (
        base.scheme,
        base.hostname,
        base.port,
    ):
        return False
    base_path = posixpath.normpath(base.path or "/").rstrip("/")
    return not base_path or path == base_path or path.startswith(base_path + "/")


def _relative_path(path: str) -> tuple[str, str]:
    raw_path, _, embedded = path.partition("?")
    clean = raw_path.strip()
    if "\\" in clean or clean.startswith("//"):
        raise EndpointRefused("a skill path must be relative to the base URL")
    parsed = urlsplit(clean)
    if parsed.scheme or parsed.netloc:
        raise EndpointRefused("a skill path must not be an absolute URL")
    decoded = unquote(clean)
    if ".." in decoded.split("/") or decoded.startswith("/../"):
        raise EndpointRefused("a skill path must not escape the base URL")
    return decoded, embedded


def _require_under_base(joined: str, base_path: str) -> None:
    if not joined.startswith(base_path or "/"):
        raise EndpointRefused("a skill path must stay under the base URL")
    if base_path and not (joined == base_path or joined.startswith(base_path + "/")):
        raise EndpointRefused("a skill path must stay under the base URL")


def same_origin(first: str, second: str) -> bool:
    """Whether two URLs share scheme, host and effective port."""
    left = urlsplit(first)
    right = urlsplit(second)
    return (left.scheme, left.hostname, left.port) == (
        right.scheme,
        right.hostname,
        right.port,
    )


def ensure_public_endpoint(url: str, *, allow_private: bool) -> PinnedAddress:
    """Resolve and refuse loopback, link-local, metadata and private addresses."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise EndpointRefused("a skill request must be http(s) with a host")
    if parsed.username or parsed.password:
        raise EndpointRefused("a skill request must not carry URL credentials")
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = resolve_addresses(host, port)
    if not addresses:
        raise EndpointRefused(f"could not resolve {host}")
    for ip in addresses:
        _refuse_address(ip, allow_private)
    return PinnedAddress(host=host, port=port, ip=addresses[0])


def _refuse_address(ip: str, allow_private: bool) -> None:
    if is_metadata_ip(ip):
        raise EndpointRefused("the cloud metadata address is not reachable from skills")
    if not allow_private and is_blocked_ip(ip):
        raise EndpointRefused(
            "the skill base URL is public; private and loopback addresses are refused"
        )


def resolve_addresses(host: str, port: int) -> list[str]:
    """Resolve *host*, IPv4 first, with duplicates removed."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise EndpointRefused(f"could not resolve {host}: {exc}") from exc
    found: list[str] = []
    for family, _type, _proto, _canon, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        ip = str(sockaddr[0])
        if ip not in found:
            found.append(ip)
    found.sort(key=lambda item: ":" in item)
    return found


def is_metadata_ip(value: str) -> bool:
    """Whether an address is a cloud metadata endpoint."""
    return value in _METADATA_IPS


def is_blocked_ip(value: str) -> bool:
    """Loopback, link-local, private, reserved, multicast or unspecified."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return True
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def allows_private(config: SkillConfig) -> bool:
    """Whether the skill's own base URL is a private/local system."""
    if config.kind != "http":
        return False
    parsed = urlsplit(config.effective_base_urls[0])
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = resolve_addresses(parsed.hostname, port)
    except EndpointRefused:
        return False
    return any(is_blocked_ip(ip) and not is_metadata_ip(ip) for ip in addresses)


def pinned_request(
    url: str, address: PinnedAddress
) -> tuple[str, dict[str, str], dict[str, object]]:
    """The URL sent to the pinned IP, its Host header and TLS SNI extension."""
    parsed = urlsplit(url)
    default = 443 if parsed.scheme == "https" else 80
    host = f"[{address.ip}]" if ":" in address.ip else address.ip
    netloc = host if address.port == default else f"{host}:{address.port}"
    pinned = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))
    headers = {"Host": parsed.netloc}
    extensions: dict[str, object] = {}
    if parsed.scheme == "https":
        extensions["sni_hostname"] = address.host
    return pinned, headers, extensions


PLACEHOLDER_KEYS = (
    "sender.phone",
    "sender.chat_id",
    "sender.external_user_ref",
    "ticket.number",
)


def substitute_sender(text: str, run: SkillRuntime) -> str:
    """Fill only the code-owned placeholders the request actually uses."""
    used = {key for key in PLACEHOLDER_KEYS if "{" + key + "}" in text}
    if not used:
        return text
    values = sender_values(run, required=used)
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def sender_values(run: SkillRuntime, required: set[str] | None = None) -> dict[str, str]:
    """The sender and ticket facts code fills into a request."""
    ticket, contact = _ticket_and_contact(run)
    phone = _digits(contact.phone if contact is not None else run.chat_id)
    external = (ticket.external_user_ref if ticket is not None else "") or (
        contact.external_user_ref if contact is not None else ""
    )
    values = {
        "sender.phone": phone,
        "sender.chat_id": run.chat_id or (contact.chat_id if contact is not None else ""),
        "sender.external_user_ref": external,
        "ticket.number": str(ticket.number) if ticket is not None else "",
    }
    _require_values(values, required if required is not None else set(PLACEHOLDER_KEYS))
    return values


def _ticket_and_contact(run: SkillRuntime) -> tuple[Ticket | None, Contact | None]:
    if run.ticket_store is None:
        return None, None
    ticket = run.ticket_store.ticket(run.ticket_id) if run.ticket_id else None
    chat_id = ticket.chat_id if ticket is not None else run.chat_id
    return ticket, run.ticket_store.contact(chat_id)


def _require_values(values: dict[str, str], wanted: set[str]) -> None:
    missing = [key for key in wanted if not values.get(key)]
    if missing:
        raise EndpointRefused(
            "the request needs " + ", ".join("{" + key + "}" for key in missing)
        )


def _digits(value: str) -> str:
    return "".join(
        char for char in value.split("@", 1)[0].split(":", 1)[0] if char.isdigit()
    )


def foreign_identifier_error(text: str, run: SkillRuntime) -> str | None:
    """Refuse another person's identifiers in a customer or team-group run."""
    if run.armed:
        return None
    allowed = _allowed_identifiers(run)
    jid_error = _jid_error(text, allowed)
    if jid_error is not None:
        return jid_error
    number_error = _number_error(text, allowed)
    if number_error is not None:
        return number_error
    return _contact_error(text, run, allowed)


def _jid_error(text: str, allowed: set[str]) -> str | None:
    for jid in _JID.findall(text):
        if jid.casefold() not in allowed:
            return (
                "the request names another chat; customer skills may only see the sender"
            )
    return None


def _number_error(text: str, allowed: set[str]) -> str | None:
    for match in _PHONE.finditer(text):
        digits = _digits(match.group())
        if digits and digits not in allowed:
            return (
                "the request names another person; "
                "customer skills may only see the sender"
            )
    return None


def _contact_error(text: str, run: SkillRuntime, allowed: set[str]) -> str | None:
    if run.ticket_store is None:
        return None
    _ticket, own_contact = _ticket_and_contact(run)
    folded = text.casefold()
    compact = re.sub(r"[\s().+-]", "", text)
    for contact in run.ticket_store.contacts():
        if own_contact is not None and contact.chat_id == own_contact.chat_id:
            continue
        for value in (contact.phone, contact.chat_id, contact.external_user_ref):
            if _foreign_value(value, folded, compact, allowed):
                return (
                    "the request names another person; "
                    "customer skills may only see the sender"
                )
    return None


def _foreign_value(value: str, folded: str, compact: str, allowed: set[str]) -> bool:
    if not value:
        return False
    digits = _digits(value)
    if digits and digits in allowed:
        return False
    normalized = value.casefold()
    if normalized in folded and (len(digits) >= 7 or len(normalized) >= 4):
        return True
    return len(digits) >= 7 and digits in compact


def _allowed_identifiers(run: SkillRuntime) -> set[str]:
    allowed: set[str] = set()
    for value in sender_values(run, required=set()).values():
        if value:
            allowed.add(value.casefold())
            digits = _digits(value)
            if digits:
                allowed.add(digits)
    if run.chat_id:
        allowed.add(run.chat_id.casefold())
        digits = _digits(run.chat_id)
        if digits:
            allowed.add(digits)
    return allowed


def check_surface_access(
    config: SkillConfig, run: SkillRuntime, *, write: bool
) -> str | None:
    """Refuse a skill action the run's surface may not perform."""
    if run.armed or run.surface in ("operator", "system"):
        return None
    if config.customer_chats == "none":
        return "this skill is not available in customer chats"
    if write and config.customer_chats != "read_write":
        return "this skill is read-only in customer chats"
    return None


def needs_confirmation(config: SkillConfig, run: SkillRuntime) -> bool:
    """Customer writes follow the skill flag; team-group writes always ask."""
    if run.surface == "team_group":
        return True
    if run.armed or run.surface in ("operator", "system"):
        return False
    return config.writes_need_operator
