"""Code-enforced skill guardrails: URL lock, address checks and sender scope."""

import ipaddress
import posixpath
import re
import socket
from dataclasses import dataclass
from urllib.parse import SplitResult, quote, unquote, urlencode, urlsplit, urlunsplit

from wahabot.core.skills.errors import SkillError
from wahabot.core.skills.models import READ_METHODS, SkillConfig, SkillRuntime
from wahabot.core.tickets.models import Contact, Ticket

_PHONE = re.compile(r"\+\d[\d\s().-]{6,}\d")
_JID = re.compile(r"\b\d+(?:-\d+)?@[a-z.]+", re.IGNORECASE)
_METADATA_IPS = frozenset(
    {
        "169.254.169.254",
        "fd00:ec2::254",
        "100.100.100.200",
        "168.63.129.16",
    }
)


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
    encoded, embedded = relative_path(path)
    urls = config.effective_base_urls
    if not urls:
        raise EndpointRefused("the skill has no base URL")
    base = urlsplit(urls[0])
    base_encoded = posixpath.normpath(base.path or "/").rstrip("/")
    base_decoded = posixpath.normpath(decoded_path(base.path or "/")).rstrip("/")
    relative_encoded = encoded.lstrip("/")
    relative_decoded = decoded_path(relative_encoded)
    joined_encoded = (
        posixpath.normpath(f"{base_encoded}/{relative_encoded}")
        if relative_encoded
        else base_encoded
    )
    joined_decoded = (
        posixpath.normpath(f"{base_decoded}/{relative_decoded}")
        if relative_decoded
        else base_decoded
    )
    require_under_base(joined_decoded, base_decoded)
    require_under_base(joined_encoded, base_encoded)
    pieces = [
        part
        for part in (
            embedded.replace("#", "%23"),
            urlencode(query, doseq=True) if query else "",
        )
        if part
    ]
    return urlunsplit(
        (base.scheme, base.netloc, joined_encoded or "/", "&".join(pieces), "")
    )


def url_under_base(config: SkillConfig, url: str) -> bool:
    """Whether *url* still points inside one configured base path."""
    candidate = urlsplit(url)
    if candidate.username or candidate.password:
        return False
    path = safe_request_path(candidate.path)
    if path is None:
        return False
    return any(
        same_origin_and_path(candidate, base_url, path)
        for base_url in config.effective_base_urls
    )


def safe_request_path(raw: str) -> str | None:
    """A decoded normal path, or None when it tries to escape."""
    decoded = decoded_path(raw or "/")
    if has_unsafe_path(decoded):
        return None
    return posixpath.normpath(decoded)


def same_origin_and_path(candidate: SplitResult, base_url: str, path: str) -> bool:
    """Whether a parsed URL is on the base origin and under its path."""
    base = urlsplit(base_url)
    left = origin_key(candidate)
    right = origin_key(base)
    if left is None or right is None or left != right:
        return False
    base_path = posixpath.normpath(decoded_path(base.path or "/")).rstrip("/")
    return not base_path or path == base_path or path.startswith(base_path + "/")


def relative_path(path: str) -> tuple[str, str]:
    """Split a caller path into a safe encoded path and its raw query."""
    raw_path, _, embedded = path.partition("?")
    clean = raw_path.strip()
    if clean.startswith("//") or "#" in clean:
        raise EndpointRefused("a skill path must be relative to the base URL")
    parsed = urlsplit(clean)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise EndpointRefused("a skill path must not be an absolute URL")
    if has_unsafe_path(decoded_path(clean)):
        raise EndpointRefused("a skill path must not escape the base URL")
    return clean, embedded


def decoded_path(value: str) -> str:
    """Decode a path repeatedly, so double-encoded escapes are visible."""
    decoded = unquote(value)
    for _ in range(4):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def has_unsafe_path(decoded: str) -> bool:
    """Traversal, backslash or control characters in a decoded path."""
    return (
        "\\" in decoded
        or ".." in decoded.split("/")
        or any(ord(char) < 32 or ord(char) == 127 for char in decoded)
    )


def require_under_base(joined: str, base_path: str) -> None:
    """Refuse a path that leaves the configured base path."""
    if not joined.startswith(base_path or "/"):
        raise EndpointRefused("a skill path must stay under the base URL")
    if base_path and not (joined == base_path or joined.startswith(base_path + "/")):
        raise EndpointRefused("a skill path must stay under the base URL")


def same_origin(first: str, second: str) -> bool:
    """Whether two URLs share scheme, host and effective port."""
    left = origin_key(urlsplit(first))
    right = origin_key(urlsplit(second))
    return left is not None and left == right


def effective_port(parsed: SplitResult) -> int | None:
    """The URL's port, defaulting by scheme and refusing malformed values."""
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is not None:
        return port
    if parsed.scheme == "https":
        return 443
    return 80 if parsed.scheme == "http" else None


def origin_key(parsed: SplitResult) -> tuple[str, str | None, int | None] | None:
    """The (scheme, host, effective port) triple, or None when unparseable."""
    port = effective_port(parsed)
    if port is None:
        return None
    return parsed.scheme, parsed.hostname, port


def ensure_public_endpoint(url: str, *, allow_private: bool) -> PinnedAddress:
    """Resolve and refuse loopback, link-local, metadata and private addresses."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise EndpointRefused("a skill request must be http(s) with a host")
    if parsed.username or parsed.password:
        raise EndpointRefused("a skill request must not carry URL credentials")
    port = effective_port(parsed)
    if port is None:
        raise EndpointRefused("a skill request has an invalid port")
    host = parsed.hostname
    addresses = resolve_addresses(host, port)
    if not addresses:
        raise EndpointRefused(f"could not resolve {host}")
    for ip in addresses:
        refuse_address(ip, allow_private)
    return PinnedAddress(host=host, port=port, ip=addresses[0])


def refuse_address(ip: str, allow_private: bool) -> None:
    """Refuse metadata always, and private/non-global unless the base is local."""
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
    except (OSError, UnicodeError) as exc:
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
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return value in _METADATA_IPS
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return str(address) in _METADATA_IPS


def is_blocked_ip(value: str) -> bool:
    """Loopback, link-local, metadata, private, reserved or non-global."""
    if is_metadata_ip(value):
        return True
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
        or not address.is_global
    )


def allows_private(config: SkillConfig) -> bool:
    """Whether the skill's own base URL is a private/local system."""
    if config.kind != "http":
        return False
    urls = config.effective_base_urls
    if not urls:
        return False
    parsed = urlsplit(urls[0])
    if not parsed.hostname:
        return False
    port = effective_port(parsed)
    if port is None:
        return False
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
    if parsed.scheme == "https" and not is_ip_address(address.host):
        extensions["sni_hostname"] = address.host
    return pinned, headers, extensions


def is_ip_address(value: str) -> bool:
    """Whether a hostname is an IP literal."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


PLACEHOLDER_KEYS = (
    "sender.phone",
    "sender.chat_id",
    "sender.external_user_ref",
    "ticket.number",
)
_PLACEHOLDER = re.compile(
    r"\{(" + "|".join(re.escape(key) for key in PLACEHOLDER_KEYS) + r")\}"
)


def substitute_sender(text: str, run: SkillRuntime) -> str:
    """Fill sender and ticket placeholders in plain text."""
    return fill_placeholders(text, run, encode=False)


def substitute_path(text: str, run: SkillRuntime) -> str:
    """Fill placeholders in a URL path, percent-encoding each value."""
    return fill_placeholders(text, run, encode=True)


def fill_placeholders(text: str, run: SkillRuntime, *, encode: bool) -> str:
    """Replace used placeholders in one pass so values are never re-expanded."""
    used = {key for key in PLACEHOLDER_KEYS if "{" + key + "}" in text}
    if not used:
        return text
    values = sender_values(run, required=used)
    if encode:
        values = {key: quote(value, safe="") for key, value in values.items()}
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], text)


def sender_values(run: SkillRuntime, required: set[str] | None = None) -> dict[str, str]:
    """The sender and ticket facts code fills into a request."""
    ticket, contact = ticket_and_contact(run)
    identity = sender_identity(run, ticket, contact)
    values = {
        "sender.phone": sender_phone(contact, identity, run.sender_phone),
        "sender.chat_id": identity,
        "sender.external_user_ref": sender_external(ticket, contact),
        "ticket.number": str(ticket.number) if ticket is not None else "",
    }
    require_values(values, required if required is not None else set())
    return values


def sender_identity(
    run: SkillRuntime, ticket: Ticket | None, contact: Contact | None
) -> str:
    """The sender's own JID, never a team group's chat id."""
    if run.surface == "team_group":
        if not run.sender_id or run.sender_id == run.chat_id:
            return ""
        return run.sender_id
    return (
        (ticket.chat_id if ticket is not None else "")
        or run.sender_id
        or (contact.chat_id if contact is not None else "")
        or run.chat_id
    )


def sender_phone(contact: Contact | None, identity: str, alias: str = "") -> str:
    """The caller-resolved phone, the contact's phone, or identity digits."""
    if alias:
        return digits(alias)
    return digits(contact.phone if contact is not None and contact.phone else identity)


def sender_external(ticket: Ticket | None, contact: Contact | None) -> str:
    """The current contact ref, falling back to the ticket's snapshot."""
    return (contact.external_user_ref if contact is not None else "") or (
        ticket.external_user_ref if ticket is not None else ""
    )


def ticket_and_contact(run: SkillRuntime) -> tuple[Ticket | None, Contact | None]:
    """The run's ticket and the sender's contact (participant in a team group)."""
    if run.ticket_store is None:
        return None, None
    ticket = run.ticket_store.ticket(run.ticket_id) if run.ticket_id else None
    if run.surface == "team_group":
        if not run.sender_id or run.sender_id == run.chat_id:
            return ticket, None
        chat_id = run.sender_id
    elif ticket is not None:
        chat_id = ticket.chat_id
    else:
        chat_id = run.sender_id or run.chat_id
    return ticket, run.ticket_store.contact(chat_id)


def require_values(values: dict[str, str], wanted: set[str]) -> None:
    """Refuse a request whose used placeholders have no value."""
    missing = [key for key in wanted if not values.get(key)]
    if missing:
        raise EndpointRefused(
            "the request needs " + ", ".join("{" + key + "}" for key in missing)
        )


def digits(value: str) -> str:
    """The numeric local part of a phone-like JID."""
    return "".join(
        char for char in value.split("@", 1)[0].split(":", 1)[0] if char.isdigit()
    )


def foreign_identifier_error(text: str, run: SkillRuntime) -> str | None:
    """Refuse another person's identifiers in a customer or team-group run."""
    if run.armed:
        return None
    allowed = allowed_identifiers(run)
    problem = jid_error(text, allowed)
    if problem is not None:
        return problem
    problem = number_error(text, allowed)
    if problem is not None:
        return problem
    return contact_error(text, run, allowed)


def jid_error(text: str, allowed: set[str]) -> str | None:
    """Refuse a JID that is not the sender's."""
    for jid in _JID.findall(text):
        if jid.casefold() not in allowed:
            return (
                "the request names another chat; customer skills may only see the sender"
            )
    return None


def number_error(text: str, allowed: set[str]) -> str | None:
    """Refuse an international phone number that is not the sender's."""
    for match in _PHONE.finditer(text):
        value = digits(match.group())
        if value and value not in allowed:
            return (
                "the request names another person; "
                "customer skills may only see the sender"
            )
    return None


def contact_error(text: str, run: SkillRuntime, allowed: set[str]) -> str | None:
    """Refuse another known contact's identifiers."""
    if run.ticket_store is None:
        return None
    _ticket, own_contact = ticket_and_contact(run)
    folded = text.casefold()
    compact = re.sub(r"[\s().+-]", "", text)
    for contact in run.ticket_store.contacts():
        if own_contact is not None and contact.chat_id == own_contact.chat_id:
            continue
        for value in (contact.phone, contact.chat_id, contact.external_user_ref):
            if foreign_value(value, folded, compact, allowed):
                return (
                    "the request names another person; "
                    "customer skills may only see the sender"
                )
    return None


def foreign_value(value: str, folded: str, compact: str, allowed: set[str]) -> bool:
    """Whether one contact identifier appears in the request text."""
    if not value:
        return False
    number = digits(value)
    if number and number in allowed:
        return False
    normalized = value.casefold()
    if normalized in folded and (len(number) >= 7 or len(normalized) >= 4):
        return True
    return len(number) >= 7 and number in compact


def allowed_identifiers(run: SkillRuntime) -> set[str]:
    """The sender, chat and ticket identifiers a non-operator request may use."""
    allowed: set[str] = set()
    for value in sender_values(run, required=set()).values():
        if value:
            allowed.add(value.casefold())
            number = digits(value)
            if number:
                allowed.add(number)
    for jid in (run.sender_id, run.chat_id):
        if not jid:
            continue
        allowed.add(jid.casefold())
        number = digits(jid)
        if number:
            allowed.add(number)
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
