"""Contact, translation and session-pause storage for tickets."""

from wahabot.core.db.connection import connect, transaction
from wahabot.core.tickets.models import Contact
from wahabot.core.tickets.store_base import StoreBase, contact_from_row

_CONTACT_COLUMNS = (
    "id, chat_id, phone, lid, display_name, language, external_user_ref, "
    "last_inbound_at, paused_until"
)
_PHONE_DOMAINS = ("c.us", "s.whatsapp.net")


def phone_jids(number: str) -> tuple[str, ...]:
    """Canonical JIDs a configured phone number may match."""
    user, _, domain = number.partition("@")
    digits = "".join(ch for ch in user.split(":", 1)[0] if ch.isdigit())
    if not digits:
        return ()
    if domain == "lid":
        return (f"{digits}@lid",)
    if domain and domain not in _PHONE_DOMAINS:
        return ()
    return tuple(f"{digits}@{item}" for item in _PHONE_DOMAINS)


class ContactMixin(StoreBase):
    """Contact directory, translation cache and session pause state."""

    def upsert_contact(
        self,
        chat_id: str,
        *,
        phone: str = "",
        lid: str = "",
        display_name: str = "",
        language: str = "",
        external_user_ref: str = "",
        last_inbound_at: int | None = None,
    ) -> Contact:
        """Create or refresh one contact by chat id, phone or lid alias."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                row = conn.execute(
                    f"SELECT {_CONTACT_COLUMNS} FROM contacts "
                    + "WHERE chat_id = ? OR (phone != '' AND phone = ?) "
                    + "OR (lid != '' AND lid = ?) "
                    + "ORDER BY CASE WHEN chat_id = ? THEN 0 ELSE 1 END, id LIMIT 1",
                    (chat_id, phone, lid, chat_id),
                ).fetchone()
                if row is None:
                    cursor = conn.execute(
                        "INSERT INTO contacts (chat_id, phone, lid, display_name, "
                        + "language, external_user_ref, last_inbound_at) "
                        + "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            chat_id,
                            phone,
                            lid,
                            display_name,
                            language,
                            external_user_ref,
                            last_inbound_at,
                        ),
                    )
                    contact_id = int(cursor.lastrowid or 0)
                else:
                    contact_id = int(row["id"])
                    if str(row["chat_id"]) != chat_id:
                        conn.execute(
                            "UPDATE tickets SET chat_id = ? WHERE contact_id = ?",
                            (chat_id, contact_id),
                        )
                    conn.execute(
                        "UPDATE contacts SET chat_id = ?, "
                        + "phone = COALESCE(NULLIF(?, ''), phone), "
                        + "lid = COALESCE(NULLIF(?, ''), lid), "
                        + "display_name = COALESCE(NULLIF(?, ''), display_name), "
                        + "language = COALESCE(NULLIF(?, ''), language), "
                        + "external_user_ref = COALESCE(NULLIF(?, ''), "
                        + "external_user_ref), "
                        + "last_inbound_at = CASE "
                        + "WHEN ? IS NULL THEN last_inbound_at "
                        + "WHEN last_inbound_at IS NULL OR ? > last_inbound_at THEN ? "
                        + "ELSE last_inbound_at END "
                        + "WHERE id = ?",
                        (
                            chat_id,
                            phone,
                            lid,
                            display_name,
                            language,
                            external_user_ref,
                            last_inbound_at,
                            last_inbound_at,
                            last_inbound_at,
                            contact_id,
                        ),
                    )
                refreshed = conn.execute(
                    f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE id = ?",
                    (contact_id,),
                ).fetchone()
                if refreshed is None:
                    raise KeyError(f"contact {contact_id} not found after upsert")
        finally:
            conn.close()
        return contact_from_row(refreshed)

    def contact(self, chat_id: str) -> Contact | None:
        """One contact by chat id."""
        return self.query_one(
            f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE chat_id = ?",
            (chat_id,),
            contact_from_row,
        )

    def contact_by_phone(self, phone: str) -> Contact | None:
        """One contact by phone number or its chat id form.

        Matches the stored phone JID as well as the chat JID, so a
        phone-only alias behind a LID chat still resolves. Domains are
        explicit: a phone number never matches a group or LID JID.
        """
        jids = phone_jids(phone)
        if not jids:
            return None
        column = "lid" if jids[0].endswith("@lid") else "phone"
        placeholders = ", ".join("?" for _ in jids)
        return self.query_one(
            f"SELECT {_CONTACT_COLUMNS} FROM contacts "
            + f"WHERE {column} IN ({placeholders}) "
            + f"OR chat_id IN ({placeholders}) "
            + "ORDER BY id LIMIT 1",
            (*jids, *jids),
            contact_from_row,
        )

    def night_ready(self, number: str) -> bool:
        """Whether the night number is a known contact (it wrote first)."""
        return self.contact_by_phone(number) is not None

    def contact_by_id(self, contact_id: int) -> Contact | None:
        """One contact by row id."""
        return self.query_one(
            f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE id = ?",
            (contact_id,),
            contact_from_row,
        )

    def contacts(self) -> list[Contact]:
        """Every known contact, newest inbound first."""
        return self.query_all(
            f"SELECT {_CONTACT_COLUMNS} FROM contacts "
            + "ORDER BY last_inbound_at DESC, id DESC",
            (),
            contact_from_row,
        )

    def set_contact_paused(self, chat_id: str, until: int | None) -> None:
        """Set or clear one contact's pause deadline."""
        self.execute(
            "UPDATE contacts SET paused_until = ? WHERE chat_id = ?", (until, chat_id)
        )

    def set_contact_language(self, contact_id: int, language: str) -> None:
        """Store a detected customer language once."""
        self.execute(
            "UPDATE contacts SET language = ? WHERE id = ?", (language, contact_id)
        )

    def set_chat_paused(self, chat_id: str, until: int | None) -> None:
        """Set or clear a pause keyed by the chat, not a contact row."""
        if until is None:
            self.execute("DELETE FROM chat_pauses WHERE chat_id = ?", (chat_id,))
            return
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO chat_pauses (chat_id, paused_until) VALUES (?, ?) "
                    + "ON CONFLICT(chat_id) DO UPDATE SET "
                    + "paused_until = excluded.paused_until",
                    (chat_id, until),
                )
        finally:
            conn.close()

    def chat_paused_until(self, chat_id: str) -> int | None:
        """A chat-keyed pause deadline, when one is set."""
        return self.query_one(
            "SELECT paused_until FROM chat_pauses WHERE chat_id = ?",
            (chat_id,),
            lambda item: int(item["paused_until"]),
        )

    def paused_chats(self) -> list[tuple[str, int]]:
        """Every chat-keyed pause, oldest deadline first."""
        return self.query_all(
            "SELECT chat_id, paused_until FROM chat_pauses ORDER BY paused_until",
            (),
            lambda item: (str(item["chat_id"]), int(item["paused_until"])),
        )

    def translation(self, source_hash: str, target_lang: str) -> str | None:
        """A cached translation, when one exists."""
        row = self.query_one(
            "SELECT text FROM translations WHERE source_hash = ? AND target_lang = ?",
            (source_hash, target_lang),
            lambda item: str(item["text"]),
        )
        return row

    def save_translation(
        self, source_hash: str, target_lang: str, text: str, at: int | None = None
    ) -> None:
        """Cache one translation by source hash and target language."""
        moment = self.now() if at is None else at
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO translations "
                    + "(source_hash, target_lang, text, created_at) "
                    + "VALUES (?, ?, ?, ?) ON CONFLICT(source_hash, target_lang) "
                    + "DO UPDATE SET text = excluded.text",
                    (source_hash, target_lang, text, moment),
                )
        finally:
            conn.close()

    def session_paused_until(self, session: str) -> int | None:
        """The session's global pause deadline, when one is set."""
        row = self.query_one(
            "SELECT paused_until FROM session_state WHERE session = ?",
            (session,),
            lambda item: (
                int(item["paused_until"]) if item["paused_until"] is not None else None
            ),
        )
        return row

    def set_session_paused(self, session: str, until: int | None) -> None:
        """Set or clear the global pause for one session."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO session_state (session, paused_until, updated_at) "
                    + "VALUES (?, ?, ?) ON CONFLICT(session) DO UPDATE SET "
                    + "paused_until = excluded.paused_until, "
                    + "updated_at = excluded.updated_at",
                    (session, until, self.now()),
                )
        finally:
            conn.close()
