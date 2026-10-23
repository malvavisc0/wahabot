"""Contact, translation and session-pause storage for tickets."""

from wahabot.core.db.connection import connect, transaction
from wahabot.core.tickets.models import Contact
from wahabot.core.tickets.store_base import StoreBase, contact_from_row

_CONTACT_COLUMNS = (
    "id, chat_id, phone, lid, display_name, language, external_user_ref, "
    "last_inbound_at, paused_until"
)


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
        """Create or refresh one contact by chat id."""
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO contacts (chat_id, phone, lid, display_name, language, "
                    + "external_user_ref, last_inbound_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    + "ON CONFLICT(chat_id) DO UPDATE SET "
                    + "phone = CASE WHEN excluded.phone != '' THEN excluded.phone "
                    + "ELSE contacts.phone END, "
                    + "lid = CASE WHEN excluded.lid != '' THEN excluded.lid "
                    + "ELSE contacts.lid END, "
                    + "display_name = CASE WHEN excluded.display_name != '' "
                    + "THEN excluded.display_name ELSE contacts.display_name END, "
                    + "language = CASE WHEN excluded.language != '' "
                    + "THEN excluded.language ELSE contacts.language END, "
                    + "external_user_ref = CASE WHEN excluded.external_user_ref "
                    + "!= '' THEN excluded.external_user_ref "
                    + "ELSE contacts.external_user_ref END, "
                    + "last_inbound_at = COALESCE(excluded.last_inbound_at, "
                    + "contacts.last_inbound_at)",
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
                row = conn.execute(
                    f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
        finally:
            conn.close()
        return contact_from_row(row)

    def contact(self, chat_id: str) -> Contact | None:
        """One contact by chat id."""
        return self.query_one(
            f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE chat_id = ?",
            (chat_id,),
            contact_from_row,
        )

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
