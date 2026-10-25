"""Fernet encryption for skill credentials, with redaction helpers."""

import sqlite3
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from wahabot.core.db.connection import connect, transaction
from wahabot.core.skills.errors import SecretKeyError


class SecretStore:
    """Write-only secrets: values enter encrypted and never leave as plaintext API."""

    def __init__(self, path: Path, key: str) -> None:
        self.path = path
        self.key = key

    def connect(self) -> sqlite3.Connection:
        """A fresh connection, for diagnostics and tests."""
        return connect(self.path)

    def put(self, name: str, value: str) -> None:
        """Encrypt and store one secret; missing or invalid keys refuse."""
        token = self.fernet().encrypt(value.encode())
        conn = connect(self.path)
        try:
            with transaction(conn):
                conn.execute(
                    "INSERT INTO secrets (name, encrypted_value, updated_at) "
                    + "VALUES (?, ?, strftime('%s','now')) "
                    + "ON CONFLICT(name) DO UPDATE SET "
                    + "encrypted_value = excluded.encrypted_value, "
                    + "updated_at = excluded.updated_at",
                    (name, token.decode()),
                )
        finally:
            conn.close()

    def get(self, name: str) -> str:
        """Decrypt one secret for a call; never for a response or log."""
        ciphertext = self.ciphertext(name)
        if not ciphertext:
            raise SecretKeyError(f"secret {name} is not configured")
        try:
            return self.fernet().decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise SecretKeyError(
                f"secret {name} cannot be decrypted: WAHABOT_SECRET_KEY is wrong"
            ) from exc

    def ciphertext(self, name: str) -> str:
        """The encrypted value, or ""."""
        conn = connect(self.path)
        try:
            row = conn.execute(
                "SELECT encrypted_value FROM secrets WHERE name = ?", (name,)
            ).fetchone()
        finally:
            conn.close()
        return str(row[0]) if row is not None else ""

    def redact(self, text: str) -> str:
        """Remove every stored secret from *text* at the HTTP boundary."""
        return redact_secrets(text, self.values())

    def values(self) -> list[str]:
        """Every decryptable secret value; wrong keys are ignored, never leaked."""
        conn = connect(self.path)
        try:
            rows = conn.execute("SELECT encrypted_value FROM secrets").fetchall()
        finally:
            conn.close()
        values: list[str] = []
        try:
            fernet = self.fernet()
        except SecretKeyError:
            return values
        for row in rows:
            try:
                values.append(fernet.decrypt(str(row[0]).encode()).decode())
            except InvalidToken:
                continue
        return values

    def fernet(self) -> Fernet:
        """A validated Fernet for the configured key."""
        if not self.key.strip():
            raise SecretKeyError(
                "WAHABOT_SECRET_KEY is required to store or read skill secrets"
            )
        try:
            return Fernet(self.key.encode())
        except (ValueError, TypeError) as exc:
            raise SecretKeyError(
                "WAHABOT_SECRET_KEY must be a base64-encoded 32-byte key"
            ) from exc


def redact_secrets(text: str, values: list[str]) -> str:
    """Replace every non-trivial secret value in *text* with ``***``."""
    for value in sorted(values, key=len, reverse=True):
        if value:
            text = text.replace(value, "***")
    return text
