"""Tier 1 rolling recap: what a conversation established, beyond the FIFO.

The rolling recap is the prompt-carried summary the gate and agent
read instead of raw turns — the safety net that lets the FIFO shrink.
When ``trim_to_budget`` drops a prefix, the dropped turns are merged
into a running per-chat recap: corrections, preferences, facts,
promises, norms — the thread's *flow*. The recap must not restate what
the pinned state block already carries exactly (truth precedence:
pinned > notebook > recap), so the merge prompt forbids it and the
pinned block keeps its own sections authoritative.

Storage mirrors the memory buffer: one JSON file per chat under
``data/memory/<session>/`` — ``<chat_id>.learned.json`` — atomically
written, plaintext, wiped by ``wahabot forget`` together with the
buffer. Fail-soft throughout: a recap is an accelerator, never a
dependency, so every read renders "" and every failure only logs.

The merge is one LLM call, never on the hot path:
:meth:`RecapMerger.schedule` only *records* the dropped text (pure
state, no task, no loop), and the merge runs when
:meth:`RecapMerger.merge_pending` is awaited — the handlers do that
at the persist points, after the reply has been sent and outside the
chat lock. Merges serialize on an internal lock: with the job off the
chat lock, overlapping runs can call ``merge_pending`` concurrently,
and without the lock two merges of the same chat could read the same
existing recap and the later save would silently drop the other
batch's content. A failed merge
drops that batch (the turns are already out of the buffer; the loss
is the recap's coverage, never the run).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from loguru import logger

from wahabot.core.persistence import persistable
from wahabot.core.tokens import text_tokens

#: Envelope format version. Unknown versions are ignored (recap-less
#: start) and left untouched — a future bot version may still read them.
VERSION = 1

#: The recap rides the prompt whole, so it stays small — Tier 1 is
#: capped at ~500 tokens.
LEARNED_RECAP_TOKEN_LIMIT = 500

#: The recap rides the prompt whole; keep its line cap at the merge prompt's
#: ten even if an older or hand-edited file grew larger.
RECAP_MAX_LINES = 10

#: One merge batches at most this much dropped text; older drops from a
#: single huge trim are discarded (they predate the recap's coverage, and
#: merges on later trims keep the recap current from here on).
_DROP_BATCH_CHAR_LIMIT = 16_000

_ENVELOPE_VERSION = "version"
_ENVELOPE_SESSION = "session"
_ENVELOPE_CHAT_ID = "chat_id"
_ENVELOPE_UPDATED_AT = "updated_at"
_ENVELOPE_RECAP = "recap"

MERGE_PROMPT = (
    "You maintain the running recap of one WhatsApp conversation. Merge the "
    "NEW TURNS into the EXISTING RECAP and reply with the merged recap only.\n"
    "- Keep established corrections, preferences, facts, promises, and "
    "group norms; drop chit-chat and anything superseded.\n"
    "- When newer turns contradict the recap, the newer statement wins: "
    "replace the old line, never keep both. If the assistant corrected an "
    "earlier statement, write one `Retracted: <old> -> <new>` line and do "
    "not keep the retracted claim as a standalone fact.\n"
    "- Do not restate ticket state, contact details, or anything the pinned "
    "state block already carries — the recap is the conversation's flow, not "
    "its facts.\n"
    "- Plain bullet lines, at most 10, each one short sentence, in the "
    "conversation's language."
)


def learned_file(data_dir: Path, session: str, chat_id: str) -> Path:
    """Path of a chat's recap file, next to its memory file."""
    return data_dir / "memory" / session / f"{chat_id}.learned.json"


def load_learned(data_dir: Path, session: str, chat_id: str) -> str:
    """Load a chat's recap, or "" when absent, unreadable or invalid.

    Fail-soft: never raises, and a provably unparseable file is renamed
    ``.bad`` for forensics (the same policy as the memory buffer).
    """
    if not persistable(chat_id):
        return ""
    path = learned_file(data_dir, session, chat_id)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Unreadable recap file {path}: {exc}", path=path, exc=exc)
        return ""
    except json.JSONDecodeError as exc:
        logger.warning(
            "Corrupt recap file {path}, renamed to .bad: {exc}", path=path, exc=exc
        )
        _rename_to_bad(path)
        return ""
    if not isinstance(envelope, dict):
        _rename_to_bad(path)
        return ""
    data = cast(dict[str, Any], envelope)
    if data.get(_ENVELOPE_VERSION) != VERSION:
        logger.info(
            "Ignoring recap file {path} with unknown version {version}",
            path=path,
            version=data.get(_ENVELOPE_VERSION),
        )
        return ""
    if data.get(_ENVELOPE_SESSION) != session or data.get(_ENVELOPE_CHAT_ID) != chat_id:
        logger.info(
            "Ignoring recap file {path}: envelope does not match the path", path=path
        )
        return ""
    recap = data.get(_ENVELOPE_RECAP)
    return capped_recap(recap) if isinstance(recap, str) else ""


def save_learned(
    data_dir: Path, session: str, chat_id: str, recap: str, tokenizer: Any = None
) -> None:
    """Write a chat's recap to disk atomically, capped, swallowing failures.

    The payload goes to a ``.tmp`` sibling then ``os.replace``, so a crash
    mid-write leaves the previous good recap. Over-cap recaps are
    truncated at the front (newest lines last survive — the recap's
    lines are newest-information-last). A recap failure must never
    break a run.
    """
    if not persistable(chat_id):
        logger.warning("Not persisting recap for bad chat id {id!r}", id=chat_id)
        return
    recap = capped_recap(recap.strip(), tokenizer)
    path = learned_file(data_dir, session, chat_id)
    tmp = path.with_name(path.name + ".tmp")
    try:
        envelope = {
            _ENVELOPE_VERSION: VERSION,
            _ENVELOPE_SESSION: session,
            _ENVELOPE_CHAT_ID: chat_id,
            _ENVELOPE_UPDATED_AT: int(time.time()),
            _ENVELOPE_RECAP: recap,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(envelope), encoding="utf-8")
        os.replace(tmp, path)
        logger.info(
            "Saved recap for {chat_id}: {lines} lines",
            chat_id=chat_id,
            lines=len(recap.splitlines()),
        )
    except Exception as exc:
        logger.warning("Failed to save recap to {path}: {exc}", path=path, exc=exc)


def forget_learned(data_dir: Path, session: str, chat_id: str) -> bool:
    """Delete a chat's recap file; True when a file was actually removed."""
    if not persistable(chat_id):
        logger.warning("Not forgetting recap of bad chat id {id!r}", id=chat_id)
        return False
    try:
        learned_file(data_dir, session, chat_id).unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Failed to delete recap file: {exc}", exc=exc)
        return False
    return True


def capped_recap(recap: str, tokenizer: Any = None) -> str:
    """*recap* cut to the line and token caps, newest lines kept."""
    if not recap:
        return recap
    lines = recap.splitlines()
    if len(lines) > RECAP_MAX_LINES:
        recap = "\n".join(lines[-RECAP_MAX_LINES:])
    if text_tokens(recap, tokenizer) <= LEARNED_RECAP_TOKEN_LIMIT:
        return recap
    lines = recap.splitlines()
    kept: list[str] = []
    for line in reversed(lines):
        candidate = "\n".join([*kept, line]) if kept else line
        if text_tokens(candidate, tokenizer) > LEARNED_RECAP_TOKEN_LIMIT:
            break
        kept.append(line)
    trimmed = "\n".join(reversed(kept)).strip()
    return trimmed if trimmed else recap[: LEARNED_RECAP_TOKEN_LIMIT * 4].strip()


class RecapMerger:
    """Records dropped-turn text per chat; merges when `merge_pending` runs.

    ``schedule`` never calls the LLM — it only records, so bare test and
    library runs that trim but never merge leave nothing running. The
    merge runs when :meth:`merge_pending` is awaited (the handlers, at
    the persist points, after the reply has been sent and outside the
    chat lock), one bounded LLM
    call per chat with pending text, under the caller's semaphore when
    one is given. A failed merge logs and drops that batch.
    """

    def __init__(
        self,
        llm: Any,
        data_dir: Path,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._llm = llm
        self._data_dir = data_dir
        self._semaphore = semaphore
        self._pending: dict[tuple[str, str], str] = {}
        #: One lock per chat, so only merges of the same chat serialize
        #: (the read-modify-write of that chat's recap) while different
        #: chats still merge concurrently.
        self._merge_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def schedule(self, session: str, chat_id: str, dropped_text: str) -> None:
        """Record *dropped_text* for a later merge; a no-op when empty."""
        if not dropped_text.strip():
            return
        key = (session, chat_id)
        previous = self._pending.get(key, "")
        batch = f"{previous}\n{dropped_text}".strip()
        self._pending[key] = batch[-_DROP_BATCH_CHAR_LIMIT:]

    def has_pending(self) -> bool:
        """Whether any chat has unmerged dropped text recorded."""
        return bool(self._pending)

    async def merge_pending(self) -> None:
        """Merge every pending chat; each failure only drops its batch.

        Serialized per chat: the job runs outside the chat lock, so a
        merge still in flight must never race the next run's merge of
        the same chat — without the lock both would read the same
        existing recap and the later save would drop the other batch's
        content. Different chats take different locks and still merge
        concurrently.
        """
        while self._pending:
            (session, chat_id), text = self._pending.popitem()
            async with self.lock_for((session, chat_id)):
                try:
                    await self.merge(session, chat_id, text)
                except Exception as exc:
                    logger.warning(
                        "Recap merge failed for {chat_id}; batch dropped: {exc}",
                        chat_id=chat_id,
                        exc=exc,
                    )

    def lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        """The per-chat merge lock, created on first use."""
        lock = self._merge_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._merge_locks[key] = lock
        return lock

    async def merge(self, session: str, chat_id: str, dropped_text: str) -> None:
        existing = load_learned(self._data_dir, session, chat_id)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=MERGE_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "EXISTING RECAP:\n"
                    f"{existing or '(none)'}\n\nNEW TURNS:\n{dropped_text}"
                ),
            ),
        ]
        if self._semaphore is not None:
            async with self._semaphore:
                response = await self._llm.achat(messages)
        else:
            response = await self._llm.achat(messages)
        recap = str(response.message.content or "").strip()
        if not recap:
            return
        save_learned(self._data_dir, session, chat_id, recap)


def _rename_to_bad(path: Path) -> None:
    """Move a known-garbage recap file aside for inspection."""
    try:
        os.replace(path, path.with_name(path.name + ".bad"))
    except OSError as exc:
        logger.warning("Could not rename {path} to .bad: {exc}", path=path, exc=exc)
