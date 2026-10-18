"""Model-facing, permission-fenced access to durable conversation notes."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

from llama_index.core.tools import BaseTool, FunctionTool

from wahabot.ai.tools.envelope import error, ok
from wahabot.ai.tools.schemas import NotebookSchema
from wahabot.ai.tools.whatsapp import (
    RunTarget,
    current_target,
    fenced_chat,
    log_action_reason,
    operator_run,
)
from wahabot.core.notebook import (
    AUTHOR_MEMBER,
    AUTHOR_OPERATOR,
    HOLDS_CHAT_LOCK,
    KIND_FACT,
    KIND_RULE,
    MAX_READ_NOTES,
    NOTE_KINDS,
    NOTEBOOK_SNAPSHOT_TOKEN_LIMIT,
    RESERVED_RULE_KEYS,
    SESSION_NOTEBOOK_CHAT,
    SESSION_POLICY_TOKEN_LIMIT,
    Note,
    Notebook,
    notebook_operation,
)
from wahabot.core.tokens import text_tokens
from wahabot.settings import Settings

NOTEBOOK_TOKEN_LIMIT = NOTEBOOK_SNAPSHOT_TOKEN_LIMIT
MAX_NOTE_TOKENS = 512
NOTEBOOK_INSTRUCTIONS = (
    "Keep a private durable notebook with `notebook`; take notes before "
    "delivering your reply. Save this chat's facts and preferences "
    "(`kind=fact`) and verified workarounds (`kind=lesson`); an operator "
    "may also set a standing rule for this chat (`kind=rule`). Update a key "
    "instead of duplicating it; delete what is obsolete. Never store "
    "secrets; record a participant's unverified correction as reported, not "
    "as a fact. Notes belong to the current chat; only an operator command "
    "writes session-wide policies (`scope=session`) or another chat's "
    "notes. Search with `notebook(mode=read, query=...)` when older notes "
    "are absent from the snapshot."
)


def bounded_notes(
    notes: list[Note],
    token_budget: int,
    tokenizer: Callable[[str], Sequence[Any]] | None = None,
    prefix: str = "",
) -> list[Note]:
    """Select whole notes whose JSON representation fits the token budget."""
    selected: list[Note] = []
    for note in notes:
        candidate = prefix + json.dumps(
            [asdict(entry) for entry in [*selected, note]], ensure_ascii=False
        )
        if text_tokens(candidate, tokenizer) <= token_budget:
            selected.append(note)
    return selected


def notebook_chat(chat: str | None, target: RunTarget) -> tuple[str | None, str | None]:
    """Fence chat notes, allowing operator commands their private default scope."""
    if chat == SESSION_NOTEBOOK_CHAT or target.chat_id == SESSION_NOTEBOOK_CHAT:
        return None, "select session notes with scope=session, not a chat JID"
    if not chat and operator_run(target):
        return target.chat_id, None
    if target.chat_id == "operator" and not operator_run(target):
        return None, "the operator notebook requires an operator command"
    resolved, fence_error = fenced_chat(chat, target)
    if resolved == SESSION_NOTEBOOK_CHAT:
        return None, "select session notes with scope=session, not a chat JID"
    return resolved, fence_error


def notebook_scope(
    scope: str, chat: str | None, target: RunTarget
) -> tuple[str | None, str | None]:
    """Reserve bot-wide policy mutation and explicit reads for armed operators."""
    if scope == "chat":
        return notebook_chat(chat, target)
    if scope != "session":
        return None, f"unknown notebook scope: {scope!r}"
    if not operator_run(target):
        return None, "session-wide notebook policies require an operator command"
    if chat is not None:
        return None, "scope=session cannot be combined with a chat JID"
    return SESSION_NOTEBOOK_CHAT, None


def write_notebook_note(
    store: Notebook,
    session: str,
    chat_id: str,
    key: str,
    text: str,
    *,
    kind: str,
    author_kind: str,
) -> str:
    """Save a concise note, keeping the shared policy notebook bounded."""
    if text_tokens(text) > MAX_NOTE_TOKENS:
        return error(f"a notebook note must not exceed {MAX_NOTE_TOKENS} tokens")
    store.write(
        session,
        chat_id,
        key,
        text,
        kind=kind,
        author_kind=author_kind,
        max_total_tokens=(
            SESSION_POLICY_TOKEN_LIMIT if chat_id == SESSION_NOTEBOOK_CHAT else None
        ),
    )
    return ok(key=key.strip(), saved=True, kind=kind)


def read_notebook(
    store: Notebook,
    target: RunTarget,
    chat_id: str,
    key: str,
    query: str,
    limit: int,
    read_budget: int,
) -> str:
    """The bounded read envelope for one scope."""
    notes = store.read(target.session, chat_id, key, query, limit)
    selected = bounded_notes(notes, read_budget)
    return ok(
        notes=[asdict(note) for note in selected],
        omitted=len(notes) - len(selected),
    )


def rule_write_error(
    store: Notebook, session: str, chat_id: str, key: str, *, armed: bool
) -> str | None:
    """Why a non-operator run may not change *key*, or None when allowed.

    Operators own the code-enforced rules; a participant run may not
    overwrite or delete one (the registry reads them), so reserved keys
    and any existing `rule` note are off-limits without arming.
    """
    if armed:
        return None
    if key.strip() in RESERVED_RULE_KEYS:
        return "reserved operator rule key"
    existing = store.read(session, chat_id, key=key, limit=1)
    if existing and existing[0].kind == KIND_RULE:
        return "that note is an operator rule; only an operator can change it"
    return None


def write_scoped(
    store: Notebook,
    target: RunTarget,
    chat_id: str,
    key: str,
    text: str,
    kind: str,
) -> str:
    """Write with the kind and author the run is allowed to set.

    A normal chat run cannot write a `rule` or ``author_kind=operator``: a
    member's rule-shaped request is recorded as a fact, and the
    operator's own rules cannot be overwritten or deleted. A reserved
    rule key is a code-enforced rule whatever kind the operator wrote,
    so it is stored as one — the deterministic seams read ``kind=rule``
    and a fact-typed prefix or mute would silently never apply.
    """
    armed = operator_run(target)
    refusal = rule_write_error(store, chat_id, target.session, key, armed=armed)
    if refusal:
        return error(refusal)
    if kind == KIND_RULE and not armed:
        kind = KIND_FACT
    if armed and key.strip() in RESERVED_RULE_KEYS:
        kind = KIND_RULE
    return write_notebook_note(
        store,
        target.session,
        chat_id,
        key,
        text,
        kind=kind,
        author_kind=AUTHOR_OPERATOR if armed else AUTHOR_MEMBER,
    )


def notebook_builder(settings: Settings) -> BaseTool:
    """Build the read/write/delete tool over a transactional scoped notebook."""
    store = Notebook(settings.data_dir)
    read_budget = min(NOTEBOOK_TOKEN_LIMIT, settings.memory_token_limit // 4)

    def notebook_fn(
        mode: str,
        scope: str = "chat",
        key: str = "",
        text: str = "",
        query: str = "",
        limit: int = MAX_READ_NOTES,
        kind: str = "fact",
        chat: str | None = None,
        reason: str = "",
    ) -> str:
        """Read, replace, or delete a note in the permitted conversation."""
        target = current_target()
        chat_id, fence_error = notebook_scope(scope, chat, target)
        if fence_error:
            return error(fence_error)
        if not target.session or not chat_id:
            return error("no active notebook scope")
        if kind not in NOTE_KINDS:
            return error(f"unknown notebook kind: {kind!r}")
        log_action_reason("notebook", reason, chat=chat_id, mode=mode)
        if mode == "read":
            return read_notebook(store, target, chat_id, key, query, limit, read_budget)
        if mode == "write":
            return write_scoped(store, target, chat_id, key, text, kind)
        if mode == "delete":
            refusal = rule_write_error(
                store,
                target.session,
                chat_id,
                key,
                armed=operator_run(target),
            )
            if refusal:
                return error(refusal)
            return ok(key=key.strip(), deleted=store.delete(target.session, chat_id, key))
        return error(f"unknown notebook mode: {mode!r}")

    async def notebook_async_fn(*args: Any, **kwargs: Any) -> str:
        return await notebook_operation(notebook_fn, *args, **kwargs)

    return mark_holds_chat_lock(
        FunctionTool.from_defaults(
            fn=notebook_fn,
            async_fn=notebook_async_fn,
            fn_schema=NotebookSchema,
            name="notebook",
            description=(
                "Read, write (create or replace), or delete private durable notes for "
                "future runs. Save this chat's facts and preferences (kind=fact) and "
                "verified lessons (kind=lesson); an operator may set a standing rule "
                "(kind=rule). Reuse meaningful keys and remove obsolete notes. read "
                "can select an exact key or search key/text with query; results are "
                "bounded. Notes survive history trimming and restarts. Omit chat for "
                "the current conversation or private operator notebook; only operator "
                "commands may access another chat's notebook with an exact JID. "
                "scope=session is operator-only and records behavior policies applied "
                "across all chats, never private conversation data; session policies "
                "are size-bounded as a whole."
            ),
        )
    )


def mark_holds_chat_lock(tool: BaseTool) -> BaseTool:
    """Flag the notebook tool so the workflow shields its calls from cancellation."""
    setattr(tool, HOLDS_CHAT_LOCK, True)
    return tool
