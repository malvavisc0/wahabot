# Fix plan: real WhatsApp mentions in `send_message`

**Status**: proposal · **Date**: 2026-09-04 · **Scope**: `core/waha.py`, `ai/tools/whatsapp.py`, `ai/tools/schemas.py`

---

## The problem in one paragraph

When the bot wants to single someone out in a group, the best it can do
today is type `@Alice` as plain text. That is not a WhatsApp mention:
no highlight, no notification — just characters. A real mention needs
the sender to ship a *list of JIDs* alongside the message text
(`mentions: ["<jid>@lid"]`), and each mentioned JID's display
name must appear in the text as `@<name>`. WhatsApp matches the two up
and renders the blue pill. wahabot currently can't do any of this:
the HTTP client drops the field, the tool doesn't expose it, and the
model has no way to learn which JID belongs to which person.

## How WhatsApp mentions actually work

A mention is a *pair* of things travelling together in the sendText
request:

```
text:     "y tu @Alice, callate"      ← @ + display name IN the text
mentions: ["<jid>@lid"]       ← the JID of that person
```

The client (WhatsApp web, phone, or in our case WAHA) overlays the
`@Name` substring with the rich pill and registers a notification for
that JID. If the two sides don't correspond — a JID listed but no
matching `@Name` in the text, or the reverse — you get either nothing
or an awkward dangling `@`.

Two identity systems matter here, and the bot already handles both on
the *incoming* side (`bot_jids` in `ai/messages.py`):

- **Phone JIDs** — `<jid>@c.us` — used in normal groups.
- **LID JIDs** — `<jid>@lid` — used in communities and some
  newer groups; this is what `mentionedJidList` carries in the traces
  we looked at.

The roster (`get_chat_overview`) returns participants as objects with
both `id` and a display name (`name` / `pushname` / `notifyName`), so
the JID-to-name mapping is already fetchable — it just never reaches
the model in a usable shape.

## The fix, layer by layer

### 1. `WahaClient.send_text` — pass mentions through

`core/waha.py`: add an optional parameter and stop dropping it:

```python
def send_text(
    self, session: str, chat_id: str, text: str,
    reply_to: str | None = None,
    mentions: list[str] | None = None,
) -> None:
    body: dict[str, Any] = {"session": session, "chatId": chat_id, "text": text}
    if reply_to:
        body["reply_to"] = reply_to
    if mentions:
        body["mentions"] = mentions
    ...
```

WAHA's sendText schema accepts `mentions` as an array of strings; the
field name is `mentions` (not `mentionedJidList` — that's the *incoming*
field name in event payloads).

### 2. `send_message` tool — expose it to the model

`ai/tools/whatsapp.py`: new optional `mentions` parameter, JSON-encoded
as a list of JIDs (the tool-schema system handles list[str] via
pydantic). The description is the critical part — the model must learn
the pairing rule or it will keep sending plain text:

> `mentions`: Optional list of WhatsApp JIDs to @-mention. For every
> JID here, its owner's display name must appear in `text` as
> `@<name>` — WhatsApp highlights and notifies based on this pairing.
> Get JIDs and names from `get_chat` (participants) or
> `fetch_chat_messages` (sender ids). Never invent JIDs.

The one-send latch and error envelopes stay as they are; a mention ride
along is just another field in the same POST.

### 3. Make JIDs *findable* — roster with names

The model can't mention people it can't identify. Two changes:

- **`get_chat`**: today it caps `participant_jids` at 20 entries and
  gives bare JIDs with no names. Change the summary to include
  `participants: [{"id": "<jid>", "name": "Alice"}, ...]` — pairs,
  not two separate lists the model has to zip by guesswork.
- **`slim_message`**: keep the sender's `participant` JID (already
  kept) — this gives per-message context like "the person who just said
  X is JID Y", which pairs with the roster to build a name→JID mental
  map.

`get_chat` output is already budget-capped; participant pairs cost ~50
chars each, so 20 participants ≈ 1000 chars — within the envelope if we
keep the per-message body cap from the earlier fix.

### 4. Prompt nudge (one line)

`data/sessions/default.json` — the system prompt's Style section gets
one sentence, so the model knows the feature exists:

> To mention someone in a group, pass their JID in `mentions` and put
> `@<their name>` in the text — plain `@name` typing alone doesn't
> notify anyone.

## What we deliberately do NOT do

- **No automatic mention-detection in `text`.** Guessing "did the
  model mean to mention Alice when it typed @Alice?" from the
  roster is fuzzy-matching with real notification side effects. The
  model asks explicitly via the `mentions` field or it didn't happen.
  (A lint warning in the tool result — "your text contains @Name but
  no mention was attached" — is a possible later refinement, not v1.)
- **No incoming-mention forwarding.** If the bot is asked to relay
  "tell @X that Y", it should resolve X itself via the roster, not
  copy the incoming `mentionedJidList` — those JIDs refer to the
  *source* chat's participants and may not exist in the target chat.

## Test plan

Same style as the existing envelope/list tests:

- `send_text` with mentions produces a POST body containing
  `mentions: [...]` (mock client, inspect request).
- Tool happy path: model passes `mentions=["<lid>"]`, text contains
  `@Alice`, envelope reports `mentions: 1`.
- Misuse cases: JIDs without matching names in text → message still
  sends (WhatsApp degrades gracefully) but the envelope notes it;
  empty mentions list → field omitted entirely.
- Roster path: `get_chat` returns `participants` as id/name pairs.
- Smoke test end-to-end: scripted first LLM response includes a
  mention, RecordingWaha asserts the `mentions` array reached the
  "HTTP" layer.

## Risks and notes

- **WAHA version**: `mentions` on sendText is supported since WAHA
  2023.x; the instance at 192.168.1.212:3000 is recent (the traces
  show current-gen payloads), so no version gate needed. Verify once
  by hand with curl before rollout.
- **LID vs phone JID**: in LID groups the *incoming* roster may show
  `@lid` ids while people's clients render names; both formats work
  in `mentions` as long as the JID is a member of that chat. Using the
  roster's exact ids avoids the ambiguity entirely — one more reason
  to make get_chat the source of truth.
- **Token cost**: negligible; one extra optional field per call and
  slightly richer `get_chat` output.

## Order of implementation

1. `WahaClient.send_text` mentions param (core enabler, 5 lines).
2. `get_chat` participant pairs (makes JIDs resolvable).
3. `send_message` schema + description (model-facing surface).
4. Prompt sentence.
5. Tests + smoke script update.
6. One manual curl check against the live WAHA.
