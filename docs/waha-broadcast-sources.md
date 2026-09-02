# WAHA Broadcast Sources (Status & Newsletters)

Not every `message` webhook is a chat message. Status updates and
newsletter posts arrive as `message` events with special sender JIDs,
and **replying to them is wrong** — the WAHA send API targets chats,
not broadcasts.

## Status updates — `status@broadcast`

| Field | Value / meaning |
|---|---|
| `from` | `status@broadcast` — the status feed, not a real chat |
| `participant` / `_data.author` | the poster's JID (`<poster-jid>@lid` here) |
| `to` | us — the status *viewer* |
| `notifyName` | poster's display name (anonymized) |
| `_data.type` / `subtype` | `chat` / `url` — a text status with link preview |
| `body` / `_data.matchedText` | the status text (an Instagram URL here) |
| `_data.statusAttributions` | `externalShare` with `actionUrl`, `source: 1` — cross-posted from Instagram |
| `_data.title` | link preview title |
| `_data.thumbnail` | base64 JPEG link-preview image (large — trim before storing) |
| `_data.cannotBeRanked` / `canBeReshared` | status privacy flags |

Notes:

- `hasMedia` is `false` even though the status renders with a
  preview image — the thumbnail lives in `_data.thumbnail`, and the
  real media (if any) is a *status* media, not a chat attachment.
- `from` ending in `@broadcast` is the reliable marker. A status may
  also reference `@lid` participant JIDs.
- WAHA docs: engine-dependent; WEBJS exposes `_data`, other engines
  may differ.

## Newsletters — `@newsletter`

Channel/newsletter posts similarly arrive as `message` events from a
JID ending `@newsletter`. Same treatment: read-only, do not reply.

## Handling rule

whabot's `is_replyable()` (in `src/whabot/ai/messages.py`) rejects
senders ending in:

```python
NON_REPLYABLE_SUFFIXES = ("@broadcast", "@newsletter")
```

It runs **before** `extract_text()` in the message handler — status
and newsletter events are dropped without reaching the agent, and the
bot never attempts to send to `status@broadcast`.
