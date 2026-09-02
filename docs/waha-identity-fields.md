# WAHA Message Identity Fields

How `from`, `participant`, and `to` identify who sent what — and how
whabot's access lists use them. JIDs below are anonymized.

## Field semantics per chat type

**1:1 chat** (`@c.us`):

| Field | Meaning |
|---|---|
| `from` | the other person's JID |
| `to` | our JID (the account the bot runs on) |
| `participant` | absent, or equals the sender |

**Group** (`@g.us`):

| Field | Meaning |
|---|---|
| `from` | **the group JID** — not a person |
| `participant` | **the individual member** who sent the message |
| `to` | our JID |

**Status update** (`status@broadcast`):

| Field | Meaning |
|---|---|
| `from` | `status@broadcast` — the status feed |
| `participant` | the status poster's JID |
| `to` | us — the viewer |

## Addressing model

In groups the conversation lives at `from` (the group), while the
human behind each message is `participant`. This inverts what you'd
naively expect: `from` is "where the message landed" (which chat), not
"who typed it". JIDs may use `@lid` (linked-device IDs) or `@c.us`
(phone numbers) — `@lid` values are stable identifiers that do not
directly expose a phone number.

## How `bot/config.json` uses these

`chat_allowed()` (in `src/whabot/core/filters.py`) checks, in order:

1. **Blacklist** — matched against both `from` and `participant`
   (blacklisted participant → drop, even if their group is whitelisted).
2. **Whitelist** (when non-empty) — allowed only if either `from` or
   `participant` is listed.

So you can:

- Whitelist a **group JID** → answer everyone in that group
- Whitelist a **participant JID** → answer one person in any chat they
  write to (subject to the group also being allowed)
- Blacklist a **participant JID** → silence one person everywhere

## Access list file

`bot/config.json` (path override: `WHABOT_ACCESS_CONFIG`):

```json
{
  "whitelist": ["<group-or-person-jid>@g.us"],
  "blacklist": []
}
```

Both empty = answer everybody. Blacklist always wins.
