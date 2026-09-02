# Session Config (`data/sessions/<session>.json`)

Every WhatsApp session is configured by a JSON file at
`data/sessions/<session>.json` (the session name is `WHABOT_SESSION`,
default `default`). whabot loads it at startup via
`load_session_config()` and uses it for access control, the agent's
system prompt, and how the bot participates in group chats.

## Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `whitelist` | `string[]` | `[]` | Allowed chats/participants. Empty = answer everybody. |
| `blacklist` | `string[]` | `[]` | Denied chats/participants. Always wins over the whitelist. |
| `system_prompt` | `string \| null` | `null` | Overrides the agent system prompt. Supports `{{date}}`, `{{time}}`, `{{now}}`, `{{tz}}` variables. Falls back to `WHABOT_AGENT_SYSTEM_PROMPT`. |
| `bot_name` | `string \| null` | `null` | The bot's display name (e.g. `"Kai"`). Used only as a fallback mention matcher. |
| `bot_mention_regex` | `string \| null` | `null` | A regex detecting when the bot is addressed in a group. Defaults to a case-insensitive whole-word `@?<bot_name>`. |
| `group_participation` | `"never" \| "mentioned" \| "judicious"` | `"mentioned"` | How the bot joins group conversations (below). |

Example:

```json
{
  "whitelist": ["120363012345678901@g.us"],
  "blacklist": [],
  "system_prompt": "You are Kai, a helpful assistant... Today is {{date}}. The current time is {{time}} ({{tz}}).",
  "bot_name": "Kai",
  "bot_mention_regex": "(?i)@?[kĸ]a[iy]",
  "group_participation": "judicious"
}
```

## System prompt variables

`system_prompt` supports these placeholders (resolved at startup with
`WHABOT_TIMEZONE`, default `UTC`):

- `{{date}}` — e.g. `2026-09-02`
- `{{time}}` — e.g. `14:30`
- `{{now}}` / `{{datetime}}` — e.g. `2026-09-02 14:30 UTC`
- `{{tz}}` — the timezone name, e.g. `America/Santiago`

## Group participation

In **1:1 chats** the bot always replies. In **groups** the
`group_participation` mode decides when to wake the agent:

| Mode | Behavior |
|---|---|
| `never` | Never reply in groups. |
| `mentioned` (default) | Reply only when the bot is mentioned by name/regex **or** when someone replies to one of the bot's messages. |
| `judicious` | Also run the agent on unmentioned group text. The system prompt guides it to reply only when its input adds value; if the agent returns empty, the bot stays silent. |

### Mention detection

A group message "addresses" the bot when:

1. the bot's own JID appears in `mentionedJidList` (real WhatsApp
   @-mention), **or**
2. `bot_mention_regex` (or the `bot_name` fallback) matches the message
   text.

The regex is applied as-is, so you control the variants — for example,
to let people reach Kai as `@kai`, `@kay`, `ĸay` or just `kai`:

```json
"bot_mention_regex": "(?i)@?[kĸ]a[iy]"
```

`(?i)` makes the character class case-insensitive, so `@KAY`,
`ĸay`, `kai` and `kay` all count as a mention.

## Access control

`chat_allowed()` (in `src/whabot/core/filters.py`) applies
whitelist/blacklist against both the chat `from` JID and the message
`participant`:

1. **Blacklist** — a blacklisted char or participant is always dropped.
2. **Whitelist** (when non-empty) — allowed only if `from` **or**
   `participant` is listed.

Practical uses:

- Whitelist a **group JID** → answer everyone in that group.
- Whitelist a **participant JID** → answer one person wherever they
  write (subject to their chat also being allowed).
- Blacklist a **participant JID** → silence one person everywhere.
