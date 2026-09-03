# Session Config (`data/sessions/<session>.json`)

Every WhatsApp session is configured by a JSON file at
`data/sessions/<session>.json` (the session name is `WAHABOT_SESSION`,
default `default`). wahabot loads it at startup via
`load_session_config()` and uses it for access control, the agent's
system prompt, and how the bot participates in group chats.

## Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `whitelist` | `string[]` | `[]` | Allowed chats/participants. Empty = answer everybody. |
| `blacklist` | `string[]` | `[]` | Denied chats/participants. Always wins over the whitelist. |
| `goal` | `string` | `""` | Optional objective prepended to the rendered system prompt as a `Goal:` block. Supports the same placeholders as `system_prompt`. Leave empty to skip. |
| `system_prompt` | `string` | **required** | The agent system prompt. Supports `{{date}}`, `{{time}}`, `{{now}}`, `{{tz}}`, `{{bot_name}}` variables. The server refuses to start without it. |
| `bot_name` | `string \| null` | `null` | The bot's display name (e.g. `"Kai"`). Used only as a fallback mention matcher. |
| `bot_mention_regex` | `string \| null` | `null` | A regex detecting when the bot is addressed in a group. Defaults to a case-insensitive whole-word `@?<bot_name>`. |
| `group_participation` | `"never" \| "mentioned" \| "judicious"` | `"mentioned"` | How the bot joins group conversations (below). |

Example:

```json
{
  "whitelist": ["<group-jid>@g.us"],
  "blacklist": [],
  "goal": "Be a warm, super helpful chat participant.",
  "system_prompt": "You are Kai, a friend in this WhatsApp group... Today is {{date}}. Current time {{time}} ({{tz}}).",
  "bot_name": "Kai",
  "bot_mention_regex": "(?i)(?<![a-z@])@?k[aā]i(?![a-z])",
  "group_participation": "judicious"
}
```

The `system_prompt` is the bot's whole personality — how it talks,
what it knows, when to stay quiet. Write it as a persona description
(a friend, a participant), not as "a helpful assistant": the model
will copy whatever tone you set here into every reply. Keep style
rules in the prompt itself (e.g. "short replies", "no markdown",
"match the group's slang") — they change the output far more than
any code-side default.

## System prompt variables

`system_prompt` supports these placeholders (resolved at startup with
`WAHABOT_TIMEZONE`, default `UTC`):

- `{{date}}` — e.g. `2026-09-02`
- `{{time}}` — e.g. `14:30`
- `{{now}}` / `{{datetime}}` — e.g. `2026-09-02 14:30 UTC`
- `{{tz}}` — the timezone name, e.g. `America/Santiago`
- `{{bot_name}}` — the `bot_name` field, e.g. `Kai`

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

The regex is applied as-is, so you control the variants. Add word
boundaries (`(?<![a-z@])…(?![a-z])`) so the name only counts as a
standalone word and does not fire inside ordinary words like
`kaizo` or `chikai`:

```json
"bot_mention_regex": "(?i)(?<![a-z@])@?k[aā]i(?![a-z])"
```

`(?i)` makes it case-insensitive (so `kAI` matches), and the
lookarounds require a word boundary before and after, so `kAI`,
`@kai` and `KAI!` are mentions but `kaizo` is not.

## Access control

`chat_allowed()` (in `src/wahabot/core/filters.py`) applies
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
