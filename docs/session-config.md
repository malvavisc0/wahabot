# Session Config (`data/sessions/<session>.json`)

Every WhatsApp session is configured by a JSON file at
`data/sessions/<session>.json` (the session name is `WAHABOT_SESSION`,
default `default`). wahabot loads it at startup via
`load_session_config()` and uses it for access control, the agent's
system prompt, and how the bot participates in group chats.

Manage these files with the CLI: `wahabot sessions init` writes a
starter config, `wahabot sessions list` shows existing ones, and
`wahabot sessions view` prints one with the system prompt rendered
(placeholders expanded, `# Goal` section prepended) as markdown —
`--plain` prints the prompt as raw text, `--raw` skips the
placeholder expansion.

## Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `whitelist` | `string[]` | `[]` | Allowed chats/participants. Empty = answer everybody. |
| `blacklist` | `string[]` | `[]` | Denied chats/participants. Always wins over the whitelist. |
| `goal` | `string` | `""` | Optional objective prepended to the rendered system prompt as a `# Goal` section. Supports the same placeholders as `system_prompt`. Leave empty to skip. |
| `memory` | `string` | `""` | Optional free-text durable facts the operator maintains, rendered as a `# Memory` section after `# Goal`. Supports the same placeholders as `system_prompt`; limited to 1024 tokens (the server refuses to start a config that exceeds it). Like `goal`, it is read through the hot reloader — edits apply to the next run without a restart. |
| `system_prompt` | `string` | **required** | The agent system prompt. Supports `{{date}}`, `{{time}}`, `{{now}}`, `{{tz}}`, `{{bot_name}}`, `{{host}}` variables. The server refuses to start without it. |
| `bot_name` | `string \| null` | `null` | The bot's display name (e.g. `"Kai"`). Used only as a fallback mention matcher. |
| `bot_mention_regex` | `string \| null` | `null` | A regex detecting when the bot is addressed in a group. Defaults to a case-insensitive whole-word `@?<bot_name>`. |
| `group_participation` | `"never" \| "mentioned" \| "judicious"` | `"mentioned"` | How the bot joins group conversations (below). |
| `reply_window_days` | `integer` | `7` | Recovered rows older than this many days become operator-required work and get no customer reply. Also bounds an operator relay: older customer chats hold the draft until the customer writes again. Bounds 1–30; hot-reloaded. |
| `close_after_days` | `integer` | `5` | Days a done ticket waits before it auto-closes; a customer message inside the window reopens it. Open or waiting tickets untouched for the same window also close. Bounds 1–365; hot-reloaded. |
| `pause_after_human_hours` | `integer` | `2` | Hours a human-operator message pauses the assistant in that chat. Bounds 0–168; `0` disables the pause. |
| `operator_language` | `string` | `""` | Language for operator alerts and confirmations. Empty falls back to English; confirmations also use the operator's detected language. |
| `timezone` | `string` | `""` | IANA zone for support hours and local-day summaries (e.g. `Europe/Madrid`). Empty falls back to `WAHABOT_TIMEZONE`. An unknown zone is rejected at load. |
| `hours` | `object` | `{}` | Opening hours keyed `mon`…`sun`, each `{"open": "09:00", "close": "17:00"}` or `{"closed": true}`. Empty means always open. An overnight window (`22:00`–`02:00`) is supported. |
| `away_message` | `string` | `""` | Customer-facing outside-hours text. `{ticket}` and `{next_opening}` are filled by code; empty uses the bundled wording. |
| `night_number` | `string` | `""` | After-hours contact for urgent work. It is only messaged after that number has written once; otherwise the alert notes the setup gap and never messages a stranger first. |
| `team_groups` | `string[]` | `[]` | Group chat ids that run as the internal team surface. Listed groups become mentioned-only and can never write tickets, message customers, use TTS or greet first. |
| `knowledge_only` | `boolean` | `false` | Opt-in answers-only support mode. When on, uncovered questions get "I don't know" plus a handover offer instead of open-domain answering. |
| `voice_replies` | `boolean` | `false` | When TTS is configured, answer a customer's voice note with a synthesized voice note. The web Settings page shows this only when TTS is available. |

Example:

```json
{
  "whitelist": ["<group-jid>@g.us"],
  "blacklist": [],
  "goal": "Be a warm, super helpful chat participant.",
  "memory": "The workshop meets on Thursdays. Never mention the Bridge Club in groups.",
  "system_prompt": "You are Kai, a friend in this WhatsApp group... Today is {{date}}. Current time {{time}} ({{tz}}).",
  "bot_name": "Kai",
  "bot_mention_regex": "(?i)(?<![a-z@])@?k[aā]i(?![a-z])",
  "group_participation": "judicious",
  "reply_window_days": 7,
  "timezone": "Europe/Madrid",
  "hours": {
    "mon": {"open": "09:00", "close": "17:00"},
    "tue": {"open": "09:00", "close": "17:00"},
    "wed": {"open": "09:00", "close": "17:00"},
    "thu": {"open": "09:00", "close": "17:00"},
    "fri": {"open": "09:00", "close": "17:00"},
    "sat": {"closed": true},
    "sun": {"closed": true}
  },
  "away_message": "Ticket {ticket}: we are away and reply after {next_opening}.",
  "knowledge_only": true,
  "voice_replies": false,
  "team_groups": ["<internal-team-jid>@g.us"]
}
```

`reply_window_days` bounds recovery of accepted messages. A recovered row
past the window becomes operator-required and the operator gets a notice;
a row older than `WAHABOT_INTAKE_RECOVERY_MINUTES` (30 minutes by default)
is still answered, but the reply starts with a short apology.

## Tickets and the operator desk

Every customer 1:1 chat is a ticket. The bot opens it silently, answers
when it can, and hands it to the operator only when a person must act.
The operator replies in the "Message yourself" chat: quote any alert (or
a forwarded attachment) with what was done, or name the ticket
(`142: refunded`). The bot drafts the customer message in the customer's
language, refuses a draft that adds an amount, date, time or reference
the operator did not write, and refuses one that drops an amount or
percentage the operator did write. The customer's language is detected
from their text; it drives the holding reply and the alert's Language
line. While a ticket waits on the operator, the customer gets one short
holding reply per hour (and another after a fresh alert). If WAHA does
not confirm an alert, the handover stays queued and the 60-second tick
reconciles it against the operator chat before retrying, so one decision
does not page twice.

Operator commands in the self-chat: `what's open?`, `close 142`,
`reopen 142`, `pause`, `resume`, `pause <name>`, `resume <name>`.
`pause` with no name pauses every chat in the session; `resume` clears
it. A human message typed in a customer chat pauses that chat for
`pause_after_human_hours`, and messages received while paused are saved
but never replayed. The pause is keyed by the chat, so it does not
create a stub contact for a customer who has not written yet.

Only one ticket per chat can be active at a time. A different issue is
split into a new ticket only when the current one is `open` or `done`;
a ticket that is waiting on the operator or the customer must be closed
by a person first, so that work is never detached.

Use `goal` for what the bot should aim for and `memory` for durable
facts it should know (people's preferences, standing decisions,
corrections). The model-facing `notebook` tool keeps its own per-chat
durable notes in `data/notebook.sqlite3`; `memory` is the one the
operator edits directly.

The `system_prompt` is the bot's whole personality — how it talks,
what it knows, when to stay quiet. Write it as a persona description
(a friend, a participant), not as "a helpful assistant": the model
will copy whatever tone you set here into every reply. Keep style
rules in the prompt itself (e.g. "short replies", "no markdown",
"match the group's slang") — they change the output far more than
any code-side default.

## Support hours

`hours` and `timezone` decide whether the business is open. Inside a
window the bot answers normally and alerts the operator. Outside it,
non-urgent handovers are deferred: the ticket becomes needs-you, no
individual alert is sent immediately, and the composed alert is queued
for delivery at the next opening. Urgent handovers still alert
immediately and additionally notify `night_number` when that contact is
ready. The customer is told when to expect a reply using
`away_message`, with `{ticket}` and `{next_opening}` filled in code.
Overnight windows and DST transitions are evaluated in `timezone`.

At the first opening of each local day the operator receives one
morning summary (yesterday's ticket counts and needs-you work, urgent
first, with ages). `last_summary_date` makes it once per local day and
restart-safe; the summary message id is stored so quoted ticket numbers
still route. The summary cancels the queued notice for every ticket it
lists and records a nudge, so the same opening cannot page twice.
Delivered deferred notices and nudges are stored as alert ids too, so
quoting one still routes to its ticket.

## Published answers (Teach)

The operator can publish question/answer pairs by replying to a ticket
alert with an explanation. The reply classifier marks it as an
`answer` (not a business action), sends the customer relay, and asks
`Should I answer this myself next time? Reply save N.` (N is the
ticket number). A bare `save` also works and targets the newest unsaved
candidate; a second `save` reports that there is nothing to save.

Enabled answers are injected as one whole `# Answers` section when the
set fits the 6000-token budget. A set that does not fit
is not pasted at all; the model searches it with `search_knowledge`.
Answers are reference material: they never outrank safety or the
session's Instructions, and returned hits are recorded in
`answer_uses`. Deleting an answer is a soft delete and the latest one
can be undone.

## Internal team groups

`team_groups` is an allowlist, not a mode. Each listed group is
additive: it overrides the session's group participation to
`mentioned` for that chat only, so unlisted groups keep their existing
`never`/`mentioned`/`judicious` behavior. A run in a listed group has
surface `team_group` and the restrictions are enforced in code:

- never open, update or close a ticket; read-only ticket facts are
  available (including open work for "what's open?");
- never send from the group run to a customer chat;
- no customer-system writes directly: a configured skill may stage a write
  for the operator, and a team-group write always needs the operator's own
  `yes`, even when the skill sets `writes_need_operator: false`;
- no TTS and no posting first. The run wakes only when the newest body
  names the assistant, quotes its message, or replies to it.

Group traffic is still journaled and chat-scoped notebook notes still
work; a group note never becomes a customer-chat note. Skills are
installed once for the deployment, not per session: their
`customer_chats` field controls whether customer and team-group runs may use
them (`read` or `read_write`). Customer writes stage unless
`writes_need_operator` is false; team-group writes always wait for the
operator. Operator commands can use every enabled skill whose test is current.

## System prompt variables

`system_prompt` supports these placeholders (resolved at startup with
`WAHABOT_TIMEZONE`, default `UTC`):

- `{{date}}` — e.g. `Wednesday 2nd September, 2026`
- `{{time}}` — e.g. `2:30 PM`
- `{{now}}` / `{{datetime}}` — e.g. `2026-09-02 14:30 UTC`
- `{{tz}}` — the timezone name, e.g. `America/Santiago`
- `{{bot_name}}` — the `bot_name` field, e.g. `Kai`
- `{{host}}` — a snapshot of the machine the bot runs on (OS, Python, Node, the shell `run_shell_command` uses, and the extra binaries the shell tool may use); not timezone-dependent

## Group participation

In **1:1 chats** the bot always replies. In **groups** the
`group_participation` mode decides when to wake the agent:

| Mode | Behavior |
|---|---|
| `never` | Never reply in groups. |
| `mentioned` (default) | Reply only when the bot is mentioned by name/regex **or** when someone replies to one of the bot's messages. |
| `judicious` | Unaddressed group messages go to the triage gate first. The gate sees the message, the last few turns, the chat's mute list and a bounded gate brief (assistant name, timezone, operator, participation mode, personal-life exception and current goal); only a participation verdict wakes the agent. Addressed messages always run. If the gate is unavailable, unaddressed groups stay silent. |

`wahabot sessions init` writes `never` because a fresh config has no
`bot_name`; the schema default (when the key is omitted) is `mentioned`.

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

Access control gates *chat participants*. **Operator commands**
(`wahabot tell`, or a `fromMe` message matching `bot_mention_regex`
sent to the bot's own self-chat) bypass it by design — the HMAC key
and physical control of the WhatsApp account are the operator
credential, and a command names its own targets; see
`docs/agent-workflow.md` → "Beyond the chat turn".

The operator's ordinary messages on the account (typed in a group or DM,
not the self-chat) are subject to the same whitelist as the chat they are
in; there they are remembered as the operator's, and naming the bot wakes
it — see `docs/conversations.md` §1a.
