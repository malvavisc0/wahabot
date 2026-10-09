# How conversations work

This document explains, end to end, what happens between "a person
sends a WhatsApp message" and "the bot replies" — how the bot decides
whether to speak, what it sees of the conversation, how it remembers,
and how replies, quotes and mentions are produced. It is the reference
for anyone tuning prompts, debugging a silent bot, or wondering why
the bot answered the way it did.

```
WAHA ──webhook──▶ FastAPI ──dispatch──▶ handler pipeline ──▶ agent run ──▶ tools ──▶ WAHA
   (event JSON)   (HMAC check)   (dedup, staleness, ACL,     (LLM + tool     (send_message,
                                  address check)             loop) memory)     react, fetch, ...)
```

---

## 1. Inbound: from WhatsApp to the agent

### The event

Every chat activity arrives as a WAHA webhook event: a JSON payload
(`POST /api/webhook/{session}`) with the message text, sender, chat id,
and a raw `_data` blob (the WhatsApp engine's own record — where
mentions, quoted-message links and per-sender display names live).
Each event is HMAC-SHA512 signed; the bot rejects anything whose
signature does not match the configured key, and journals every
accepted event verbatim to `data/events/<session>/*.jsonl` — a
greppable, replayable record of everything the bot ever saw.

### The gates

Before the agent ever wakes up, the message passes through, in order:

1. **Deduplication.** WhatsApp redelivers events (connectivity, WAHA
   restarts, our own webhook 500s). Every message id is remembered for
   the redelivery window; duplicates are dropped silently. At the
   cache cap the oldest entries are evicted one by one — a redelivery
   racing a cache turnover is still deduplicated. The window meets the
   staleness bound (300 s) so the two guards overlap with no gap: a
   redelivery inside the window is dropped here, past it by
   staleness — never answered twice.
2. **Staleness.** Messages timestamped before the bot process started
   are backlog (history resyncs, container restarts), not fresh chat;
   they are skipped so the bot never wakes up hours late and answers
   a conversation that moved on.
3. **Self-chat command check.** A `fromMe` message where the account
   is on both ends (`from` is one of the bot's own JIDs and `to`, when
   present, is too — on LID-linked accounts WAHA reports the self-chat
   as `from=<phone>@c.us, to=<lid>@lid`) and the body matches
   `bot_mention_regex` is an **operator
   command** delivered from WhatsApp (the "message yourself" chat is
   the operator console): it bypasses the chat gates below and runs
   on the shared operator context (one rolling history across
   commands), and the reply comes back as a quote-reply in
   the same chat. A **voice note** works too: self-chat audio is
   transcribed before this check (the console is trusted — only the
   operator's own devices can post there), so a spoken "kai do x"
   runs like the typed command. The bot's own writes into that chat — command
   replies, escalations, session notifications, tool deliveries —
   are echo-tracked by id, so their `fromMe` bounce-backs never
   re-trigger the command path (a forwarded message or injected
   report cannot make the bot command itself).
4. **Access control.** The chat must be whitelisted in the session
   config (`data/sessions/<session>.json`) — and not blacklisted.
   Unknown chats are ignored entirely.
5. **Address check** (groups only). The bot is a guest in groups and
   must be addressed to speak:
   - someone **pill-mentions** it (a real WhatsApp `@` mention of the
     bot's JID — the `mentionedJidList` on the event, checked against
     both the bot's phone JID and its LID),
   - someone **quotes a bot message** (a reply to something the bot
     said), or
   - someone **types its name** (matched against `bot_mention_regex`,
     e.g. `kai`, `kAI`, `@kai` — word-bounded, so "skate" never wakes
     it).
   
   The config's `group_participation` mode loosens or tightens this:
   `mentioned` (default) requires one of the three above;
   `judicious` also runs the agent on plain group messages — any group
   that already passed the whitelist/blacklist gate — and lets it
   decide for itself whether to speak (see §3, staying silent).

   Direct messages skip the address check entirely — the person chose
   to talk to the bot, so it answers.

### What the agent sees

The message is wrapped in a small annotation envelope before entering
the agent:

```
[Name <jid>] the actual text                  (groups; [<jid>] when no name is known)
[Name] the actual text                         (direct messages)
[Name <own-jid>] [operator message] text      (the human operator, see §1a)
[message id: false_<chat-jid>_<hash>_<participant>]
[quoting] Name <jid>: "the text being replied to"
[mentions: @111 is Ana <111@lid>; @222 is the shared account (you and your operator)]
[you were addressed: …]
```

- The **sender tag** carries the display name *and* the sender's full
  JID in groups — the same string is the mention handle: to @-mention
  that person later, the model copies the `<jid>`'s user part into an
  `@<user-part>` token (see §5). When no name is known the tag degrades
  to `[<jid>]`; DMs keep the bare `[Name]`.
- The **message id** is there so the model can reference the message
  later: quote it in `send_message(reply_to=…)`, or react to it.
- The **quoting** line appears when the message is a reply to an
  earlier one and carries what that earlier message said and who said
  it. A quote of the shared account says which teammate wrote it:
  `you <jid>` (the bot), `your operator <jid>` (the human), or
  `this account, you or your operator <jid>` for messages sent before
  authorship was recorded.
- The **mentions** line appears when the message @-tags people (the
  first ten tags). WhatsApp writes a tag into the text as bare digits
  (`@111222333444555`), so without it the model cannot know who was
  tagged — or that the tag is the shared account itself.
- The **addressed** line states mechanically how the message reached
  the bot, and is precise because the account is shared:
  - `this message names you — it is for you` — the text matches
    `bot_mention_regex`;
  - `this message tags the shared account, which may mean you or your
    operator` — the account's JID is in `mentionedJidList`;
  - `this message replies to your message` / `… to a message your
    operator typed — it may be meant for him` / `… to a message from
    the shared account, written by you or your operator`;
  - `your operator, the human sharing this account, named you` — an
    operator-typed message that names the bot (§1a).
- The **reaction notes** (`[reaction X from Name <jid> to your
  message: …]`) name the reactor the same way. A reaction made by the
  bot's own account — the operator tapping on any linked device, or
  the echo of our own `react_to_message` — never folds.
- Bracketed notes are metadata for the model, never to be repeated
  verbatim in replies. Members cannot forge them: typed copies are
  broken by `ai/scrub.py`.

**Names.** WAHA's message-history API returns no display names in LID
groups, and the group roster lists members by phone JID while their
messages carry LIDs. Names therefore come from the **name book**
(`core/identity.py`): every webhook carries the sender's `notifyName`,
which is recorded per JID in `data/identity/<session>/names.json` and
used for sender tags, quotes, reaction and mention notes, `read_chat`
results and outgoing `@Name` mentions. The book is keyed by the
person's JID, not by chat — a name is captured from every chat the
account is in and surfaces wherever that JID appears (the same person,
the same push name, in any chat). The per-chat roster (chat
overview + recent messages, cached for one hour) still wins where it
has a name.

**JIDs.** A linked-device suffix (`<user>:41@lid`, a message or
reaction typed on WhatsApp Web) is stripped everywhere: it names a
device, not a person.

### 1a. The operator: one account, two teammates

The bot runs on a human's WhatsApp account, and that human (the
operator) keeps using it. WAHA marks everything the account sends
`fromMe`, and in the webhook puts the **account** in `from` and the
**chat** in `to` — the chat a `fromMe` message belongs to is its `to`.
The webhook's `source` field tells the two teammates apart: `api` is a
send through WAHA (the bot), `app` is the human on a phone or
WhatsApp Web (documented in `docs/openapi.json`; available in webhook
events only).

- **Authorship is recorded at send time** in the author book
  (`data/identity/<session>/authors.json`, keyed by the message's
  short id), because quotes, reactions and `read_chat` later refer to
  messages by id alone and WAHA's history API carries no `source`.
  `read_chat` (`list`/`search`) marks each `fromMe` message with
  `author`: `bot`, `operator`, or `unknown` (sent before recording).
- **The bot's own sends** (`source=api`) are already in memory; their
  echoes are skipped.
- **An operator-typed message** (`source=app`) in a whitelisted chat is
  remembered as a **user** turn,
  `[Name <own-jid>] [operator message] text` — a teammate's words,
  never stored as the bot's own and never answered as a stranger's.
- **The operator naming the bot** (`bot_mention_regex`) wakes it in
  that chat, with the same chat fence — even in `never` groups, since
  the operator is trusted. The turn is rendered as the operator's and
  carries the operator variant of the addressed note. (Tagging is
  different: a member tagging the account's JID wakes the bot, but the
  operator's tag names their own account and does not.)
- A `fromMe` message without `source` (another engine) is ignored:
  without authorship it could be the bot's own echo, and treating it
  as the operator could make the bot answer itself.
- A message in the **self-chat** that starts with the bot's name is an
  operator command instead (§1, gate 3).

WAHA must deliver the account's own messages for any of this:
subscribe the webhook to `message.any` (see `docs/install.md`).

### Images

When the model supports vision and the message carries an image (or
image URLs sniffed from its text), the images are downloaded, checked
against the size budget, and attached to the turn as image blocks. The
model sees the picture; oversized images are skipped with a note.

---

## 2. Memory: what the bot remembers of the conversation

Per chat, the bot keeps a rolling conversation in memory:

- Every turn — the annotated user message, the model's tool calls and
  their results, and what the bot ultimately said — is stored in a
  per-chat buffer, trimmed to the oldest-free token budget
  (`WAHABOT_MEMORY_TOKEN_LIMIT`, default 8000 tokens). A delivered
  delivery-tool pair (empty assistant tool-call + result envelope) is
  collapsed at run end into one plain assistant message holding the
  delivered content (sent text, reaction emoji, or a media/forward
  marker) — the retained history reads like the chat, not like the
  API scaffolding that produced it.
- The **system prompt is never in the buffer** — it is re-rendered
  from the session config at the start of every run (goal, role,
  style, current date/time), so trimming can never evict the bot's
  identity.
- Before each run the buffer is *sanitized*: dangling tool-call groups
  from crashed runs are repaired, alternation is enforced, so the LLM
  never sees a malformed history.
- Memory is **persistent**: at the end of each agent run (and after every
  memory-only fold) the per-chat buffer is written to
  `data/memory/<session>/<chat-id>.json` — plaintext, one file per chat,
  so a chat's history survives restarts, deploys and LRU evictions. A
  restart reloads a chat's memory lazily on its next message; an
  LRU-evicted chat (past 1000 live contexts) reloads from disk instead of
  starting blank. The load/save/restore implementation lives in
  `wahabot.core.persistence`; this section summarizes its behavior.
- Memory is **continuity, not learning**: everything older than the
  token budget is trimmed and forgotten — corrections, tone norms and
  facts included. The planned learned-memory layer (rolling summary +
  typed fact store injected into the system prompt) is specified in
  `docs/plans/learned-memory.md`.
- Memory **mirrors the chat**: what the model's self-history records as
  its own words is exactly what the chat saw. A reply that was never
  delivered — a leaked `stay_silent` token written as text instead of
  the tool call, an invented error payload, post-delivery chatter — is
  filtered from storage by the same visibility definition that guards
  delivery (`chat_visible_text`, applied in `remember`), so the
  self-history can never re-teach the model its own bugs. (The
  one-time purge script `scripts/purge_leaked_silence.py` removed the
  leaked tokens stored by the pre-fix handler.)
- Messages the **operator types on the shared account** are stored as
  `[Name <own-jid>] [operator message] …` user turns (§1a): memory-only,
  no run, unless the operator names the bot. A chat with no prior
  conversation (neither live nor on disk) folds nothing.
- **Old tool calls keep their shape.** Past the two most recent turns,
  history is squeezed: thinking blocks go, tool results keep only their
  verdict, and long tool-call argument values are cut to an 80-char
  preview — but every argument *key* stays. The model learns how to
  call its tools from its own replayed calls; an earlier squeeze that
  kept only `reason` taught it to send `{"reason": …}` alone. Calls
  stored in that shape are repaired from the message's original call
  on the next load.

### Wiping memory

`wahabot forget <chat-id>` wipes one chat's memory — it posts a signed
`forget` event to the running bot, which drops the live context and the
file under the chat's run lock (so an in-flight run can't resurrect it).
`wahabot forget operator` wipes the shared operator-command history the
same way. With
the bot stopped, the equivalent is `rm data/memory/<session>/<chat-id>.json`.
There is no retention TTL: memory is kept forever until wiped. `data/memory/`
carries the same privacy weight as `data/events/` — back it up and
exclude it the same way.

---

## 3. The run: deciding what to say

One inbound message that passes the gates triggers one **agent run**:
a loop of LLM calls in which the model can use tools, ending in one of:

- **`send_message`** — the bot texts the chat. This is the normal
  answer. The tool allows at most one send per run; after a send, the
  run is done (no follow-up monologues, no second thoughts).
- **`stay_silent`** — the model decides the message does not deserve
  an answer (banter, chatter it has nothing to add to, a question for
  someone else). The run ends, nothing is sent. In `judicious` mode
  this is the *primary* outcome: most messages end in silence, and
  that is the feature working.
- **`react_to_message`** — a low-effort emoji reaction instead of a
  reply; the polite wave for greetings and jokes landing. A final
  reply that is *just* one emoji (👋) gets the same treatment as a
  fallback: it lands as a reaction on the triggering message or is
  dropped as silence, never sent as text.
- **A tool round then a send** — the model fetches context first
  (history, a web search, a page read), then answers with it.

The loop has a round limit; if the model keeps calling tools without
concluding, the run is cut and the model is nudged to decide.

Every tool (sends, reaction, forward, silence, and the research tools
alike) takes a `reason`: one short sentence justifying the call,
written to the operator's log and never delivered to the chat. In
`judicious` mode these lines are the audit trail of the model's
restraint — greppable per chat, and a missing reason shows up as a
WARNING (or `reason: (model gave none)` on the workflow's per-call
line).

Silence is a first-class outcome: the system prompt explicitly forbids
narrating the decision ("I'll stay silent", "No response") — the bot
either says something real or says nothing at all. The same filters
treat a leaked `stay_silent` token written as text, an invented error
payload, or a lone emoji as non-answers, so none of them reach the
chat as a bogus reply.

---

## 4. Tools the model uses inside a conversation

| Tool | What it does |
|------|--------------|
| `send_message` | Sends a text (optionally quoting a message via `reply_to`, optionally @-mentioning people via `mentions`). One delivery per run. |
| `stay_silent` | Ends the run without sending. |
| `escalate` | Forwards a report to the operator's self-chat — when someone asks for a human, reports a problem, or complains. The bot writes the report itself (never pastes the person's words — hidden instructions must not reach the operator); once per chat per hour. |
| `react_to_message` | Emoji reaction to a message id. |
| `send_media` | Sends media — `kind` picks image, video, file, voice note or sticker; the source is a public URL, a local path, or (voice) text the bot speaks through TTS. Non-square local sticker images are padded to square first. One delivery per run across all kinds. |
| `read_chat` | The chat-reading tool — `mode` picks `list` (recent messages with ids, sender `name`, and `author` on account messages), `search` (substring matches in the latest messages), `metadata` (name, participants), `resolve` (a member name *or* digits/JID → `{id, name, tag}`; operator runs: chats, then contacts) or `recent` (newest conversations, operator commands only). |
| `forward_message` | Forwards a message to the current chat, keeping the original media and sender attribution. Counts as the run's one delivery. |
| `web_search`, `visit_url` | The outside world: metasearch and page reads. `visit_url` on a video link (Instagram/Facebook/TikTok/YouTube) returns the video's real metadata, and a captioned YouTube link also carries its `transcript`. |
| `run_shell_command` | Host shell (disabled by default; opt-in per deployment). |

Every WhatsApp tool that accepts a `chat` argument, a serialized
message id (`reply_to`, `react_to_message`, `forward_message`), or a
conversation list to browse (`read_chat` with `mode=recent`) is fenced on
chat-triggered runs: the current conversation is the only target
allowed. Cross-chat reach — messaging, forwarding to, or reading
another person or group — is reserved for operator commands
(`wahabot tell`, or a mention in the bot's self-chat), whose
instructions are the one trusted source of cross-chat intent. An
operator command has **no default chat**: it runs on the synthetic
`operator` context, so every chat tool needs an explicit JID and is
refused without one (previously the literal `operator` reached WAHA as
`chats/operator/…` and failed with a 500). A
participant asking the bot to deliver or snoop outside the chat gets
a tool refusal envelope, and a refusal never produces a delivery.
`read_chat`'s `mode=resolve` is the one scoped exception: a chat run
resolves names against the current chat's own participants (so it can
find a JID to @-mention), never the operator's contact book;
`mode=recent` stays operator-only outright.
The one exception is `escalate`, which has no aimable target at all:
it always lands in the operator's own self-chat, at most once per
chat per hour. A confirmed escalation also leaves a durable record in
the audit journal (`data/audit/<session>/<date>.jsonl`, status
`open`), which `wahabot escalations` lists after the chat
notification scrolled away.

Tool results come back as JSON envelopes — `{"ok": true, …}` or
`{"ok": false, "error": "…"}` — never as raised exceptions; failures
are data the model can react to. List tools slim messages inline
(bodies capped, raw `_data` noise dropped) and spill the full raw
window to a temp file whose `file.path` rides along, so every result
is parseable and nothing requested is silently dropped.

---

## 5. Outbound: replying, quoting, mentioning

### Plain reply

`send_message(text)` — the default shape. In groups this is a normal
message in the chat; the bot's name is whatever WhatsApp shows for the
account.

### Quote-replying

`send_message(reply_to=<message id>, text)` ships the text as a
**native WhatsApp quote-reply**: the quoted bubble is attached above
the bot's message. The ids come from the `[message id: …]` annotations
or from `read_chat` (`mode=list`). Quoting is how the bot keeps replies
attached to the right person in fast-moving group chats.

### @-Mentioning

A WhatsApp mention is a pair: the text contains `@<digits>` and the
send carries the matching JID in `mentions`. The model only writes the
text; delivery (`deliver_chat_text`, shared by the tool and the final
reply) builds the pair:

1. `@<digits>` tokens (6+ digits) are matched by user part against the
   chat roster — overview participants plus the last 50 senders — and
   become mention JIDs in whichever namespace the roster holds.
2. `@<digits>@lid` / `@<digits>@c.us` are rewritten to `@<digits>`
   first (WhatsApp shows only the `@<digits>` part as a tag; the
   server tail would stay visible).
3. `@Name` / `@First Last` is rewritten to `@<digits>` when exactly one
   other roster member has that full or first name in the name book;
   ambiguous or unknown names stay plain text. The shared account's
   own name is never resolved (tagging it notifies nobody).
4. A numeric token missing from the roster is checked against WAHA's
   LID→phone map (`GET /api/{session}/lids/{lid}`): LID groups list
   members by phone JID but speak by LID, so a member who has not
   spoken recently is still taggable.

Tokens that still match nobody are sent as plain text and reported in
the tool envelope's `warning`; the send acknowledgment never confirms
that WhatsApp notified anyone. `read_chat(mode="resolve")` returns the
`tag` to write for a member found by name or digits.

### Reactions

`react_to_message(message_id, reaction)` — an emoji on the message.
The polite low-effort acknowledgement: greetings, a joke landing,
good news.

---

## 6. Tracing a conversation

Every run is fully traced (Langfuse when configured): the annotated
prompt, each LLM round with model parameters and token usage, every
tool call with arguments and results (JIDs masked in exports), and the
terminal decision (send/silent/react). A trace with **no GENERATION**
observations and a ~10ms workflow span means the run died before
reaching the LLM — typically a misconfiguration, not a model
judgement. A generation whose input you can read is the exact prompt
the model saw, message for message.

---

## 7. Configuration knobs that shape conversations

| Knob | Where | Effect |
|------|-------|--------|
| `goal` | session config | The one-line purpose prepended to every system prompt. |
| `system_prompt` | session config | The bot's personality, style and speaking-bar rules (`{{bot_name}}`, date/time placeholders). |
| `bot_name` / `bot_mention_regex` | session config | What the group must type to address the bot. |
| `group_participation` | session config | `mentioned` (address-gated) or `judicious` (reads everything, self-decides). |
| `whitelist` / `blacklist` | session config | Which chats the bot lives in. |
| `WAHABOT_MEMORY_TOKEN_LIMIT` | env | How much of the conversation the model sees per run. |
| `WAHABOT_MEMORY_PERSIST` | env | Whether per-chat memory is written to disk (survives restarts/LRU evictions); `wahabot forget <chat-id>` wipes one chat. |
| `WAHABOT_VISION` | env | Whether image messages are shown to the model. |
| `WAHABOT_LLM_*` | env | Provider, model and sampling (temperature/top_p/top_k…). |

Session config is hot-reloaded per message: edit the JSON while the
bot runs and the next message uses the new prompt; a broken edit keeps
the last good config (and logs it) rather than crashing the bot.

---

## 8. Failure modes, briefly

- **Bot never answers**: check the gates — chat not whitelisted, not
  addressed (no mention/quote/name), stale timestamps, or duplicate
  suppression of a redelivered event. Traces confirm which.
- **Bot answers twice**: it can't — the one-send latch blocks a second
  send per run; a duplicate reply is two runs on one message, i.e. a
  dedup window that closed (see the journal for the double event).
- **Bot forgot the conversation**: memory now persists to
  `data/memory/`; a missing or corrupt file degrades to a blank start
  (the corrupt file is quarantined as `.bad` alongside). Wipe with
  `wahabot forget <chat-id>` if it should genuinely reset.
- **Mention didn't notify**: the token named nobody on the roster —
  check the send envelope's `warning` and the tool call in the trace.
  `@Name` resolves only for a unique, learned name.
- **Names show as bare JIDs**: the member has not written since the
  name book started; names are learned from webhooks, not fetched.
- **Operator messages missing from memory**: the WAHA webhook is not
  subscribed to `message.any`, or the chat is not whitelisted.
- **Identity confusion (bot vs operator vs members)**: the system
  prompt states the account's own JIDs (`{{own_jid}}`, `{{own_lid}}`,
  `{{own_identities}}`, filled from WAHA `get_me` at startup and on
  every session recovery); turns, quotes and `read_chat` say which
  teammate wrote each account message (§1a). When the identity is not
  yet captured, the prompt's identity lines are dropped rather than
  rendered stale.
