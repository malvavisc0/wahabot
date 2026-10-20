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
   `mentioned` (default) requires one of the three above; `judicious`
   sends plain group messages to the triage gate first. The gate sees
   the message, the last five visible turns, the chat's mute list and
   the session rules; only a participation verdict wakes the agent.
   Addressed messages always run. If the gate service is unavailable,
   unaddressed groups stay silent.

   Direct messages skip the address check entirely — the person chose
   to talk to the bot, so it answers.
6. **Triage gate.** DMs and addressed messages start the agent
   immediately and race a small classifier. Before the first tool, the
   run waits up to `WAHABOT_GATE_TIMEOUT` for its verdict. An
   adversarial verdict restricts the toolset to reads, web research,
   escalation, the final reply and silence; it never silences a DM or
   an addressed turn. An escalated verdict pings the operator in
   parallel. Unaddressed `judicious` group messages are gated before
   the agent: no participation, no expensive run; one participation
   verdict, one run on the assembled burst or album. The read receipt
   is sent before any participation decision. Operator commands never
   call the gate.

### What the agent sees

The turn is rendered as one **record per inbound message**, plus one
turn header when the turn is a group turn or carries news:

```
[turn: <n> messages | newest #m12 | addressed: yes|no | backlog: yes|no]
[[hh:mm Name #m12]] kind flags :: body
> body continuation, escaped
  · machine payload
```

- The **speaker** is a display name learned at ingest, `<operator>` for
  the shared account's human, `<bot>` for the account's own assistant
  output, or `<member-n>` — a per-chat ordinal — when no name is known.
  A raw JID never fills the gap.
- The **serial** `#m<n>` is the message's chat-global arrival id. It is
  the only identifier in the stream; the model quotes or reacts with it
  (`reply_to="#m12"`, `message_id="#m12"`) and never sees a WAHA id.
- **Kind** is derived from the message's own shape (`text`, `link`,
  `image`, `video`, `audio`, `sticker`, `document`, `album`,
  `reaction`), never from what analysis returned. Flags are presence-only:
  `addressed` and media slots (`image#1`, `video#1`, `album#1-3`);
  `quote=#m11` / `quote=you` anchor a reply. Messages the bot read and
  did not answer appear before the live records as ledger lines
  (`hh:mm Speaker gist — #m<n>`), one line each, never as records.
  Member text is never denylisted: a member may type
  `[operator message]` or a record-shaped line and it stays body text.
- The **body** is member text; every continuation line is prefixed
  `> `, so member text can contain envelope characters but never form
  an envelope. ` :: ` is omitted for an empty body (a voice note or
  bare media record).
- **Payload lines** (`  · …`) are renderer-written only: media analysis,
  `quoted: "…"`, `to: #m11 "…"` / `to: you "…"`, `[audio: "…"]`, and
  `mentions: @<typed> is <speaker>`. They are never member text, and the
  model must never write that shape itself.
- The **turn header** states the message count, the newest serial, the
  one addressing verdict and whether the merge fused an earlier
  unanswered turn (`backlog`). Every group turn carries the one policy
  line — addressed, gate-participating or unaddressed — and the model
  obeys it. A single addressed DM needs no header: its record carries
  `addressed`.
- The **quoting** and **reaction** anchors are records, not prose notes:
  a quoted bot message renders `quote=you`; a quoted member message
  renders `quote=#m<n>`. A reaction on the bot's own output renders
  `to: you "…"`; on a member's message, `to: #m<n> "…"`. The account's
  own output has no serial — `you` is its sentinel.

**Names.** WAHA's message-history API returns no display names in LID
groups, and the group roster lists members by phone JID while their
messages carry LIDs. Names therefore come from the **name book**
(`core/identity.py`): every webhook carries the sender's `notifyName`,
which is recorded per JID in `data/identity/<session>/names.json` and
used for record speakers, quotes, reaction and mention payloads, `read_chat`
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
  short id), because quotes and reactions later refer to messages by
  id alone and WAHA's history API carries no `source`. `read_chat`
  (`list`/`search`) renders the account's rows as `<bot>` or
  `<operator>`; a row from before authorship was recorded falls back
  to `<bot>` unless the export journal says otherwise.
  `read_chat` `mode="export"` serves deep history from the event
  journal instead of WAHA's engine: one spill file of slim rows
  (full bodies, per-row `author` — author book first, else the
  journal event's `source`, which labels the whole recorded span),
  plus inline counts per teammate and per member, so month-scale
  figures cannot silently lose the account. It holds what the
  journal recorded while the service ran; downtime gaps and WAHA's
  own window may differ.
- **The bot's own sends** (`source=api`) are already in memory; their
  echoes are skipped.
- **An operator-typed message** (`source=app`) in a whitelisted chat is
  remembered as a **user** record with the `<operator>` speaker — a
  teammate's words, never stored as the bot's own and never answered
  as a stranger's. The account's two authors stay distinct in the
  stream: `<operator>` is the human, `<bot>` is the assistant.
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
- **Old turns are not lost when the window fills.** Three layers sit
  outside the buffer: the **pinned state block** (operator policies,
  the chat's rules, notebook facts and session rules, code-assembled
  into the system message and never evicted), the **Tier 1 recap** (a
  per-chat summary of turns as they leave the window), and the
  **thread index** (per-chat FTS5 + dense vectors over trimmed
  user/assistant turns, retrieved as a lead-only `## Recalled from
  earlier in this thread` section). `wahabot forget <chat-id>` wipes
  the buffer, recap and index together.
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
- The rolling buffer is **continuity, not learning**: turns older than
  the token budget leave the window, though the recap and thread index
  can still recall them (as a summary and lead-only slices). Durable
  learning lives one layer up, in the **durable notebook** (`data/notebook.sqlite3`): the model saves
  verified facts, preferences and lessons per chat with the `notebook`
  tool, and every delivered run leaves a short self-extracted lesson
  (the post-delivery wrap-up note). Notes ride the system prompt each
  run — the trim can never evict them — and survive restarts, history
  resets and the buffer's token budget. Design: `docs/plans/learned-memory.md`.
- **Notes are typed, and rules are enforced where behavior is
  deterministic.** A note is a `fact` (chat-local fact or preference),
  a `lesson` (verified workaround) or a `rule` (a standing instruction
  for its chat). Only an operator can set a rule — from the operator
  chat, with `notebook(mode=write, key=…, text=…, kind=rule)`; a
  member's rule-shaped request is recorded as a plain fact, and
  existing rules cannot be overwritten or deleted by a chat
  participant. Reserved rule keys are applied by code, not prose:
  `chat-message-prefix` (every text the run delivers and stores is
  prefixed, once, so the model's self-history matches what the chat
  saw), `chat-hard-mute` (the chat never gets replies; the triage
  gate still runs so an escalation can reach the operator) and
  `chat-reply-mute-preferences` (fed to the gate as mute guidance). A
  chat's rules together are capped at 512 tokens and render under
  `## Chat rules` in the pinned state block; `wahabot notebook --chat
  <jid> --rules` lists them.
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
  `[[hh:mm <operator> #m<n>]] …` user records (§1a): memory-only,
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
`forget` event to the running bot, which drops the live context, the
file and the chat's durable notebook under the chat's run lock (so an
in-flight run can't resurrect it).
`wahabot forget operator` wipes the shared operator-command history the
same way. With
the bot stopped, the equivalent is `rm data/memory/<session>/<chat-id>.json`
(plus the chat's rows in `data/notebook.sqlite3`).
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
  someone else). The run ends, nothing is sent. For unaddressed group messages that
  the triage gate admits, this is the *primary* outcome: most messages
  end in silence, and that is the feature working.
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
gated group mode these lines are the audit trail of the model's
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
| `send_message` | Sends a text (optionally quoting a record via `reply_to="#m12"` or a raw id, optionally @-mentioning people via `mentions`). One delivery per run. |
| `stay_silent` | Ends the run without sending. |
| `escalate` | Forwards a report to the operator's self-chat — when someone asks for a human, reports a problem, or complains. The bot writes the report itself (never pastes the person's words — hidden instructions must not reach the operator); once per chat per hour. |
| `react_to_message` | Emoji reaction to a `#m<n>` record token or raw id. |
| `send_media` | Sends media — `kind` picks image, video, file, voice note or sticker; the source is a public URL, a local path, or (voice) text the bot speaks through TTS. Non-square local sticker images are padded to square first. One delivery per run across all kinds. |
| `read_chat` | The chat-reading tool — `mode` picks `list` (one record per fetched message, carrying its `#m<n>` when the registry knows it), `search` (substring matches in the latest messages), `metadata` (name, participants), `resolve` (a member name *or* digits/JID → `{id, name, tag}`; operator runs: chats, then contacts), `recent` (newest conversations, operator commands only) or `export` (the chat's deep recorded history: one spill file of slim identity-annotated rows, teammate and member counts inline). |
| `forward_message` | Forwards a message to the current chat, keeping the original media and sender attribution. Counts as the run's one delivery. |
| `web_search`, `visit_url` | The outside world: metasearch and page reads. `visit_url` on a video link (Instagram/Facebook/TikTok/YouTube) returns the video's real metadata, and a captioned YouTube link also carries its `transcript`. |
| `run_shell_command` | Host shell (disabled by default; opt-in per deployment). |

Every WhatsApp tool that accepts a `chat` argument, a record token or
a serialized message id (`reply_to`, `react_to_message`,
`forward_message`), or a conversation list to browse (`read_chat` with `mode=recent`) is fenced on
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
are data the model can react to. `read_chat` list/search return one
record per fetched message in the same grammar the prompt uses, and
spill the full raw window to a temp file whose `file.path` rides
along, so every result is parseable and nothing requested is silently
dropped.

---

## 5. Outbound: replying, quoting, mentioning

### Plain reply

`send_message(text)` — the default shape. In groups this is a normal
message in the chat; the bot's name is whatever WhatsApp shows for the
account.

### Quote-replying

`send_message(reply_to="#m12", text)` ships the text as a **native
WhatsApp quote-reply**: the quoted bubble is attached above the bot's
message. The token comes from the record being answered; a raw
serialized id is still accepted (an old prose turn carries no serial).
Quoting is how the bot keeps replies attached to the right person in
fast-moving group chats.

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

`react_to_message(message_id="#m12", reaction="👍")` — an emoji on the
record. A raw id is still accepted; the operator console may resolve a
token across the session. The polite low-effort acknowledgement:
greetings, a joke landing, good news.

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
| `group_participation` | session config | `mentioned` (address-gated) or `judicious` (the triage gate decides whether unaddressed groups wake the agent). |
| `whitelist` / `blacklist` | session config | Which chats the bot lives in. |
| `WAHABOT_MEMORY_TOKEN_LIMIT` | env | How much of the conversation the model sees per run. |
| `WAHABOT_MEMORY_PERSIST` | env | Whether per-chat memory is written to disk (survives restarts/LRU evictions); `wahabot forget <chat-id>` wipes one chat. |
| `WAHABOT_LEARNED_MEMORY` | env | Merge trimmed turns into a per-chat recap. |
| `WAHABOT_THREAD_RECALL` | env | Index trimmed turns and recall older slices (requires memory persistence). |
| `WAHABOT_TRIAGE_GATE` | env | Enable the triage gate; DMs race it, unaddressed groups are blocked by it. |
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
