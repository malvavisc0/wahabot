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
   racing a cache turnover is still deduplicated.
2. **Staleness.** Messages timestamped before the bot process started
   are backlog (history resyncs, container restarts), not fresh chat;
   they are skipped so the bot never wakes up hours late and answers
   a conversation that moved on.
3. **Access control.** The chat must be whitelisted in the session
   config (`data/sessions/<session>.json`) — and not blacklisted.
   Unknown chats are ignored entirely.
4. **Address check** (groups only). The bot is a guest in groups and
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
   `judicious` also runs the agent on plain messages in whitelisted
   groups and lets it decide for itself whether to speak (see §3,
   staying silent).

   Direct messages skip the address check entirely — the person chose
   to talk to the bot, so it answers.

### What the agent sees

The message is wrapped in a small annotation envelope before entering
the agent:

```
[Sender Name] the actual text
[message id: false_<chat-jid>_<hash>@lid]
[quoting] Sender: "the text being replied to"
```

- The **sender name** (from WhatsApp pushnames) tells the model who's
  talking, so it can keep track of the humans in a multi-person chat.
- The **message id** is there so the model can reference the message
  later: quote it in `send_message(reply_to=…)`, or react to it.
- The **quoting** line appears when the message is a reply to an
  earlier one and carries what that earlier message said (and who said
  it) — including when the quoted message is the bot's own.
- Bracketed notes are metadata for the model, never to be repeated
  verbatim in replies.

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
  (`WAHABOT_MEMORY_TOKEN_LIMIT`, default 8000 tokens).
- The **system prompt is never in the buffer** — it is re-rendered
  from the session config at the start of every run (goal, role,
  style, current date/time), so trimming can never evict the bot's
  identity.
- Before each run the buffer is *sanitized*: dangling tool-call groups
  from crashed runs are repaired, alternation is enforced, so the LLM
  never sees a malformed history.
- Memory is **process-local**. A restart or a container rebuild wipes
  it: the bot restarts each conversation blank. The group's real
  history is still in WhatsApp, and the model can pull it back on
  demand with the `fetch_chat_messages` tool (see §4). Chats evicted
  from the LRU (the least recently used of 1000 chats) also restart
  blank.

This is a deliberate trade-off: privacy (nothing about group members
is persisted beyond the raw event journal) over long-term continuity.

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
  reply; the polite wave for greetings and jokes landing.
- **A tool round then a send** — the model fetches context first
  (history, a web search, a page read), then answers with it.

The loop has a round limit; if the model keeps calling tools without
concluding, the run is cut and the model is nudged to decide.

Silence is a first-class outcome: the system prompt explicitly forbids
narrating the decision ("I'll stay silent", "No response") — the bot
either says something real or says nothing at all.

---

## 4. Tools the model uses inside a conversation

| Tool | What it does |
|------|--------------|
| `send_message` | Sends a text (optionally quoting a message via `reply_to`, optionally @-mentioning people via `mentions`). One per run. |
| `stay_silent` | Ends the run without sending. |
| `react_to_message` | Emoji reaction to a message id. |
| `fetch_chat_messages` | Recent history of any chat as JSON (ids, senders, texts) — the model's window into conversations, including ones it just "woke up" in. |
| `search_messages` | Text search over a chat's recent history. |
| `get_chat` | Chat metadata and, for small chats, the participant list with JIDs and names — the source for mention ids. Names are read from the chat's recent messages (rosters in LID groups carry bare JIDs only), so the call costs one extra history fetch. |
| `forward_message` | Forwards a message to another chat. |
| `send_image` | Sends an image from a public URL. |
| `web_search`, `visit_url`, `fetch_current_stock_price`, `get_youtube_transcript` | The outside world: metasearch, page reads, tickers, video transcripts. |
| `run_shell_command` | Host shell (disabled by default; opt-in per deployment). |

Tool results come back as JSON envelopes — `{"ok": true, …}` or
`{"ok": false, "error": "…"}` — never as raised exceptions; failures
are data the model can react to. List tools trim themselves to whole
messages within a character budget and flag `truncated`, so the model
always reads parseable results and knows when to fetch more.

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
or from `fetch_chat_messages`. Quoting is how the bot keeps replies
attached to the right person in fast-moving group chats.

### @-Mentioning

`send_message(mentions=[<JID>], text="…@Name…")` produces a **real
mention**: the mentioned person's client highlights the message and
notifies them. The rule is a pair — every JID passed in `mentions`
must have its owner's display name written in the text as `@<name>`;
WhatsApp matches the two up. JIDs and names come from `get_chat`'s
participant list or from message history (`participant` fields).
Typing `@name` alone in the text is *not* a mention — no highlight,
no notification — which is why the tool description and the system
prompt both spell the pairing out for the model. When the model
passes `mentions` without any `@` in the text, the tool still sends
but returns a `warning` in its envelope saying nobody was notified,
so the model can correct itself. The bot mentions sparingly: only
when directing something at a specific person.

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
- **Bot forgot the conversation**: restart wiped process-local memory;
  the model rebuilds context via `fetch_chat_messages` or acts fresh.
- **Mention didn't notify**: missing `mentions` JIDs or a name in text
  that doesn't match the JID's owner — check the tool call arguments
  in the trace. A send whose text had no `@` at all comes back with a
  `warning` field in the tool envelope.
- **Participant list shows bare JIDs, no names**: the name lookup
  (recent-message `notifyName`s) failed — the chat history was
  unreadable; the debug log says why. The roster itself is fine.
