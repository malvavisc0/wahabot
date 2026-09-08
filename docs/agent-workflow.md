# Agent Workflow

Every WhatsApp message that reaches wahabot is answered by a LlamaIndex
**Workflow** that behaves like a function-calling agent. Instead of
wiring the agent together from a high-level framework, we build it
explicitly, step by step, following the official reference:

> <https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/>

The result is a small, inspectable pipeline with three moving parts, a
per-chat memory, and the ability to call tools whenever the model
decides they are needed — looping back and forth until the model is
ready to give a final, human-friendly answer.

## Where the code lives

The AI code is split into focused modules under `src/wahabot/ai/`:

| Module | What it does |
|---|---|
| `workflow.py` | `FunctionCallingAgentWorkflow` (the `@step` methods), `load_llm`, `build_agent` |
| `events.py` | The workflow events (`InputEvent`, `ToolCallEvent`) |
| `context.py` | Sender tagging, reply-context rendering + the `handle_message` entrypoint |
| `messages.py` | Message classification and extraction (`extract_text`, `image_media`, `is_replyable`, …) |
| `history.py` | Chat-history repair (`sanitize_chat_history`) and budget trimming (`trim_to_budget`) |
| `tools/whatsapp.py` | The bundled WhatsApp tools |
| `tools/external.py` | Web, finance, YouTube & (opt-in) shell tool builders |
| `tools/schemas.py` | Explicit Pydantic parameter schemas for every tool |
| `tools/envelope.py` | The unified JSON envelope (`ok` / `error`) every tool returns |
| `tools/web_search.py` / `tools/visit_url.py` | Web lookup tools (webserp CLI, curl_cffi page fetch) |
| `tools/url_images.py` / `tools/shell.py` | Image-URL sniffing (curl_cffi fetch) and the opt-in host shell |
| `tools/finance.py` / `tools/youtube.py` | Market data (yfinance) and YouTube transcript tools |
| `albums.py` | Album reassembly: container + images buffered into one agent turn |
| `video.py` / `vision.py` | Video frame extraction + turn anchor; image captions |
| `observability.py` | Opt-in Langfuse trace export (see below) |

## How the workflow works

A workflow is simply a set of steps with events flowing between them.
Ours has three steps:

```
StartEvent ──► prepare_chat_history ──► InputEvent
                                            │
                                            ▼
                                     handle_llm_input
                                            │
                  ┌─────────────────────────┴──────────┐
                  │  no tool calls                     │  tool calls
                  ▼                                    ▼
              StopEvent                        handle_tool_calls
                 │                                    │
                 │                                    └──► InputEvent (loop)
                 │
                 ▼
          reply text for the chat
```

Whenever a step produces an `InputEvent`, the workflow routes it back
to the LLM step — that is the loop that powers tool use. It keeps going
until the model returns an answer without asking for a tool.

### Step 1 — Prepare the conversation

Before anything else, we make sure the agent knows what the user just
said and what was said before:

- loads the chat's memory buffer (the first time a chat writes, a fresh
  buffer is created for it);
- adds the new user message to that memory and hands the full chat
  history to the next step.

### Step 2 — Ask the model

Now the LLM looks at the history and the available tools:

- it receives the tools and the conversation so far and produces its
  answer;
- the assistant response is stored in memory;
- if the model made **no** tool calls, the step finishes with a
  `StopEvent` carrying the response;
- if the model **did** request tools, a `ToolCallEvent` is emitted with
  those requests.

### Step 3 — Run the tools

The model's tool requests are carried out here:

- each request is looked up by tool name;
- unknown tools and raised exceptions are turned into `role="tool"`
  messages explaining what went wrong — the model can then adjust;
- successful outputs are stored in memory as `role="tool"` messages;
- a new `InputEvent` with the updated history is emitted, sending the
  model back to Step 2. The cycle repeats until no tools are needed.

## The events

| Event | Carries |
|---|---|
| `InputEvent` | `input: list[ChatMessage]` — the conversation history for the LLM |
| `ToolCallEvent` | `tool_calls: list[ToolSelection]` — the tools the model asked for |

The workflow also validates itself from the steps' type annotations.
If a step claims it can return `InputEvent | StopEvent`, that contract
is checked when the workflow is first built.

## Remembering the conversation

A workflow run only keeps context for as long as that run lives. Each
`agent.run()` with a **fresh** `Context` starts with empty memory, which
would make the bot forget everything between messages. To fix that,
`core/runs.py` keeps one `Context` per **session/chat pair** and reuses
it for every message from that chat in that session:

```python
ctx = _contexts.setdefault((event.session, chat_id), Context(agent))
reply = await handle_message(event, agent, ctx=ctx)
```

The context is keyed by both the WAHA session and the chat id, so each
WhatsApp chat in each session gets its own continuous conversation —
two sessions sharing the same chat id never bleed history into each
other.

Concretely: the webhook already validates that the only configured session
receives events, and replies are sent back via `event.session`. So today
only one session can be active; the `(session, chat)` key simply makes
the memory indirection correct and future-proof if multiple sessions are
ever served.

### Persistence

The per-chat buffer is not just process-local: at the end of every agent
run (and after every memory-only fold) `core.runs.persist_memory` writes
the run-end buffer to `data/memory/<session>/<chat-id>.json`, so a chat's
memory survives restarts and LRU evictions. The save points are:

1. **Message run end** — in `reply_with_agent`, right after
   `handle_message` returns and *before* the delivery early-return (a
   delivered reply exits inside the lock, so a save placed after would
   never run).
2. **Album run end** — in `deliver_album_reply`, same position.
3. **fromMe fold** — after `remember_own_message` in its locked block.
4. **Reaction fold** — after the note replacement in `reactions.py`,
   which now holds the chat's run lock (the WAHA fetch of the reacted-to
   message stays outside it).

Each load/save runs under the chat's run lock; a restore on a context miss
(`context_for`) is lazy — a chat reloads from disk on its first message
after a restart, and an LRU-evicted chat reloads instead of starting
blank. The load/save/restore implementation lives in
`wahabot.core.persistence`; `wahabot forget` wipes one chat's live
context and disk file together, under that chat's run lock so no in-flight run
can resurrect it.

### Backlog filter

WhatsApp redelivers undelivered messages when the WAHA session or the
phone reconnects, and WAHA forwards them as fresh `message` events —
without a guard, the bot answers hours-old replay. `is_stale` drops any
message sent before this process started or older than 300s; messages
with unknown timestamps still pass through.

### Memory hygiene and the token budget

Before each run reaches the LLM, the buffered history passes through two
`history.py` filters:

1. **Repair** (`sanitize_chat_history`) — a failed run can leave memory
   with a dangling user message, an assistant message advertising tool
   calls whose `tool` replies never arrived, or orphan `tool` messages.
   The OpenAI-compatible API rejects all of these, so the history is
   rebalanced first (at run start with `drop_trailing_user=True`, and
   again before every LLM call).
2. **Trim** (`trim_to_budget`) — the newest tail that fits
   `WAHABOT_MEMORY_TOKEN_LIMIT` (default 8000) is kept; tool groups
   (assistant call + its `tool` replies) are atomic so a trim never
   splits them, and the newest user turn always survives. The system
   prompt lives outside the rolling buffer so the trim can never evict
   it; its token cost is accounted via `initial_token_count` (a prompt
   larger than the budget is clamped and logged, not fatal).
3. **Cap** (`MAX_TOOL_RESULT_TOKENS`, 2000 chars) — tool outputs are
   truncated in `run_tool_call` before entering memory. A single
   oversized tool group defeats both trims: `trim_to_budget` keeps it
   ("a single group larger than the budget is kept"), then
   `ChatMemoryBuffer.get` re-trims with the real tokenizer, finds only
   that group, and drops everything but the tool message — the LLM
   request goes out with no user turn and the provider answers 400
   ("No user query found in messages").

## The entrypoint

`handle_message(event, agent, ctx=None, image=None, images=None, settings=None, waha=None)` in
`wahabot.ai.context` is the single function the handlers call. It:

1. reads the message body from `event.payload["body"]` and prefixes a
   **sender tag** — `[notifyName]`, falling back to the participant id —
   so the model can tell group members apart (the tag also persists in
   memory, giving history speaker identity for free);
2. appends the incoming message's serialized id as `[message id: …]`
   (`message_id_note`), so the model can quote or react to the very
   message it is answering via `send_message(reply_to=…)` /
   `react_to_message` without fetching ids first;
3. attaches a small `[quoting] Sender: "…"` note when the message is a
   reply to an earlier one, so the model knows what is being quoted
   (see `reply_context` / `message_replies_to`). The quoted sender
   renders as a display name: WAHA's `replyTo` snippet carries no
   `notifyName`, so `participant_names` resolves the JID against the
   group roster (cached per chat for an hour, fails soft to the bare
   id on any WAHA error);
4. runs the workflow — `await agent.run(input=user_msg, image_blocks=..., ctx=ctx)`;
5. returns `(reply, target)` — the run's final text and its run-scoped
   delivery holder (the `sent`/`reacted` latches), so the handler can
   tell a tool-delivered run from a text reply without touching
   run-internal state. The handler sends the text back to the chat
   through WAHA as a quote-reply to the triggering message
   (`reply_to`), unless a tool already delivered.

### Run concurrency and the run-scoped target

Runs in **different chats proceed in parallel**; runs in the **same
chat serialize** on that chat's run lock (`chat_lock` in
`core/runs.py` — they share one `Context` and one memory buffer).
Operator commands lock their own `("…", "operator")` key — they
serialize against each other (one shared history buffer) but never
queue behind (or block) any chat. The lock table is
bounded and evicts idle entries oldest-first, but **a lock that is
held or has a queued waiter is never evicted**: between `release()`
and the waiter resuming, `asyncio.Lock.locked()` is already `False`,
so a `_chat_lock_pending` in-flight count is the only witness —
evicting there would hand the next caller a fresh lock running
concurrently with the old waiter.

Everything a run needs — session, chat id, the once-per-run
`sent`/`reacted` delivery latches, and the operator arming flag —
lives in a `RunTarget` dataclass bound for the run's duration through
`contextvars` (`bind_target` / `current_target` in
`ai/tools/whatsapp.py`). The workflow engine creates every step task
(and each tool's `asyncio.to_thread` worker) inside the run's
context, so every tool of a run resolves **its own run's** target:
concurrent runs can never read each other's chat id, latches, or —
critically — the operator arming flag (`operator_run`), which is
what keeps cross-chat reach welded to the command that armed it.

LLM fan-out is bounded across all parallel runs by the agent's
`llm_semaphore` (`max_concurrent_llm`, default 4): a burst of
simultaneous chats queues at the endpoint instead of multiplying
provider cost without limit. The caption calls (image, album, video)
pass the same semaphore through, so vision traffic counts inside the
same budget.

### Images (vision)

When `WAHABOT_VISION` is `true` (default) and the model is vision-capable,
the handler downloads photo and sticker messages via
`WahaClient.download_media` (`image_media(event)` gates on
`_data.type in ("image", "sticker")` — videos/documents are ignored),
streaming with a `WAHABOT_MAX_IMAGE_BYTES` cap so an oversized image is
skipped instead of buffered. WebP payloads (stickers are animated webp)
are normalized to a static PNG first frame (`first_frame_png`) — vision
models take stills, not animations. The download happens after the
group-participation check, so unaddressed group images cost nothing. The
bytes pass through as `image`, the workflow carries them on the run as
`image_blocks`, and `with_image` injects them into a **copy** of the
newest user message for the first LLM call only — memory stays text-only,
so no megabyte payloads enter the rolling buffer and a tool-call loop
never resends the picture.

Because the pixels are one-shot, every downloaded image is also
**captioned up front** (`ai/vision.py`): one small vision call per
image returns a single sentence ("beer glass, foam shaped like a
bear"), stored as `image["caption"]`. The captioning happens at the
download sites in `handlers.py` — *before* the chat's run lock is taken — so
the extra LLM call never extends the serialized agent-run section.
`handle_message` weaves the caption into the user message text —
`(image shows: beer glass, foam shaped like a bear)`; the `shows`
framing keeps an instruction-like caption reading as a description of
pixels, not as sender text. The caption lives in the rolling buffer
like any other chat line, so round 2+ of the same run and later turns
referring back to the picture keep a concrete text anchor instead of
the model confabulating about an image it can no longer see. A failed
caption degrades to the bare `(image)` marker and never sinks the turn.
The turn's text side is the sender's caption, or the `(image…)` marker
when there is none. When `WAHABOT_VISION=false`, or a download
fails, the turn degrades to text-only.

### Image URLs in text (vision)

A bare link in the text ("look at this
`https://host/path/pic.png/revision/latest`") is not a media message, so
`wahabot.ai.tools.url_images` sniffs image URLs out of the body: any path
segment ending in an image extension qualifies (wikia-style derivative
paths included), up to `WAHABOT_MAX_URL_IMAGES` per message. Each URL is
streamed with the same Chrome TLS impersonation `visit_url` uses —
the **Content-Type header** decides whether it is an image (the wikia
example serves `image/webp` despite the `.png` path), and the
`WAHABOT_MAX_IMAGE_BYTES` cap aborts oversized bodies mid-stream.
Fetched images join the WhatsApp-attached ones as additional
`image_blocks` on the same first-LLM-call injection; failures are
logged and skipped, never fatal for the turn.

## Voice notes

When `WAHABOT_TRANSCRIBE_URL` points at a WhisperX service, voice notes
are transcribed to text **before** the agent run, so a note behaves
exactly like typed input (gating, memory, reply, silence filters all
apply unchanged) — with one exception: a voice note in the operator's
**self-chat** transcribes ahead of the gates, so a spoken "kai do x"
runs as an operator command (the console is trusted — only the
operator's own devices can post there; everywhere else transcription
stays behind the whitelist). `message_kind` maps the WEBJS voice-note type `ptt`
(and the audio-file type `audio`) to `audio`; `core/transcribe.py`
downloads the `media.url` bytes (capped by `WAHABOT_MAX_AUDIO_BYTES`),
POSTs them to `{transcribe_url}/transcribe` and joins the returned
segments. The empty `body` is replaced by `[voice note] <transcript>` —
handle_message just re-reads the body, so no agent signature changes.
The `[voice note]` prefix flags the text as imperfect ASR. Notes transcribe
outside the chat's run lock (in parallel with other chats' runs); transcription/download failures log a
warning and drop the message, keeping the seen marker so WAHA
redeliveries cannot trigger a retry storm. A bare note in a `mentioned`
group is skipped before any download. Empty `WAHABOT_TRANSCRIBE_URL`
disables the whole path.

## Beyond the chat turn: commands, reactions, session health

Three event flows sit outside the plain message → reply pipeline:

- **Operator commands** run the same agent over the operator's **own
  rolling history** (context key `"operator"`) — shared across all
  commands, LRU-evicted and persisted like a chat's, but touching no
  chat's memory; no whitelist, no group gating. The turn is prefixed
  `[operator command]`; the session prompt keys on it (deliver with
  `send_message(chat=…)` to the named target, resolve names with
  `resolve_chat`, browse recency with `recent_chats`). The shared
  run-scoped target binding points at the event's `from` ("operator"), so a bare
  `send_message` behaves like a DM. Two issuers reach this path:
  `command` events posted to the webhook by `wahabot tell`
  (`wahabot.commands.build_command_event`, signed with the same HMAC
  as real WAHA traffic), and **self-chat mentions** — a `fromMe`
  message in the bot's own "message yourself" chat matching
  `bot_mention_regex` (`self_command_instruction` in `messages.py`),
  which the message handler converts to a command event. The
  self-chat reply is quote-sent back into that chat
  (`send_self_reply`), and every bot-sent message that lands in the
  self-chat — command replies, escalations, session up/down
  notifications, tool deliveries aimed there — has its id recorded
  in the echo cache (`core/echoes.py`), so the `fromMe` echo event
  WhatsApp produces is never re-parsed as a fresh operator command.
  Without that mark, a prompt-injected escalation report (or a
  forwarded group message) reading "kAI …" would execute as a
  trusted command with cross-chat reach.
- **Reactions** (`message.reaction` events) to the bot's own messages
  are folded into that chat's memory as `[reaction 👍 from Sender to
  your message: "…"]` notes — context for the next turn, never an
  agent run. The `true_`/`false_` id prefix decides ownership before
  any WAHA fetch; one note per target message, latest wins. The
  memory fold holds the chat's run lock (the WAHA fetch stays outside
  it) and is persisted to the chat's memory file like any other
  fold — so a reaction to a bot message in an LRU-evicted chat
  reloads from disk instead of being dropped.
- **Session health** (`session.status` events): only `WORKING` is
  healthy. `status.py` seeds the flag from `GET /api/sessions/{session}`
  at startup, mutes message/command handling while unhealthy (before
  the seen-marker, so WAHA redelivery retries after recovery), and
  notifies the operator's own WhatsApp on transitions.

## LLM observability (Langfuse)

`wahabot.ai.observability` exports every agent run's LLM calls —
prompts, completions, token usage, model, latency — to
[Langfuse](https://langfuse.com) when credentials are configured.
Strictly opt-in and fail-soft: without both `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` (plus optional `LANGFUSE_BASE_URL` for the
region/self-hosted instance, read via `Settings` from `.env`),
`enable_langfuse` is a no-op and nothing leaves the process; when they
are set, `LlamaIndexInstrumentor` emits OpenTelemetry spans through the
global tracer provider and the Langfuse client ships them from
background threads.

Each turn is wrapped in `chat_trace_attributes(chat_id)`, which stamps
the OpenTelemetry context with a stable `wa:<chat_id>` session id and a
`wahabot` tag — so one WhatsApp chat shows up as one session in the
Langfuse UI, with each bot turn as a trace. WhatsApp JIDs are PII; the
export-stage `mask_otel_spans` hook rewrites span attributes to
`[jid redacted]` before they leave the process — all address forms:
`@c.us`, `@g.us`, `@lid` (linked-device identities) and `@broadcast`
(the session id itself is
exempt — masking it would collapse every chat into one anonymous
session). Credentials are checked
once with a best-effort `auth_check` (a failure warns but never disables
tracing), and an `atexit` flush covers the server's shutdown.

> **Group participation.** `handle_message` is only reached for a group
> message when the participation mode decides to wake the agent —
> `mention` (addressed to the bot by JID/@-mention or `bot_mention_regex`)
> or `judicious` (agent decides). In `judicious` mode an empty reply
> means "stay silent" and nothing is sent. See
> [`docs/session-config.md`](./session-config.md).

## The built-in tools

The agent ships with a set of WhatsApp tools, all bound to the runtime's
WahaClient and refreshed per message from the handler (each uses the
same mutable `(session, chat)` holder). They are registered via
`build_default_tools(waha, holder)` in `handlers.py`:

```python
agent = build_agent(
    settings,
    tools=build_default_tools(waha, send_tool_holder),
)
```

Every tool returns a compact JSON envelope rather than raising —
`{"ok": true, ...payload}` on success, `{"ok": false, "error": "..."}`
on failure (built once by `wahabot.ai.tools.envelope.ok` / `.error`, so
no tool hand-rolls JSON). A failed call never crashes the workflow; it
just feeds an `error` envelope back to the model.

Most tools take an optional `chat` argument: omit it to act on the
**current chat** (the one the incoming message came from). Passing a
JID to reach another group or person (e.g. `1234567890@g.us`,
`9876543210@c.us`) is **operator commands only** — every WhatsApp
tool runs through the `fenced_chat` gate (`whatsapp.py`), which
refuses a cross-chat target on any run that a chat message woke.
A chat participant asking the bot to DM, forward to, or read someone
outside the conversation gets an `error` envelope (logged at WARNING),
never a delivery. The fence opens for exactly one trusted channel:
operator commands (`wahabot tell` or a self-chat mention),
where the instruction itself names the target — and `resolve_chat` /
`recent_chats` (the contact roster and chat list) refuse to run at
all outside operator runs. The one sanctioned exception for chat runs
is `escalate`: it takes no `chat` parameter and always targets the
bot's own self-chat (the operator's "message yourself" chat), rate-
limited to one report per chat per hour.

Serialized message ids carry their chat's JID (`false_<jid>_<hash>`),
so ids are a second way to aim a tool elsewhere:
`fenced_message_id` (`whatsapp.py`) applies the same rule to
`react_to_message`, `send_message`'s `reply_to` and
`forward_message`'s source id — an id embedding any chat but the
current one is refused on chat runs (ids without a recognizable JID
pass through; WAHA validates those server-side). A malformed `chat`
value (no `@`, so neither the current chat nor a resolvable JID) gets
its own `not a valid chat id` error rather than the cross-chat
refusal.

### Tool inventory

| Tool | Params | WAHA endpoint | Purpose |
|---|---|---|---|
| `send_message` | `chat?`, `text`, `reply_to?`, `mentions?` | `POST /api/sendText` | Send a text (current chat, or operator-named target); `reply_to` quotes a message; `mentions` tags contacts; once per run (shared latch) |
| `stay_silent` | — | — | End the run with no reply at all (terminal: the workflow stops before executing it) |
| `escalate` | `report` | `POST /api/sendText` (to the bot's own chat) | Forward a report to the operator's self-chat — for "I want a human" requests, complaints, reports. No `chat` parameter (target is fixed); once per chat per hour (cooldown); writes the report itself, never pastes the person's words; refused on operator runs (a command already talks to the operator) |
| `react_to_message` | `message_id`, `reaction` | `PUT /api/reaction` | Emoji-react to a message (empty = remove); once per run |
| `send_image` | `url`, `caption?`, `chat?` | `POST /api/sendImage` | Send an image from a URL (probed pre-send; 404/410 refused); once per run (shared latch) |
| `send_file` | `url?`, `path?`, `caption?`, `filename?`, `chat?` | `POST /api/sendFile` | Send a document (PDF, etc.) from a URL (probed like `send_image`) or a local file; once per run (shared latch) |
| `fetch_chat_messages` | `chat?`, `limit?` | `GET /api/{session}/chats/{chatId}/messages` | Read recent chat messages (JSON `messages` list) |
| `get_chat` | `chat?` | `POST /api/{session}/chats/overview` | Chat metadata (name, participants, …) |
| `search_messages` | `query`, `chat?`, `limit?` | `GET /api/messages` (local filter) | Find recent messages by text / media |
| `forward_message` | `message_id`, `chat?` | `POST /api/forwardMessage` | Forward a message to a chat; once per run (shared latch) |
| `resolve_chat` | `name` | `GET /api/{session}/chats`, `GET /api/contacts/all` | Operator-only: resolve a person/group name to chat JIDs (exact match first, then substring; ≤5 candidates) |
| `recent_chats` | `limit?` | `GET /api/{session}/chats` | Operator-only: list the newest conversations (each `{id, name}`), for instructions that go by recency instead of name |

All tool implementations live under `src/wahabot/ai/tools/` (WhatsApp
tools in `whatsapp.py`, external tools in `external.py`); the
WAHA HTTP calls are in `src/wahabot/core/waha.py`. Details in the
subsections below.

### Messaging

| Tool | Purpose |
|---|---|
| `send_message(text, chat=None, reply_to=None)` | Send a text — current chat, or the operator-named target (`reply_to`, a serialized message id, sends it as a native quote-reply) |
| `send_image(url, caption="", chat=None)` | Send an image from a public URL (mimetype inferred from the URL extension), with an optional caption |
| `send_file(url=None, path=None, caption="", filename=None, chat=None)` | Send a document (PDF, etc.) — from a public `url` (WAHA downloads it) or a local `path` for files the agent created (base64, capped at `WAHABOT_MAX_FILE_BYTES`); mimetype and filename inferred from the extension |
| `forward_message(message_id, chat=None)` | Forward an existing message (by serialized id) to a chat |

A URL passed to `send_image`/`send_file` is probed first
(`probe_media_url`): malformed or non-http(s) links and definitive
404/410 responses are refused (models sometimes invent media URLs),
while connection/timeout errors only warn and let WAHA try — its
network path may succeed where the probe's failed. The session prompt
forbids invented URLs outright.

In all four, `chat` is operator-commands-only; on chat runs the fence
refuses any target other than the current conversation. `reply_to` and
`message_id` are id-fenced the same way: an id from another chat is
refused (operator runs excepted).

```python
send_message(text="Just replying here")  # current chat
send_message(text="exactly this", reply_to="false_1111@c.us_ABC")  # quote-reply
send_message(chat="1234567890@g.us", text="Hello team!")  # to a group
send_image(url="https://example.com/plot.png", caption="Q3 chart")
send_file(url="https://example.com/paper.pdf", caption="the paper")
send_file(path="/tmp/report.pdf", chat="1234567890@g.us")  # a file it created
forward_message(message_id="false_1111@c.us_ABC")
```

### Reactions

| Tool | Purpose |
|---|---|
| `react_to_message(message_id, reaction)` | React with an emoji; empty `reaction` removes the bot's reaction. The id is fence-checked: on chat runs it must belong to the current conversation |

```python
react_to_message(message_id="false_1111@c.us_ABC", reaction="👍")
react_to_message(message_id="false_1111@c.us_ABC", reaction="")  # remove
```

Underneath it calls WAHA `PUT /api/reaction` (see
`docs/openapi.json` → `/api/reaction`); the client method is
`WahaClient.send_reaction`.

### Reading context

| Tool | Purpose |
|---|---|
| `fetch_chat_messages(chat=None, limit=20)` | Recent messages as a JSON `messages` list, each entry carrying its serialized `id` (for react/forward), body, sender and media info |
| `get_chat(chat=None)` | Chat metadata summary (name, participant count + JIDs, …) via `/chats/overview` |
| `search_messages(query, chat=None, limit=20)` | Find recent messages containing a text substring |
| `resolve_chat(name)` | Operator-only: resolve a person/group name to chat JIDs — chats first, contacts as fallback; the answer to "send it to *Familia*" |
| `recent_chats(limit=10)` | Operator-only: the newest conversations as `{id, name}` pairs; the answer to "summarize my latest 5 chats" |

```python
fetch_chat_messages(limit=10)  # read the current conversation
get_chat(chat="1234567890@g.us")  # group metadata
search_messages(query="invoice", chat="1234567890@g.us")
resolve_chat(name="Familia")  # → matches: [{id, name}, …]
recent_chats(limit=5)  # → chats: [{id, name}, …] newest first
```

> WAHA's `GET /api/messages` requires a `chatId`, so `search_messages`
> always scopes to one chat and filters recently fetched messages
> locally (`WahaClient.search_messages`) — results cover the recent
> window, not arbitrary old messages.
> `get_chat` uses `POST /api/{session}/chats/overview` since the spec
> offers no plain `GET .../chats/{chatId}`.

### External research

Beyond the WhatsApp tools, the agent ships lookup tools for up-to-date
external information. All follow the same convention: they return the
JSON envelope (`{"ok": ...}`) and never raise.

| Tool | Params | Source | Purpose |
|---|---|---|---|
| `web_search` | `query`, `max_results?` | `webserp` CLI | Metasearch (Google/DuckDuckGo/Brave/…) — no API key |
| `visit_url` | `url` | `curl_cffi` | Fetch a page's visible text with a real Chrome TLS fingerprint (avoids blocks) |
| `fetch_current_stock_price` | `ticker` | `yfinance` | Current price + day change for stock/ETF/crypto |
| `get_youtube_transcript` | `url` | `youtube-transcript-api` | Video captions as text (needs captions on; returns inline, truncated) |

Ticker normalization handles lowercase, `BTCUSD`/`BTC/USD` → `BTC-USD`.
`web_search` shells out to the `webserp` CLI (from the `webserp` package);
`visit_url` uses `curl_cffi` with a Chrome impersonation fingerprint. Both
honor `WAHABOT_WEB_SEARCH_TIMEOUT` and `WAHABOT_WEB_SEARCH_PROXY`.

## Adding custom tools

Tools are optional but easy to plug in:

```python
from llama_index.core.tools import FunctionTool
from wahabot.ai.workflow import build_agent


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


agent = build_agent(settings, tools=[FunctionTool.from_defaults(add)])
```

To wire them in, pass the tools through `handlers.register_agent_handler`
by calling `build_agent(settings, tools=[...])` there. From then on, the
workflow runs any requested tool automatically.

## Engine semantics that the design relies on

These behaviors of the underlying engine were verified against the
installed source, and the workflow's safeguards depend on them. Read
this before touching `workflow.py` or `handlers.py`.

### The engine is the `workflows` package, not `llama_index.core.workflow`

`llama_index.core.workflow` only re-exports the standalone
[`workflows`](https://github.com/run-llama/workflows) package
(`inspect.getfile(Workflow)` → `workflows/workflow.py`). When debugging,
read *that* source — the llama-index copy is a shim.

### `Context.store` persists across runs on a shared `Context`

`Context.to_dict()` serializes the **global state store, event queues,
buffers, and broker log**, and `_workflow_run` builds the next run from
that snapshot. Reusing a per-chat `Context` (as `handlers.py` does)
therefore carries `memory`, `tool_rounds`, and `image_blocks` into the
next run. Two consequences:

- **Any per-run value in the store must be reset at run start.**
  `prepare_chat_history` resets `tool_rounds = 0`, `last_tool_calls`,
  and `image_blocks` for exactly this reason — removing those lines
  silently leaks state across runs (e.g. the round budget shrinking
  run over run).
- **A shared `Context` refuses concurrent runs**
  (`ContextStateError: Cannot start a new run while context is already
  running`). The per-chat run locks in `handlers.py` are what stand
  between the bot and that exception — two turns in the SAME chat must
  never run concurrently (they share the `Context` and its memory
  buffer); turns in different chats hold no shared `Context` and run
  in parallel. Commands share one `Context` under the `"operator"`
  key and serialize on its lock, so they never contend with chat
  runs — only with each other.

The store is a `DictState` (a Pydantic model that shoves undeclared
keys into a `_data` dict), which is why heterogenous values — a
`ChatMemoryBuffer` object, an int, a list — coexist without a schema.
`store.set(path, value)` is a single-path write under a write lock.

### Step dispatch is event-type routing; termination is structural

The control loop is a reducer: a step's returned event is published,
and the step whose accepted type matches runs next. The tool loop is
literally `handle_llm_input → ToolCallEvent → handle_tool_calls →
InputEvent → handle_llm_input`. The round limit works *because* of
this: at the limit the step returns `StopEvent` instead of
`ToolCallEvent`, so nothing is published that `handle_tool_calls`
accepts — **no flag the model can talk its way past**. The 120 s
workflow timeout is enforced independently by a broker timeout tick
(`WorkflowTimeoutError`), so the two bounds stack: 50 rounds *or*
120 s, whichever comes first.

### `ChatMemoryBuffer` trims on read, not on write

`aput`/`aset` only append/replace in the chat store — **nothing is
trimmed at write time**. The token trim lives in
`ChatMemoryBuffer.get`, which also raises
`ValueError("Initial token count exceeds token limit")` if the
`initial_token_count` (our system-prompt cost) exceeds the budget —
that is why `system_token_count` clamps instead of letting it raise.
Because trimming is read-side, `chat_history` re-trims and `aset`s
the buffer on every round; the store can be at most one tool-round
over budget when `wrap_up_response` reads it via `aget_all()`.

Two of `get`'s behaviors shape the design:

- Its tokenizer counts `" ".join(str(m.content))` — real tokens, not
  our chars≈tokens estimate — so a history that passed
  `trim_to_budget` can still be over budget here. `get` then keeps
  dropping oldest messages until only the newest survives; if that is
  a `tool` message, the request has no user turn (the 400 above).
  `MAX_TOOL_RESULT_TOKENS` exists so no single group can force this.
- It never starts history on `assistant`/`tool` — it drops extra
  leading messages to avoid it — which compounds the collapse above.

### Tool calls live in two places on a message

LlamaIndex carries tool calls as `ToolCallBlock` objects in
`message.blocks` (modern path) or `additional_kwargs["tool_calls"]`
(legacy path), with blocks taking precedence. `ToolCallBlock` fields
are `tool_call_id` / `tool_name` / `tool_kwargs` — there is **no**
`arguments` attribute. Anything counting or inspecting tool calls
(`token_count`, `history.py`'s group logic) must check blocks first
and fall back to kwargs, or it will silently see zero.

## A few constraints worth knowing

- The LLM must be an OpenAI-compatible **chat completions** model with
  function calling (`is_chat_model=True`, `is_function_calling_model=True`
  in `load_llm`); the workflow constructor fails loudly if function
  calling is missing.
- Two library boundaries in `load_llm`/`ObservableOpenAILike` are
  external constraints, not design choices. **Sampling-parameter
  routing**: `top_p` and `presence_penalty` are first-class OpenAI SDK
  parameters and ride `additional_kwargs` (merged straight into the API
  request body), but `top_k`, `min_p` and `repetition_penalty` are not —
  the SDK's typed `create()` signature rejects them with a `TypeError`
  before any request is sent, so they ride `extra_body`, which the SDK
  forwards verbatim in the JSON body for OpenAI-compatible providers
  that do accept them. **Instrumentation payload**: the OTel llama-index
  instrumentor reads `model_dict["model"]` and
  `model_dict["temperature"]` for the `gen_ai.request.*` span
  attributes, but the base `OpenAILike.to_payload` only exposes metadata
  (`model_name`, no temperature) — leaving both as `None` and spamming
  OTel "Invalid type NoneType" warnings per LLM call.
  `ObservableOpenAILike.to_payload` exists solely to add those two keys.
- Workflow runs have a timeout (`WAHABOT_RUN_TIMEOUT`, 120 s by
  default; 0 disables it), so a runaway tool
  loop cannot hang the webhook forever.
- The tool loop is also bounded by `WAHABOT_TOOL_ROUND_LIMIT`
  (default 50): a model that keeps re-issuing tool calls — small
  models at low temperature can repeat the same call deterministically —
  is stopped after that many LLM→tool round trips with one final
  tool-free wrap-up call (logged with its trigger: the round limit, or
  a non-delivery round after a completed delivery). The counter resets
  at every run start.
- The workflow's delivery set (`DELIVERY_TOOLS` in `workflow.py`) is
  `send_message`, `send_image`, `forward_message`, and `react_to_message`.
  They share a single delivery latch and collectively deliver **at most
  once per run**: after a successful send, further delivery calls return
  an error envelope instead of sending, so a looping model cannot spam
  the chat even below the round limit. `react_to_message` is likewise
  bounded to one reaction per run. `send_file` is **not** part of this
  set: it enforces its own once-per-run latch in the run-scoped target
  (a second call errors out), but the workflow treats its result as an
  ordinary tool result — no delivery gate, no collapse, no
  post-delivery text drop.
- `stay_silent` is the explicit exit for "no reply": the system prompt
  tells the model to call it instead of writing an empty string (which
  small models tend to replace with narration like "I'll stay silent
  here — …"). It is **terminal**: the workflow stops the run as soon as
  the call appears, without executing it or looping the result back
  into another LLM round — a follow-up model response cannot leak text
  into the chat. A batch mixing `stay_silent` with other calls still
  ends silently: nothing else in the batch runs. As a last line of
  defense, `handle_message` drops replies that merely narrate a silence
  (`is_silence_narration`).
- A final text produced *after* a delivery tool succeeded is dropped,
  not stored or returned: it never reached the chat (the one-delivery
  latch already fired), and memory mirrors the chat. What the chat did
  see is preserved by collapsing the delivery tool group into one plain
  assistant message holding the delivered content — the sent text for
  `send_message`, the emoji for `react_to_message`, a bracketed marker
  (with caption) for `send_image`/`forward_message` — so the
  model keeps sight of what it already said instead of re-answering.
  The wrap-up call after a completed delivery is subject to the same
  drop. Research runs (no delivery tool) keep their final answer.
- Two structural loop breaks bound a degenerate run before the round
  limit: **a repeated identical tool call ends the run immediately**
  (consecutive rounds re-issuing the same call name+arguments mean the
  tool already answered "you already did that"), and **once a delivery
  has fired, only delivery-tool rounds may continue** — a legitimate
  "reply, then react" still works, but non-delivery rounds wrap the
  turn up (with the wrap-up answer dropped as post-delivery chatter).
  When a model emits research and delivery calls in one batch, research
  calls run first so no non-delivery operation occurs after delivery.
  In both wrap-up cases the assistant message advertising the
  never-executed calls is **not stored**: the chat API requires every
  advertised tool call to have a matching tool response, so storing it
  would poison the next run's history with an invalid turn.
  The sampling defaults (temperature 1.0, top_p 0.95 — the model card
  values, see `Settings`) make the deterministic repeat unlikely in
  the first place; these two breaks are the hard guarantee behind it.
- `is_silence_narration` also drops post-delivery chatter ("I already
  reacted to that message, so I'm done here.") and `is_error_narration`
  drops invented API-error payloads (e.g. a made-up
  `{"error": {"code": "resource_exhausted", …}}` naming a provider the
  bot never used) — model-authored noise never reaches the chat.
- List tools (`fetch_chat_messages`, `search_messages`) return
  *slimmed* messages (`slim_message`): WAHA's raw `_data` blob
  (~90% of the payload) is stripped before enveloping, so results stay
  valid JSON and small enough for the memory budget. Tool outputs are
  additionally capped at `MAX_TOOL_RESULT_TOKENS` chars.
- Tools run inside a `try/except`: a failing tool never crashes the
  workflow. It only feeds an error message back to the model, which can
  then decide what to do next.
