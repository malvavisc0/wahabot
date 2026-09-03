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
| `tools/url_images.py` | Image-URL sniffing from message text (curl_cffi fetch, Content-Type check) |
| `tools/finance.py` / `tools/youtube.py` | Market data (yfinance) and YouTube transcript tools |
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
`handlers.py` keeps one `Context` per **session/chat pair** and reuses
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

`handle_message(event, agent, ctx=None, image=None, settings=None, waha=None)` in
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
5. returns the response text (`result.message.content`, stripped),
   which the handler sends back to the chat through WAHA as a
   quote-reply to the triggering message (`reply_to`).

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
`image_blocks`, and `_with_image` injects them into a **copy** of the
newest user message for the first LLM call only — memory stays text-only,
so no megabyte payloads enter the rolling buffer and a tool-call loop
never resends the picture. The turn's text side is the caption, or
`(image)` when there is none. When `WAHABOT_VISION=false`, or a download
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
`[jid redacted]` before they leave the process (the session id itself is
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
**current chat** (the one the incoming message came from), or pass a
JID to reach another group or person (e.g. `1234567890@g.us`,
`9876543210@c.us`).

### Tool inventory

| Tool | Params | WAHA endpoint | Purpose |
|---|---|---|---|
| `send_message` | `chat?`, `text`, `reply_to?` | `POST /api/sendText` | Send a text (current chat or elsewhere); `reply_to` quotes a message; once per run |
| `stay_silent` | — | — | End the run with no reply at all |
| `react_to_message` | `message_id`, `reaction` | `PUT /api/reaction` | Emoji-react to a message (empty = remove) |
| `send_image` | `url`, `caption?`, `chat?` | `POST /api/sendImage` | Send an image from a URL |
| `fetch_chat_messages` | `chat?`, `limit?` | `GET /api/{session}/chats/{chatId}/messages` | Read recent chat messages (JSON `messages` list) |
| `get_chat` | `chat?` | `POST /api/{session}/chats/overview` | Chat metadata (name, participants, …) |
| `search_messages` | `query`, `chat?`, `limit?` | `GET /api/messages` (local filter) | Find recent messages by text / media |
| `forward_message` | `message_id`, `chat?` | `POST /api/forwardMessage` | Forward a message to a chat |

All tool implementations live under `src/wahabot/ai/tools/` (WhatsApp
tools in `whatsapp.py`, external tools in `external.py`); the
WAHA HTTP calls are in `src/wahabot/core/waha.py`. Details in the
subsections below.

### Messaging

| Tool | Purpose |
|---|---|
| `send_message(text, chat=None, reply_to=None)` | Send a text — current chat (omit `chat`) or another group/person; `reply_to` (a serialized message id) sends it as a native quote-reply |
| `send_image(url, caption="", chat=None)` | Send an image from a public URL (mimetype inferred from the URL extension), with an optional caption |
| `forward_message(message_id, chat=None)` | Forward an existing message (by serialized id) to a chat |

```python
send_message(text="Just replying here")  # current chat
send_message(text="exactly this", reply_to="false_1111@c.us_ABC")  # quote-reply
send_message(chat="1234567890@g.us", text="Hello team!")  # to a group
send_image(url="https://example.com/plot.png", caption="Q3 chart")
forward_message(message_id="false_1111@c.us_ABC")
```

### Reactions

| Tool | Purpose |
|---|---|
| `react_to_message(message_id, reaction)` | React with an emoji; empty `reaction` removes the bot's reaction |

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

```python
fetch_chat_messages(limit=10)  # read the current conversation
get_chat(chat="1234567890@g.us")  # group metadata
search_messages(query="invoice", chat="1234567890@g.us")
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
  `prepare_chat_history` resets `tool_rounds = 0` and `image_blocks`
  for exactly this reason — removing those lines silently leaks state
  across runs (e.g. the round budget shrinking run over run).
- **A shared `Context` refuses concurrent runs**
  (`ContextStateError: Cannot start a new run while context is already
  running`). The global `_agent_lock` in `handlers.py` is what stands
  between the bot and that exception — do not run two agent turns
  concurrently, even for different chats, without per-chat locking.

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
that is why `_system_token_count` clamps instead of letting it raise.
Because trimming is read-side, `_chat_history` re-trims and `aset`s
the buffer on every round; the store can be at most one tool-round
over budget when `_early_stopping_response` reads it via `aget_all()`.

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
(`_token_count`, `history.py`'s group logic) must check blocks first
and fall back to kwargs, or it will silently see zero.

## A few constraints worth knowing

- The LLM must be an OpenAI-compatible **chat completions** model with
  function calling (`is_chat_model=True`, `is_function_calling_model=True`
  in `load_llm`); the workflow constructor fails loudly if function
  calling is missing.
- Workflow runs have a timeout (120 s by default), so a runaway tool
  loop cannot hang the webhook forever.
- The tool loop is also bounded by `WAHABOT_TOOL_ROUND_LIMIT`
  (default 50): a model that keeps re-issuing tool calls — small
  models at low temperature can repeat the same call deterministically —
  is stopped after that many LLM→tool round trips. The counter resets
  at every run start.
- `send_message` (and `send_image` / `forward_message`) deliver **at
  most once per run**: after a successful send, further calls return an
  error envelope instead of sending, so a looping model cannot spam
  the chat even below the round limit. `react_to_message` is likewise
  bounded to one reaction per run.
- `stay_silent` is the explicit exit for "no reply": the system prompt
  tells the model to call it instead of writing an empty string (which
  small models tend to replace with narration like "I'll stay silent
  here — …"). As a last line of defense, `handle_message` drops
  replies that merely narrate a silence (`is_silence_narration`).
- A final text produced *after* a delivery tool succeeded is dropped,
  not stored or returned: small models pattern-complete their own last
  assistant text at low temperature, so the "answer" after
  `send_message` is usually a stale repeat, not a new reply. Research
  runs (no delivery tool) keep their final answer.
- List tools (`fetch_chat_messages`, `search_messages`) return
  *slimmed* messages (`slim_message`): WAHA's raw `_data` blob
  (~90% of the payload) is stripped before enveloping, so results stay
  valid JSON and small enough for the memory budget. Tool outputs are
  additionally capped at `MAX_TOOL_RESULT_TOKENS` chars.
- Tools run inside a `try/except`: a failing tool never crashes the
  workflow. It only feeds an error message back to the model, which can
  then decide what to do next.
