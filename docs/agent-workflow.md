# Agent Workflow

Every WhatsApp message that reaches whabot is answered by a LlamaIndex
**Workflow** that behaves like a function-calling agent. Instead of
wiring the agent together from a high-level framework, we build it
explicitly, step by step, following the official reference:

> <https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/>

The result is a small, inspectable pipeline with three moving parts, a
per-chat memory, and the ability to call tools whenever the model
decides they are needed — looping back and forth until the model is
ready to give a final, human-friendly answer.

## Where the code lives

The AI code is split into focused modules under `src/whabot/ai/`:

| Module | What it does |
|---|---|
| `workflow.py` | `FunctionCallingAgentWorkflow` (the `@step` methods), `load_llm`, `build_agent` |
| `events.py` | The workflow events (`InputEvent`, `ToolCallEvent`) |
| `context.py` | Reply-context rendering + the `handle_message` entrypoint |
| `messages.py` | Message classification and extraction (`extract_text`, `is_replyable`, …) |
| `history.py` | Chat-history repair (`sanitize_chat_history`) and budget trimming (`trim_to_budget`) |
| `schemas.py` | Explicit Pydantic parameter schemas for every tool |
| `tools.py` | The bundled WhatsApp tools |
| `web_search.py` / `visit_url.py` | Web lookup tools (webserp CLI, curl_cffi page fetch) |
| `finance.py` / `youtube.py` | Market data (yfinance) and YouTube transcript tools |

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
                  │  no tool calls                    │  tool calls
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
   `WHABOT_MEMORY_TOKEN_LIMIT` (default 8000) is kept; tool groups
   (assistant call + its `tool` replies) are atomic so a trim never
   splits them, and the newest user turn always survives. The system
   prompt lives outside the rolling buffer so the trim can never evict
   it; its token cost is accounted via `initial_token_count` (a prompt
   larger than the budget is clamped and logged, not fatal).

## The entrypoint

`handle_message(event, agent, ctx=None)` in `whabot.ai.context` is the
single function the handlers call. It:

1. reads the message body from `event.payload["body"]`;
2. attaches a small `[Message quoting] …` note when the message is a
   reply to an earlier one, so the model knows what is being quoted
   (see `reply_context` / `message_replies_to`);
3. runs the workflow — `await agent.run(input=user_msg, ctx=ctx)`;
4. returns the response text (`result.message.content`, stripped),
   which the handler sends back to the chat through WAHA.

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

Every tool returns a short status string (or an error message) rather
than raising — a failed call never crashes the workflow, it just feeds
the error back to the model.

Most tools take an optional `chat` argument: omit it to act on the
**current chat** (the one the incoming message came from), or pass a
JID to reach another group or person (e.g. `1234567890@g.us`,
`9876543210@c.us`).

### Tool inventory

| Tool | Params | WAHA endpoint | Purpose |
|---|---|---|---|
| `send_message` | `chat?`, `text` | `POST /api/sendText` | Send a text (current chat or elsewhere) |
| `react_to_message` | `message_id`, `reaction` | `PUT /api/reaction` | Emoji-react to a message (empty = remove) |
| `send_image` | `url`, `caption?`, `chat?` | `POST /api/sendImage` | Send an image from a URL |
| `fetch_chat_messages` | `chat?`, `limit?` | `GET /api/{session}/chats/{chatId}/messages` | Read recent chat messages as text |
| `get_chat` | `chat?` | `POST /api/{session}/chats/overview` | Chat metadata (name, participants, …) |
| `search_messages` | `query`, `chat?`, `limit?` | `GET /api/messages` (local filter) | Find recent messages by text / media |
| `forward_message` | `message_id`, `chat?` | `POST /api/forwardMessage` | Forward a message to a chat |

All tool implementations live in `src/whabot/ai/tools.py`; the
WAHA HTTP calls are in `src/whabot/core/waha.py`. Details in the
subsections below.

### Messaging

| Tool | Purpose |
|---|---|
| `send_message(text, chat=None)` | Send a text — current chat (omit `chat`) or another group/person |
| `send_image(url, caption="", chat=None)` | Send an image from a public URL (mimetype inferred from the URL extension), with an optional caption |
| `forward_message(message_id, chat=None)` | Forward an existing message (by serialized id) to a chat |

```python
send_message(text="Just replying here")  # current chat
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
| `fetch_chat_messages(chat=None, limit=20)` | Recent messages as text lines, each prefixed `[id:...]` (for react/forward); media-only messages render as `[mimetype] filename` |
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
external information. All follow the same convention: they return a
status string and never raise.

| Tool | Params | Source | Purpose |
|---|---|---|---|
| `web_search` | `query`, `max_results?` | `webserp` CLI | Metasearch (Google/DuckDuckGo/Brave/…) — no API key |
| `visit_url` | `url` | `curl_cffi` | Fetch a page's visible text with a real Chrome TLS fingerprint (avoids blocks) |
| `fetch_current_stock_price` | `ticker` | `yfinance` | Current price + day change for stock/ETF/crypto |
| `get_youtube_transcript` | `url` | `youtube-transcript-api` | Video captions as text (needs captions on; returns inline, truncated) |

Ticker normalization handles lowercase, `BTCUSD`/`BTC/USD` → `BTC-USD`.
`web_search` shells out to the `webserp` CLI (from the `webserp` package);
`visit_url` uses `curl_cffi` with a Chrome impersonation fingerprint. Both
honor `WHABOT_WEB_SEARCH_TIMEOUT` and `WHABOT_WEB_SEARCH_PROXY`.

## Adding custom tools

Tools are optional but easy to plug in:

```python
from llama_index.core.tools import FunctionTool
from whabot.ai.workflow import build_agent


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


agent = build_agent(settings, tools=[FunctionTool.from_defaults(add)])
```

To wire them in, pass the tools through `handlers.register_agent_handler`
by calling `build_agent(settings, tools=[...])` there. From then on, the
workflow runs any requested tool automatically.

## A few constraints worth knowing

- The LLM must be an OpenAI-compatible **chat completions** model with
  function calling (`is_chat_model=True`, `is_function_calling_model=True`
  in `load_llm`); the workflow constructor fails loudly if function
  calling is missing.
- Workflow runs have a timeout (120 s by default), so a runaway tool
  loop cannot hang the webhook forever.
- Tools run inside a `try/except`: a failing tool never crashes the
  workflow. It only feeds an error message back to the model, which can
  then decide what to do next.
