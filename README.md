# wahabot

A WhatsApp agent that *actually thinks*. Built on the [WAHA](https://waha.devlike.pro) HTTP API, it runs a LlamaIndex function-calling workflow behind a FastAPI webhook and gives your bot the ability to search the web, check stock prices, pull YouTube transcripts, send images, react to messages, and more — all from a group chat, without the user saying "use a tool."

The bot loops: it reads the conversation, decides whether it needs more information, calls a tool, gets the result, thinks again, and only then replies. No hand-rolled orchestration. Just a clean event-driven pipeline with three steps.

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
                  ▼
            reply text for the chat
```

## Why not just a simple LLM wrapper?

Because the interesting part isn't calling an LLM — it's giving it *hands*.

wahabot ships **11 tools by default, 12 with the optional shell tool enabled** — seven WhatsApp tools (send text, send image, react, forward, read recent messages, get chat metadata, search messages) plus four external research tools (web search via `webserp`, page fetch with Chrome TLS fingerprinting, stock prices via `yfinance`, YouTube transcripts) and an optional host shell tool (off unless `WAHABOT_SHELL_TOOL=true`). The model picks which tool to call, the workflow executes it, feeds the result back, and lets the model decide if it needs more. It stops when the model is content.

Every tool returns a compact JSON envelope — `{"ok": true, ...payload}` on success, `{"ok": false, "error": "..."}` on failure — and never raises; a failed lookup just feeds an error envelope back to the model so it can adapt. The whole run is capped at 120 s so a pathological loop can't hold the webhook hostage.

## What it can see

The agent isn't blind. With `WAHABOT_VISION=true` (default) and a vision-capable model:

- **Photo messages** are downloaded from WAHA, attached to the LLM call as image blocks for that turn only, then discarded. Chat memory stays text-only — no megabyte payloads polluting your rolling buffer.
- **Bare image URLs in text** (e.g. a wikia derivative link) are sniffed out, fetched with `curl_cffi`'s Chrome TLS impersonation, and included as additional image blocks. The Content-Type header decides if it's actually an image.
- **Media albums** ("multi-image posts") arrive as a container + N individual images via WAHA's WEBJS internals. The handler groups them by `parentMsgKey` and waits for `expectedImageCount` before processing.
- **Bare links** to pages are fetchable by the agent itself via the `visit_url` tool.
- Oversized images (> 10 MB) are skipped gracefully. Download failures degrade to a text-only turn.

## Staying out of the way

The bot is selective about what it replies to:

- **Backlog filter** — messages sent before the server started, or older than 5 min, are dropped as WhatsApp replay backlog.
- **Group participation** — configurable per session: `never` (bot is silent in groups), `mentioned` (default — only when @-mentioned or replying to the bot), or `judicious` (agent runs on every group message but may return empty to stay silent).
- **Status & newsletters** — events from `status@broadcast` and `@newsletter` JIDs are rejected before reaching the agent. No nonsensical replies to Instagram cross-posted statuses.
- **Access control** — per-session whitelist/blacklist in `data/sessions/<session>.json`. Blacklist always wins. Both empty = answer everybody.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in values
```

Key env vars:

| Variable | Purpose | Default |
|---|---|---|
| `WAHABOT_LLM_MODEL` | OpenAI-compatible chat model with function calling | **required** |
| `WAHABOT_LLM_TIMEOUT` | Per-request LLM HTTP timeout (s; client retries disabled) | `60` |
| `WAHABOT_WEBHOOK_HMAC_KEY` | Shared secret matching WAHA's `hmac.key` | **required** |
| `WAHABOT_HOST` / `WAHABOT_PORT` | Webhook server bind | `0.0.0.0:8080` |
| `WAHABOT_LOG_LEVEL` | loguru level | `INFO` |
| `WAHABOT_SESSION` | WAHA session name | `default` |
| `WAHABOT_MEMORY_TOKEN_LIMIT` | Per-chat rolling memory budget | `8000` |
| `WAHABOT_VISION` | Enable image understanding | `true` |
| `WAHABOT_MAX_IMAGE_BYTES` | Per-image download cap | `10485760` |
| `WAHABOT_MAX_URL_IMAGES` | Max image-URLs to fetch per message | `2` |
| `WAHABOT_WEB_SEARCH_MAX_RESULTS` | Default web search results | `5` |
| `WAHABOT_WEB_SEARCH_TIMEOUT` | webserp subprocess timeout (s) | `30` |
| `WAHABOT_WEB_SEARCH_PROXY` | Optional proxy for webserp | — |
| `WAHABOT_SHELL_TOOL` | Enable shell tool (off by default; run unprivileged/sandboxed) | `false` |
| `WAHABOT_SHELL_TIMEOUT` | Shell command timeout (s) | `30` |
| `WAHABOT_SHELL_MAX_OUTPUT` | Max chars returned from a shell command | `2000` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Opt-in tracing (see below) | — |

## Quick start

```bash
uv sync
cp .env.example .env   # fill in WAHABOT_* + LLM values
uv run wahabot sessions init          # write a starter session config
uv run wahabot serve                  # start the webhook server
```

Point WAHA at the webhook in your session config:

```json
{"url": "http://host:8080/api/webhook", "events": ["message"], "hmac": {"key": "your-secret-key"}}
```

`wahabot sessions init` writes a starter `data/sessions/default.json` — edit it
to at least set a `system_prompt`:

```json
{
  "goal": "Be a witty friend in this WhatsApp group.",
  "system_prompt": "You are Kai, a witty friend in this WhatsApp group. Today is {{date}}. Keep it short — no markdown, match the group's energy.",
  "bot_name": "Kai",
  "bot_mention_regex": "(?i)(?<![a-z@])@?kai(?![a-z])",
  "group_participation": "judicious",
  "whitelist": [],
  "blacklist": []
}
```

The `system_prompt` is the bot's entire personality. Write it like you're describing a friend, not a service. The model will mirror whatever tone you set.

### Other commands

```bash
uv run wahabot version                                   # show the version
uv run wahabot config                                    # show WAHABOT_* env (secrets redacted)
uv run wahabot sessions list                             # list session configs
uv run wahabot sessions view [--name N] [--raw]          # show a config, prompt rendered
uv run wahabot serve [--host H] [--port P] [--reload]    # webhook server
```

## Startup logs

`wahabot serve` prints a short banner so one glance tells you what's running:
version + Python, session name, LLM model/endpoint, memory token ceiling, and
enabled features (vision / shell / langfuse). It then logs the WAHA session's
live identity, the loaded session config summary, tracing status, and the tools
the agent was built with:

```
Info: wahabot 0.0.4 (Python 3.14.6)
Info: Session: default
Info: LLM: gpt-4o-mini @ https://api.openai.com/v1
Info: Memory: 8000 token ceiling
Info: Features: vision, no-shell
Info: Webhook: http://0.0.0.0:8080/api/webhook/default
Info: WAHA session default is live as My Name (4917...@c.us)
Info: Loaded session config from data/sessions/default.json: 0 whitelisted, 0 blacklisted, group_participation=mentioned
Info: Agent ready with 11 tools: fetch_chat_messages, forward_message, ...
```

For a machine-readable dump of every `WAHABOT_*` value (secrets redacted) use
`uv run wahabot config`. A shell tool and Langfuse only show up when
`WAHABOT_SHELL_TOOL=true` / `LANGFUSE_*` keys are configured. The `Agent ready`
line lists every tool the model can call this session.

## Observability

Set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` and every agent turn is exported to [Langfuse](https://langfuse.com) — prompts, completions, token counts, latency, tool calls — as one session per WhatsApp chat. JIDs are masked before they leave the process. Without credentials, tracing is a no-op and nothing leaves.

## Architecture (deep dive)

The agent workflow lives under `src/wahabot/ai/` as a set of focused modules:

| Module | Role |
|---|---|
| `workflow.py` | The three-step `FunctionCallingAgentWorkflow`, `load_llm`, `build_agent` |
| `events.py` | `InputEvent` / `ToolCallEvent` |
| `context.py` | Sender tagging, reply-context rendering, `handle_message` entrypoint |
| `messages.py` | Message classification, `extract_text`, `image_media`, `is_replyable` |
| `history.py` | `sanitize_chat_history` (repair) + `trim_to_budget` (token budget) |
| `tools/whatsapp.py` | The seven WhatsApp tools |
| `tools/external.py` | Web, finance, YouTube & (opt-in) shell tool builders |
| `tools/schemas.py` | Pydantic parameter schemas for every tool |
| `tools/envelope.py` | The unified JSON envelope (`ok` / `error`) every tool returns |
| `tools/web_search.py` / `tools/visit_url.py` / `tools/url_images.py` / `tools/shell.py` | Web lookup, image-URL & shell tool functions |
| `tools/finance.py` / `tools/youtube.py` | Market data and transcript tools |
| `observability.py` | Langfuse export |

Before every LLM call, the chat history passes through two hygiene steps: **repair** (fixes dangling tool calls, orphan messages, trailing user turns that would make the API reject the payload) and **trim** (keeps the newest tail that fits the token budget, treating tool-call groups as atomic).

Memory is keyed by `(session, chat_id)` — each WhatsApp conversation gets its own continuous context. Tool results are stored as `role="tool"` messages so the model can reference them across the loop.

## Development

```bash
uv run ruff check --fix .      # lint
uv run ruff format .           # format
uv run basedpyright            # type check
uv run radon cc src -s         # complexity (no C+ blocks allowed)
```

## Docs

- [Agent workflow — full pipeline explanation](docs/agent-workflow.md)
- [Session config — fields, group participation, access control](docs/session-config.md)
- [WAHA identity fields — `from` / `participant` / `to` semantics](docs/waha-identity-fields.md)
- [WAHA albums — multi-image reassembly](docs/waha-albums.md)
- [WAHA broadcast sources — status/newsletter handling](docs/waha-broadcast-sources.md)
