# wahabot

A WhatsApp bot that answers chats with an LLM agent — one that *actually thinks*. It reads the conversation, decides what it needs, calls tools, reads the results, and only then replies. Built on the [WAHA](https://waha.devlike.pro) HTTP API behind a small FastAPI webhook.

It hears voice notes. It sees photos. It searches the web, checks stock prices, pulls YouTube transcripts, sends documents, reacts with emoji, and stays quiet when it has nothing to add — all without anyone saying "use a tool."

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

## Giving the model hands

The interesting part isn't calling an LLM — it's what the LLM can *do*.

wahabot ships **14 tools**: nine WhatsApp tools (send text, send images, send files, react, forward, read recent messages, chat metadata, message search, resolve a name to a chat) plus five external ones (web search, page fetch with Chrome TLS fingerprinting, stock prices, YouTube transcripts, and an opt-in host shell). The model picks the tool, the workflow executes it, feeds the result back, and the model decides whether it needs another round. It stops when it's done — never when a script says so.

Every tool answers with a compact JSON envelope — `{"ok": true, ...}` or `{"ok": false, "error": "..."}` — and never raises: a failed lookup comes back as data the model can shrug off or retry. The whole run is capped at 120 s, so a pathological loop can't hold the webhook hostage.

## What it can see and hear

The agent isn't blind, and it isn't deaf:

- **Photos** are downloaded, attached to that turn's LLM call, then discarded — chat memory stays text-only, no megabyte payloads rotting in the rolling buffer.
- **Voice notes** are transcribed by a WhisperX service (`WAHABOT_TRANSCRIBE_URL`) and arrive as `[voice note] <transcript>` — the bot hears what was said without being asked. Off when the URL is empty.
- **Albums** arrive as a container plus N images; the handler buffers them and runs the agent once, all images attached.
- **Bare image links** in text are sniffed out, fetched, and shown to the model too.
- **Reactions** to the bot's own messages are folded into memory as context — a 👍 lands quietly, visible on the next turn, never waking the agent.

## A guest, not a loudspeaker

The bot is selective about when it speaks:

- **Backlog filter** — anything sent before startup, or older than 5 minutes, is dropped as replay noise.
- **Group participation**, per session: `never`, `mentioned` (default — only when @-mentioned or quoted), or `judicious` (reads everything, speaks when it's worth it).
- **Statuses & newsletters** are rejected before reaching the agent — no replies to Instagram cross-posts.
- **Access control** — per-session whitelist/blacklist; blacklist always wins, both empty means answer everybody.
- **Session health gate** — if the WAHA session leaves `WORKING`, the bot mutes itself instead of burning LLM tokens on replies that can't be delivered, and pings the operator's own WhatsApp. Recovery is automatic.

## Talking to the bot yourself

`wahabot tell` gives the operator a direct line — not a chat message, a command:

```bash
uv run wahabot tell "send a message to Ana: the deploy is done"
uv run wahabot tell "search the latest news about elon musk and send a summary to the group Familia"
```

The agent runs the instruction with its full toolset on a fresh context. No whitelist applies, no chat history is touched, and names resolve to the right person or group automatically. The result lands in WhatsApp, not in your terminal.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in values
```

Key env vars:

| Variable | Purpose | Default |
|---|---|---|
| `WAHABOT_LLM_MODEL` | OpenAI-compatible chat model with function calling | **required** |
| `WAHABOT_LLM_TEMPERATURE` | Sampling temperature (model-card default; don't lower it) | `1.0` |
| `WAHABOT_LLM_TOP_P` | Nucleus sampling cutoff | `0.95` |
| `WAHABOT_LLM_TOP_K` | Top-k sampling candidates | `20` |
| `WAHABOT_LLM_MIN_P` | Min-p sampling floor | `0.0` |
| `WAHABOT_LLM_PRESENCE_PENALTY` | Presence penalty (raise toward 2 if the model ever repeats itself) | `0.0` |
| `WAHABOT_LLM_REPETITION_PENALTY` | Repetition penalty | `1.0` |
| `WAHABOT_LLM_TIMEOUT` | Per-request LLM HTTP timeout (s; client retries disabled) | `60` |
| `WAHABOT_WEBHOOK_HMAC_KEY` | Shared secret matching WAHA's `hmac.key` | **required** |
| `WAHABOT_HOST` / `WAHABOT_PORT` | Webhook server bind | `0.0.0.0:8080` |
| `WAHABOT_LOG_LEVEL` | loguru level | `INFO` |
| `WAHABOT_SESSION` | WAHA session name | `default` |
| `WAHABOT_MEMORY_TOKEN_LIMIT` | Per-chat rolling memory budget | `8000` |
| `WAHABOT_VISION` | Enable image understanding | `true` |
| `WAHABOT_MAX_IMAGE_BYTES` | Per-image download cap | `10485760` |
| `WAHABOT_MAX_URL_IMAGES` | Max image-URLs to fetch per message | `2` |
| `WAHABOT_MAX_FILE_BYTES` | Local-file cap for the `send_file` tool | `16777216` |
| `WAHABOT_TRANSCRIBE_URL` | WhisperX base URL for voice-note transcription (empty = off) | — |
| `WAHABOT_TRANSCRIBE_TIMEOUT` | Per-request transcription timeout (s) | `300` |
| `WAHABOT_MAX_AUDIO_BYTES` | Per-voice-note download cap | `26214400` |
| `WAHABOT_TRANSCRIBE_LANGUAGE` | Language passed to /transcribe (`auto` = detect) | `auto` |
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
{"url": "http://host:8080/api/webhook", "events": ["message", "message.reaction", "session.status"], "hmac": {"key": "your-secret-key"}}
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
uv run wahabot version                                  # show the version
uv run wahabot config                                   # show WAHABOT_* env (secrets redacted)
uv run wahabot sessions list                            # list session configs
uv run wahabot sessions view [--name N] [--raw] [--plain] # show a config, prompt rendered
uv run wahabot tell "<instruction>" [--session S]       # operator command to the agent
uv run wahabot serve [--host H] [--port P] [--reload]   # webhook server
```

## Startup logs

`wahabot serve` prints a short banner so one glance tells you what's running:
version + Python, session name, LLM model/endpoint, memory token ceiling, and
enabled features (vision / shell / transcription / langfuse). It then logs the
WAHA session's live identity, the loaded session config summary, tracing status,
and the tools the agent was built with:

```
Info: wahabot 0.2.9 (Python 3.14.6)
Info: Session: default
Info: LLM: gpt-4o-mini @ https://api.openai.com/v1
Info: Memory: 8000 token ceiling
Info: Features: vision, no-shell, no-transcribe
Info: Webhook: http://0.0.0.0:8080/api/webhook/default
Info: WAHA session default is live as My Name (4917...@c.us)
Info: Loaded session config from data/sessions/default.json: 0 whitelisted, 0 blacklisted, group_participation=mentioned
Info: Agent ready with 14 tools: fetch_chat_messages, forward_message, ...
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
| `tools/whatsapp.py` | The nine WhatsApp tools |
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
uv run python scripts/smoke_test.py   # end-to-end smoke suite
```

## Docs

- [Agent workflow — full pipeline explanation](docs/agent-workflow.md)
- [Session config — fields, group participation, access control](docs/session-config.md)
- [WAHA identity fields — `from` / `participant` / `to` semantics](docs/waha-identity-fields.md)
- [WAHA albums — multi-image reassembly](docs/waha-albums.md)
- [WAHA broadcast sources — status/newsletter handling](docs/waha-broadcast-sources.md)
