# Installation

wahabot is a Python package run with [uv](https://docs.astral.sh/uv/). This
guide covers a local install; for the container runtime (Docker/Compose) see
the environment variables below too, which apply the same way.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in values
```

Key env vars:

| Variable | Purpose | Default |
|---|---|---|
| `WAHABOT_LLM_API_BASE` | OpenAI-compatible endpoint URL | **required** |
| `WAHABOT_LLM_API_KEY` | API key for that endpoint | **required** |
| `WAHABOT_LLM_MODEL` | Chat model with function calling | `gpt-4o-mini` |
| `WAHABOT_LLM_TEMPERATURE` | Sampling temperature (model-card default; don't lower it) | `1.0` |
| `WAHABOT_LLM_TOP_P` | Nucleus sampling cutoff | `0.95` |
| `WAHABOT_LLM_TOP_K` | Top-k sampling candidates | `20` |
| `WAHABOT_LLM_MIN_P` | Min-p sampling floor | `0.0` |
| `WAHABOT_LLM_PRESENCE_PENALTY` | Presence penalty (raise toward 2 if the model ever repeats itself) | `0.0` |
| `WAHABOT_LLM_REPETITION_PENALTY` | Repetition penalty | `1.0` |
| `WAHABOT_LLM_REASONING_EFFORT` | Reasoning effort for thinking models (`low`/`medium`/`high`; empty = provider default) | — |
| `WAHABOT_LLM_TIMEOUT` | Per-request LLM HTTP timeout (s; client retries disabled) | `60` |
| `WAHABOT_RUN_TIMEOUT` | Server-side cap on one agent run (s; 0 = unlimited) | `120` |
| `WAHABOT_TOOL_ROUND_LIMIT` | Max LLM→tool round trips per run (one final tool-free call past the limit) | `50` |
| `WAHABOT_TELL_TIMEOUT` | How long `wahabot tell` waits for the run to finish (s; 0 = forever) | `30` |
| `WAHABOT_WEBHOOK_HMAC_KEY` | Shared secret matching WAHA's `hmac.key` | **required** |
| `WAHABOT_WAHA_URL` | WAHA HTTP API base URL | **required** |
| `WAHABOT_WAHA_API_KEY` | WAHA API key | **required** |
| `WAHABOT_HOST` / `WAHABOT_PORT` | Webhook server bind | `0.0.0.0:8080` |
| `WAHABOT_LOG_LEVEL` | loguru level | `INFO` |
| `WAHABOT_SESSION` | WAHA session name | `default` |
| `WAHABOT_DATA_DIR` | Root for session configs, memory and the event journal | `data` |
| `WAHABOT_TIMEZONE` | TZ for the prompt's date/time placeholders | `UTC` |
| `WAHABOT_MEMORY_TOKEN_LIMIT` | Per-chat rolling memory budget | `8000` |
| `WAHABOT_MEMORY_PERSIST` | Persist per-chat memory to `data/memory/` (survives restarts/LRU evictions) | `true` |
| `WAHABOT_VISION` | Enable image understanding | `true` |
| `WAHABOT_MAX_IMAGE_BYTES` | Per-image download cap | `10485760` |
| `WAHABOT_MAX_URL_IMAGES` | Max image-URLs to fetch per message | `2` |
| `WAHABOT_VIDEO` | Enable video understanding (frames + WhisperX transcript; needs ffmpeg) | `true` |
| `WAHABOT_MAX_VIDEO_BYTES` | Per-video download cap | `67108864` |
| `WAHABOT_VIDEO_FRAMES` | Frames sampled per video for the caption call | `6` |
| `WAHABOT_MAX_FILE_BYTES` | Local-file cap for the `send_file` tool | `16777216` |
| `WAHABOT_TRANSCRIBE_URL` | WhisperX base URL for voice-note transcription (empty = off) | — |
| `WAHABOT_TRANSCRIBE_TIMEOUT` | Per-request transcription timeout (s) | `300` |
| `WAHABOT_MAX_AUDIO_BYTES` | Per-voice-note download cap | `26214400` |
| `WAHABOT_TRANSCRIBE_LANGUAGE` | Language passed to /transcribe (`auto` = detect) | `auto` |
| `WAHABOT_WEB_SEARCH_MAX_RESULTS` | Default web search results | `5` |
| `WAHABOT_WEB_SEARCH_TIMEOUT` | webserp subprocess and `visit_url` fetch timeout (s) | `30` |
| `WAHABOT_WEB_SEARCH_PROXY` | Optional proxy for webserp | — |
| `WAHABOT_SHELL_TOOL` | Enable shell tool (off by default; run unprivileged/sandboxed) | `false` |
| `WAHABOT_SHELL_TIMEOUT` | Shell command timeout (s) | `30` |
| `WAHABOT_SHELL_MAX_OUTPUT` | Max chars returned from a shell command | `2000` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Opt-in tracing (see README) | — |

## Quick start

```bash
uv sync
cp .env.example .env   # fill in WAHABOT_* + LLM values
uv run wahabot sessions init          # write a starter session config
uv run wahabot serve                  # start the webhook server
```

Point WAHA at the webhook in your session config:

```json
{"url": "http://host:8080/api/webhook/default", "events": ["message", "message.reaction", "session.status"], "hmac": {"key": "your-secret-key"}}
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

The `system_prompt` is the bot's entire personality. Write it like you're
describing a friend, not a service. The model will mirror whatever tone you set.

### Other commands

```bash
uv run wahabot version                                  # show the version
uv run wahabot config                                   # show WAHABOT_* env (secrets redacted)
uv run wahabot sessions list                            # list session configs
uv run wahabot sessions view [--name N] [--raw] [--plain] # show a config, prompt rendered
uv run wahabot tell "<instruction>" [--session S]       # operator command to the agent
uv run wahabot forget "<chat-jid>" [--session S]        # wipe one chat's persistent memory
uv run wahabot serve [--host H] [--port P] [--reload]   # webhook server
```

The same operator path is available from WhatsApp: send a message to the
bot's own self-chat beginning with the configured `bot_mention_regex`, for
example `kAI do this and send a message to Roy`. Only matching messages sent
to the bot's own JID are treated as commands — the reply lands as a
quote-reply in that same self-chat; other `fromMe` messages remain
memory-only.

See [Session Config](session-config.md) for the full list of CLI commands
and every `data/sessions/<session>.json` field.
