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
| `WAHABOT_LLM_AUTO_CACHE` | Ask the gateway to prompt-cache requests (Requesty `auto_cache`; cached-prefix pricing for providers that support it) | `false` |
| `WAHABOT_LLM_TIMEOUT` | Per-request LLM HTTP timeout (s; client retries disabled) | `900` |
| `WAHABOT_LLM_CONTEXT_WINDOW` | Provider context window for the startup budget warning (`0` = unknown) | `0` |
| `WAHABOT_RUN_TIMEOUT` | Server-side cap on one agent run (s; 0 = unlimited) | `1200` |
| `WAHABOT_INTAKE_RECOVERY_MINUTES` | Accepted messages older than this are answered with an apology prefix | `30` |
| `WAHABOT_INTAKE_MAX_ATTEMPTS` | Retry budget for a failed intake row or outbound intent | `5` |
| `WAHABOT_INTAKE_RETENTION_DAYS` | Prune finished intake rows and settled intents older than this | `30` |
| `WAHABOT_TOOL_ROUND_LIMIT` | Max LLM→tool round trips per run (one final tool-free call past the limit) | `50` |
| `WAHABOT_TELL_TIMEOUT` | How long `wahabot tell` waits for the run to finish (s; 0 = forever) | `30` |
| `WAHABOT_WEBHOOK_HMAC_KEY` | Shared secret matching WAHA's `hmac.key` | **required** |
| `WAHABOT_WAHA_URL` | WAHA HTTP API base URL | **required** |
| `WAHABOT_WAHA_API_KEY` | WAHA API key | **required** |
| `WAHABOT_HOST` / `WAHABOT_PORT` | Webhook server bind | `0.0.0.0:8080` |
| `WAHABOT_LOG_LEVEL` | loguru level | `INFO` |
| `WAHABOT_SESSION` | WAHA session name | `default` |
| `WAHABOT_DATA_DIR` | Root for session configs, memory, the notebook, the event journal, skill folders and the intake/outbound ledger | `data` |
| `WAHABOT_SECRET_KEY` | Base64 32-byte Fernet key for skill secrets (empty = secret-bearing skills refuse writes) | — |
| `WAHABOT_TIMEZONE` | TZ for the prompt's date/time placeholders | `UTC` |
| `WAHABOT_MEMORY_TOKEN_LIMIT` | Per-chat rolling memory budget | `8000` |
| `WAHABOT_MEMORY_PERSIST` | Persist per-chat memory to `data/memory/` (survives restarts/LRU evictions) | `true` |
| `WAHABOT_LEARNED_MEMORY` | Merge trimmed turns into a per-chat recap in the pinned state block | `true` |
| `WAHABOT_THREAD_RECALL` | Index trimmed turns per chat and recall older slices (requires memory persistence) | `true` |
| `WAHABOT_EMBED_API_BASE` | OpenAI-compatible embeddings base URL, including `/v1` (empty = FTS5-only recall) | — |
| `WAHABOT_EMBED_MODEL` | Embedding model name for the dense half of thread recall | `embeddinggemma-2` |
| `WAHABOT_EMBED_TIMEOUT` | Per-request embeddings timeout (s) | `10` |
| `WAHABOT_TRIAGE_GATE` | Enable the d1-3B triage gate | `true` |
| `WAHABOT_GATE_API_BASE` | OpenAI-compatible gate chat base URL, including `/v1` (empty = gate off) | `http://omnitron.tago.lan:9194/v1` |
| `WAHABOT_GATE_MODEL` | Gate model name | `d1-3B` |
| `WAHABOT_GATE_API_KEY` | Optional bearer token for the gate service | — |
| `WAHABOT_GATE_TIMEOUT` | Max wait for one gate verdict (s) | `0.8` |
| `WAHABOT_VISION` | Enable image understanding | `true` |
| `WAHABOT_MAX_IMAGE_BYTES` | Per-image download cap | `10485760` |
| `WAHABOT_MAX_URL_IMAGES` | Max image-URLs to fetch per message | `2` |
| `WAHABOT_VIDEO` | Enable video understanding (frames + WhisperX transcript; needs ffmpeg) | `true` |
| `WAHABOT_MAX_VIDEO_BYTES` | Per-video download cap | `67108864` |
| `WAHABOT_VIDEO_FRAMES` | Frames sampled per video for the caption call | `6` |
| `WAHABOT_VISION_MAX_IMAGES` | Cap images per vision prompt (`0` = no cap; set `1` for single-image models) | `0` |
| `WAHABOT_MAX_URL_VIDEOS` | Max video-URLs to resolve via yt-dlp per message (`0` = off) | `1` |
| `WAHABOT_MAX_FILE_BYTES` | Local-file cap for the `send_file` tool | `16777216` |
| `WAHABOT_TRANSCRIBE_URL` | WhisperX base URL for voice-note transcription (empty = off) | — |
| `WAHABOT_TRANSCRIBE_TIMEOUT` | Per-request transcription timeout (s) | `300` |
| `WAHABOT_MAX_AUDIO_BYTES` | Per-voice-note download cap | `26214400` |
| `WAHABOT_TRANSCRIBE_LANGUAGE` | Language passed to /transcribe (`auto` = detect) | `auto` |
| `WAHABOT_TTS_URL` | OpenAI-compatible TTS base URL behind `send_voice(text=…)` (empty = off) | — |
| `WAHABOT_TTS_TIMEOUT` | Per-request synthesis timeout (s) | `120` |
| `WAHABOT_TTS_VOICES` | JSON map of language code → voice id | `{"en":"vd_british_male_casual","de":"vd_german_male_casual","es":"vd_spanish_male"}` |
| `WAHABOT_TTS_INSTRUCT` | JSON map of language code → frozen delivery instruct (empty = omit) | `{"en":"","de":"","es":"spoken casually, like teasing a friend in a group chat"}` |
| `WAHABOT_TTS_DEFAULT_LANGUAGE` | Language fallback when the reply's has no voice-map entry | `en` |
| `WAHABOT_IMAGE_API_BASE` | OpenAI-compatible image service base URL behind `generate_image`/`edit_image` (empty = tools off) | — |
| `WAHABOT_IMAGE_TIMEOUT` | Per-request generation/edit timeout (s) | `240` |
| `WAHABOT_IMAGE_SIZE` | Default render size (`WxH`) | `1024x1024` |
| `WAHABOT_WEB_SEARCH_MAX_RESULTS` | Default total valid search results across engines | `5` |
| `WAHABOT_WEB_SEARCH_TIMEOUT` | webserp subprocess and `visit_url` fetch timeout (s) | `30` |
| `WAHABOT_WEB_SEARCH_PROXY` | Optional proxy for search, pages, metadata, and caption requests | — |
| `WAHABOT_SHELL_TOOL` | Enable shell tool (off by default; run unprivileged/sandboxed) | `false` |
| `WAHABOT_SHELL_TIMEOUT` | Shell execution deadline (s, minimum 1), plus bounded cleanup | `30` |
| `WAHABOT_SHELL_MAX_OUTPUT` | Inline bytes per shell output stream (minimum 200; overflow attempts a file spill) | `2000` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Opt-in tracing (see README) | — |

The durable intake and outbound-intent ledger lives in `data/wahabot.db` (SQLite, WAL, numbered migrations applied automatically at startup). Back it up with `notebook.sqlite3`; a fresh database still runs, but accepted messages and pending deliveries from before a restart are lost.

## Skills

Skills are folders under `data/skills/<name>/`, each with `skill.yaml` and
`SKILL.md`. Import validates the contract, copies the folder and writes a
disabled configured row:

```bash
uv run wahabot skills import ./customer-records          # configure only
uv run wahabot skills test customer-records              # safe read check only
uv run wahabot skills import ./customer-records --enable # import + safe test + enable
uv run wahabot skills enable customer-records
uv run wahabot skills disable customer-records           # required before re-import
uv run wahabot skills list
```

`--enable` and `skills test` run only the prepared read (`GET`/`HEAD`/`OPTIONS`)
from `skill.yaml`; a write is never used as a test, and a non-SQL skill with no
safe check stays inactive with an explanation. Configure the secret before import:

```bash
export WAHABOT_SECRET_KEY="$(uv run python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
```

Store each credential through the skill backend (`secrets` table); do not put
it in `skill.yaml`. Missing keys refuse writes, and a wrong key fails clearly.
The key is required only for skills that reference a secret.

HTTP skills get a base-URL lock, address checks (loopback, link-local,
metadata and private ranges are refused unless the skill's own base is
private), same-host redirect limits with IP pinning, sender-only substitution,
per-turn/repeat limits and body/response/time caps. Customer writes are staged
in `skill_requests` unless `writes_need_operator` is false; team-group writes
always ask. Approvals confirm in the self-chat with the exact prepared request;
reply `yes` (quote the message when more than one is pending). Operator writes
execute directly.

SQL skills are operator-only: one `SELECT` or `WITH ... SELECT`, SQLite opened
`mode=ro` or Postgres through a read-only user/transaction, with comment,
multi-statement, `ATTACH`/`PRAGMA` and mutation refusal plus row/column/byte/
time caps. Install the optional driver with `uv sync --extra postgres`; without
it the tool returns a clear error instead of crashing. Draft SQL skills are
tested with a harmless read-only `SELECT 1` probe before they can be enabled.

Enabled event skills run in the background after the ticket transaction commits
for `opened`, `needs_you`, `done` and `closed`. A failure is logged and never
rolls back the ticket. `send-to-url` sends a stable, bounded JSON payload signed with
HMAC-SHA256; it carries no message bodies by default. The web authoring UI,
marketplace, workflow builder and arbitrary code execution are Phase 5 work.

Run one wahabot process per session. The chat lock and intake claim owners are in-process, so multiple workers against the same session are unsupported. The outbound retry policy treats a WAHA 5xx as a definite pre-acceptance failure; verify that assumption against your WAHA version before relying on automatic retries.

## Quick start

```bash
uv sync
cp .env.example .env   # fill in WAHABOT_* + LLM values
uv run wahabot sessions init          # write a starter session config
uv run wahabot serve                  # start the webhook server
```

Point WAHA at the webhook in your session config:

```json
{"url": "http://host:8080/api/webhook/default", "events": ["message.any", "message.reaction", "session.status"], "hmac": {"key": "your-secret-key"}}
```

`message.any` (not `message`) is required: `message` carries incoming
messages only, while the bot also needs the account's own messages —
operator commands in the self-chat, and the human operator's messages
in groups and DMs (see [How conversations work](conversations.md),
§1a).

`wahabot sessions init` writes a starter `data/sessions/default.json` — edit it
to at least set a `system_prompt`:

```json
{
  "goal": "Contribute when useful and make each reply clear and easy to understand.",
  "memory": "",
  "system_prompt": "You are Kai, an AI member of this WhatsApp group. Today is {{date}}. Match the group's tone, but use natural, complete sentences and enough context to be understood. Never sacrifice meaning for brevity or humor. Use WhatsApp formatting rather than Markdown headings. {{operator_tools}}",
  "bot_name": "Kai",
  "bot_mention_regex": "(?i)(?<![a-z@])@?kai(?![a-z])",
  "group_participation": "judicious",
  "whitelist": [],
  "blacklist": []
}
```

The `system_prompt` guides personality and participation; tool descriptions and
schemas explain the tool contracts. Write for a person to understand, not a fixed
line count. Include `{{operator_tools}}` to explain cross-chat permissions; the
runtime fence still enforces them. `memory` holds durable facts you maintain
(standing decisions, preferences — rendered as a `# Memory` prompt section,
hot-reloaded like the rest of the file); the model's own learned notes live in
its per-chat notebook instead. Inspect substitutions with
`uv run wahabot sessions view --name default`.

## Startup banner

`wahabot serve` prints a short banner so one glance tells you what's
running: version + Python, session name, LLM model/endpoint, memory
token ceiling, and enabled features (vision / video / shell /
transcription / langfuse). It then logs the WAHA session's live
identity, the loaded session config summary, and the toolset the agent
was built with:

```
Info: wahabot 0.7.3 (Python 3.14.6)
Info: Session: default
Info: LLM: gpt-4o-mini @ https://api.openai.com/v1
Info: Memory: 8000 token ceiling
Info: Features: vision, video, no-shell, no-transcribe, no-tts
Info: Webhook: http://0.0.0.0:8080/api/webhook/default
Info: WAHA session default is live as My Name (4917...@c.us)
Info: Loaded session config from data/sessions/default.json: 0 whitelisted, 0 blacklisted, group_participation=mentioned
Info: Agent ready: fetch_chat_messages, forward_message, ...
```

For a machine-readable dump of every `WAHABOT_*` value (secrets
redacted) use `uv run wahabot config`. The shell tool shows up only
with `WAHABOT_SHELL_TOOL=true`, Langfuse tracing only when the
`LANGFUSE_*` keys are set, and the `Agent ready` line lists exactly
what the model can call this session.

## Other commands

```bash
uv run wahabot version                                  # show the version
uv run wahabot config                                   # show WAHABOT_* env (secrets redacted)
uv run wahabot sessions list                            # list session configs
uv run wahabot sessions view [--name N] [--raw] [--plain] # show a config, prompt rendered
uv run wahabot tell "<instruction>" [--session S]       # operator command to the agent
uv run wahabot forget "<chat-jid>" [--session S]        # wipe one chat's persistent memory
uv run wahabot escalations [--session S] [--days N]     # list escalations from the audit journal
uv run wahabot serve [--host H] [--port P] [--reload]   # webhook server
```

The same operator path is available from WhatsApp: send a message to the
bot's own self-chat beginning with the configured `bot_mention_regex`, for
example `kAI do this and send a message to Roy`. Only matching messages sent
to the bot's own JID are treated as commands — the reply lands as a
quote-reply in that same self-chat. An operator command has no default
chat: name the target and the bot resolves it to a JID.

Elsewhere, what the operator types on the account is remembered as the
operator's (never as the bot's), and naming the bot in a group or DM
wakes it there like a member would.

See [Session Config](session-config.md) for the full list of CLI commands
and every `data/sessions/<session>.json` field.
