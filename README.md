# whabot

WhatsApp bot bridge for the [WAHA](https://waha.devlike.pro) HTTP API.
Runs a FastAPI webhook server that receives and HMAC-verifies WAHA
events, plus a typer CLI for operations.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in values
```

## Usage

```bash
uv run whabot serve [--host H] [--port P] [--reload]   # webhook server
uv run whabot config                                    # show WHABOT_* env
uv run whabot version
```

Configuration (`.env`, real env vars win):

- `WHABOT_LOG_LEVEL` — loguru level (default `INFO`)
- `WHABOT_HOST` / `WHABOT_PORT` — server bind address (default `0.0.0.0:8080`)
- `WHABOT_WEBHOOK_HMAC_KEY` — **required**; must match WAHA's `hmac.key`
- `WHABOT_MEMORY_TOKEN_LIMIT` — per-chat memory ceiling; oldest messages drop first (default `8000`)
- `WHABOT_WEB_SEARCH_MAX_RESULTS` — default `web_search` results (default `5`)
- `WHABOT_WEB_SEARCH_TIMEOUT` — `webserp` subprocess timeout in seconds (default `30`)
- `WHABOT_WEB_SEARCH_PROXY` — optional proxy URL for `webserp`

Point WAHA at the webhook:

```json
{"name": "default", "config": {"webhooks": [{"url": "http://host:8080/api/webhook", "events": ["message"], "hmac": {"key": "your-secret-key"}}]}}
```

## Behavior notes

- **Vision** — with `WHABOT_VISION=true` (default) and a vision-capable
  `WHABOT_LLM_MODEL`, incoming image messages (photo + optional caption)
  are downloaded from WAHA and shown to the model on that turn only; chat
  memory stays text-only. Bare image links in the text (up to
  `WHABOT_MAX_URL_IMAGES` per message) are fetched and shown too. Images
  over `WHABOT_MAX_IMAGE_BYTES` (10 MB) are skipped, and download failures
  degrade to a text-only turn.
- **Backlog filter** — messages sent before the server started, or older
  than 5 minutes, are dropped as WhatsApp replay backlog.
- **Agent & tools** — see [`docs/agent-workflow.md`](docs/agent-workflow.md);
  per-session access/persona in [`docs/session-config.md`](docs/session-config.md).

## Checks

```bash
uv run ruff check --fix .
uv run ruff format .
uv run basedpyright
uv run radon cc src -s
```
