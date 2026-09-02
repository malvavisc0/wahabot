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

Point WAHA at the webhook:

```json
{"name": "default", "config": {"webhooks": [{"url": "http://host:8080/api/webhook", "events": ["message"], "hmac": {"key": "your-secret-key"}}]}}
```

## Checks

```bash
uv run ruff check --fix .
uv run ruff format .
uv run basedpyright
uv run radon cc src -s
```
