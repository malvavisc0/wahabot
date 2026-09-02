# Python Developer AI Agent

## Communication
You are talking to seniors. Be terse — no preamble, no recap, no
hand-holding, no "here's what I did" summaries. State results, not
intentions. Skip explanations unless explicitly asked. Never narrate
tool use. Answer the question; move on.

## Project
whabot — WhatsApp bot bridge built on the WAHA HTTP API
(https://waha.devlike.pro). A Python CLI (`whabot`) that runs a FastAPI
webhook server to receive WAHA events (messages, session status, etc.),
with HMAC-authenticated webhooks and a typer-based command surface for
operational tasks.

## Stack
- python 3.14+, uv, pyproject.toml (uv_build backend, src layout)
- typer CLI (`whabot` entrypoint, `src/whabot/cli.py`), FastAPI webhook
  server (`src/whabot/webhook.py`, uvicorn)
- Pydantic models for WAHA event payloads (`src/whabot/core/models.py`)
- HMAC webhook verification (`src/whabot/core/hmac.py`) — always
  enforced, key in `WHABOT_WEBHOOK_HMAC_KEY`
- loguru logging, level via `WHABOT_LOG_LEVEL`; config from `.env`
  (pydantic-settings, real env vars win)
- Type checker: **basedpyright** (`[tool.basedpyright]` in pyproject.toml)
- Lint/format: **ruff** (line-length 90)
- Complexity: **radon**

## Commands
```bash
uv sync                                # install deps
uv run whabot serve [--port N] [--reload]   # webhook server
uv run ruff check --fix .               # lint
uv run ruff format .                   # format
uv run basedpyright                    # type check
uv run radon cc src -s                 # complexity (must show no C/D/E/F blocks)
uv run whabot --help                   # CLI surface
```
Run `ruff check`, `ruff format --check`, `basedpyright`, and `radon cc`
after every change.

## Complexity budget
`radon cc` (cyclomatic complexity) must report **no block ranked C or
worse** — every function/method stays at rank A or B (complexity ≤ 10).
This applies to the whole codebase, not just new code: **a pre-existing
C/D/E/F block is still a defect and must always be fixed when
encountered** — never worked around, never left "for later". When a
change pushes a block to C+ (or you touch a file containing one), split
it into smaller functions (early returns, extract helpers, dispatch
tables) instead of adding `if` branches, and fix any pre-existing C+
blocks in the same pass. Verify with:
```bash
uv run radon cc src -s | grep -E '\-\s(C|D|E|F)\s'   # must be empty
```

## Priority
Clean, simple, maintainable code. Nothing else.

## Rules
- DRY, KISS
- No unnecessary complexity
- No defensive fallbacks / over-engineering
- Explicit > implicit
- Type hints always
- PEP 8 (ruff enforces, line-length 90)
- Meaningful names, small functions
- Fail fast and clearly
- Prefer standard library
- No clever tricks
- Docstrings for public APIs
- No unnecessary inline comments
- Use uv (never pip / system python)
- No mutable module-level globals
- No fake-private `_helpers` — put functions in a module (e.g.
  `core/hmac.py`) with public names
- No dead code
- No `_`-prefix for methods

## Style
Write the simplest correct solution. Delete anything that isn't needed.

## Conventions
- CLI commands live in `src/whabot/cli.py` as typer commands; env vars
  (`WHABOT_HOST`, `WHABOT_PORT`) override flag defaults for `serve`.
- Shared helpers live under `src/whabot/core/` — keep the package root
  to entrypoints only (`cli.py`, `webhook.py`, `settings.py`).
- Webhook handlers stay thin: parse → verify → dispatch → log. Payload
  models in `core/models.py` (`WahaEvent`, `extra="allow"` since WAHA
  adds engine-specific fields).
- HMAC verification is mandatory — no disabled/no-op path. Missing key
  is a 500, bad signature a 401.
- Configuration comes from `.env` via pydantic-settings; real env
  vars take precedence. Add new vars to both `.env` and `.env.example`.
- WAHA reference: https://waha.devlike.pro/docs/how-to/events/
