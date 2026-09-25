# Python Developer AI Agent

## Communication
You are talking to seniors. Be terse — no preamble, no recap, no
hand-holding, no "here's what I did" summaries. State results, not
intentions. Skip explanations unless explicitly asked. Never narrate
tool use. Answer the question; move on.

## Project
wahabot — WhatsApp bot bridge built on the WAHA HTTP API
(https://waha.devlike.pro). A Python CLI (`wahabot`) that runs a FastAPI
webhook server to receive WAHA events (messages, session status, etc.),
with HMAC-authenticated webhooks and a typer-based command surface for
operational tasks.

## Stack
- python 3.14+, uv, pyproject.toml (uv_build backend, src layout)
- typer CLI (`wahabot` entrypoint, `src/wahabot/cli.py`), FastAPI webhook
  server (`src/wahabot/webhook.py`, uvicorn)
- Pydantic models for WAHA event payloads (`src/wahabot/core/models.py`)
- HMAC webhook verification (`src/wahabot/core/hmac.py`) — always
  enforced, key in `WAHABOT_WEBHOOK_HMAC_KEY`
- loguru logging, level via `WAHABOT_LOG_LEVEL`; config from `.env`
  (pydantic-settings, real env vars win)
- Type checker: **basedpyright** (`[tool.basedpyright]` in pyproject.toml)
- Lint/format: **ruff** (line-length 90)
- Complexity: **radon**
- Dead code: **vulture**

## Commands
```bash
uv sync                                # install deps
uv run wahabot serve [--port N] [--reload]   # webhook server
uv run ruff check --fix .               # lint
uv run ruff format .                   # format
uv run basedpyright ./src/wahabot       # type check
uv run radon cc ./src/wahabot -s        # complexity (must show no C/D/E/F blocks)
uvx --python 3.14 vulture src/ --min-confidence 60   # dead code
uv run pytest                           # end-to-end smoke + unit suite
uv run wahabot --help                   # CLI surface
```
Run `ruff check`, `ruff format --check`, `basedpyright`, `radon cc`,
and `vulture` after every change.

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

## Dead code
Rules say "no dead code"; vulture enforces it. It must run on a 3.14
parser (`uvx --python 3.14`) — the code uses PEP 758 parenthesis-free
`except A, B`, which older parsers silently skip (unparsed files look
clean). Config lives in `[tool.vulture]` in pyproject.toml
(`ignore_decorators` for framework-dispatched registrations — `@step`,
`@app.*`, `@on_*`; `ignore_names` for typer commands, pydantic members
and dotted-string references like
`uvicorn http="wahabot.core.protocol:LoggingH11Protocol"`), so the
default run is **expected to be empty**. A hit is a *suspect*: check
for a registration site before deleting; if it is a new framework
registration pattern, add it to the pyproject lists in the same pass.
Real dead code (never-read fields, zero-call-site functions) goes out
in the same pass.

## PII
Never use or commit PII — not in code, tests, docs, plans, fixtures,
examples, or commit messages. PII is any real person's or chat's data:
names, group titles, JIDs, phone numbers, message bodies, voice-note
transcripts, trace/session ids tied to a person. Rules:

- Tests, docstrings, and examples use fictional identities only
  ("Bridge Club", `491555000001@c.us`-style JIDs, invented bodies).
  This applies even when the fixture is "just" a name: a real group
  title or contact name in a test is still that person's data.
- Incident write-ups describe the failure; they do not name the
  people or reproduce their messages. Real evidence stays in local,
  untracked artifacts (incident notes, `data/`, exports).
- Before committing anything touched by a real conversation, grep the
  diff for names/JIDs you saw in it: `git diff | rg -i "<name|jid>"`.
  If it came from a chat, it does not go in the repo.

## Priority
Clean, simple, maintainable code. Nothing else.

## Commits
One-line messages only. No body, no trailers.

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
- CLI commands live in `src/wahabot/cli.py` as typer commands; env vars
  (`WAHABOT_HOST`, `WAHABOT_PORT`) override flag defaults for `serve`.
- Shared helpers live under `src/wahabot/core/` — keep the package root
  to entrypoints only (`cli.py`, `webhook.py`, `settings.py`).
- Webhook handlers stay thin: parse → verify → dispatch → log. Payload
  models in `core/models.py` (`WahaEvent`, `extra="allow"` since WAHA
  adds engine-specific fields).
- HMAC verification is mandatory — no disabled/no-op path. Missing key
  is a 500, bad signature a 401.
- Configuration comes from `.env` via pydantic-settings; real env
  vars take precedence. Add new vars to both `.env` and `.env.example`.
- WAHA reference: https://waha.devlike.pro/docs/how-to/events/
