# wahabot webhook server image; meant to be run via docker compose.
# Slim Debian with the pinned Python (see .python-version) and uv preinstalled.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    # venv and cache live inside the image/container; hardlinks can't span
    # filesystems, so always copy.
    UV_LINK_MODE=copy \
    # The base image already ships the pinned Python; never download another.
    UV_PYTHON_DOWNLOADS=never \
    # webserp is spawned as a subprocess, so the venv must be on PATH.
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Unprivileged runtime user; uid 1000 matches the typical host user so the
# bind-mounted data dir stays writable for them.
RUN useradd --uid 1000 --create-home wahabot \
    && install -d -o wahabot -g wahabot /app
USER wahabot
WORKDIR /app

# Dependency layer: only invalidated when the lockfile changes.
COPY --chown=wahabot:wahabot pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/home/wahabot/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-install-project --no-dev

# Project layer: install wahabot itself.
COPY --chown=wahabot:wahabot src ./src
COPY --chown=wahabot:wahabot README.md LICENSE ./
RUN --mount=type=cache,target=/home/wahabot/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

EXPOSE 8080
CMD ["wahabot", "serve"]
