# wahabot webhook server image; meant to be run via docker compose.
FROM astral/uv:python3.14-trixie

# Default shell for RUN and exec; the Debian base ships /bin/bash.
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# System tooling in a single apt layer (one update, one install pass):
#  media/PDF/OCR : ffmpeg, poppler, ImageMagick + Ghostscript, exiftool, webp, fonts
#  docs/convert  : pandoc, qpdf
#  archive/util  : zip, unzip, 7-zip, xz, file, ripgrep, jq, sqlite3, git, curl
#  toolchain     : build-essential, cmake, ninja, pkg-config, gdb
#  node          : nodejs + npm (trixie LTS)
RUN --mount=type=cache,target=/var/cache/apt,uid=0,gid=0 \
    --mount=type=cache,target=/var/lib/apt,uid=0,gid=0 \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        poppler-utils \
        imagemagick ghostscript \
        pandoc qpdf \
        tesseract-ocr tesseract-ocr-eng \
        webp libimage-exiftool-perl \
        fonts-liberation fonts-noto-core \
        zip unzip p7zip-full xz-utils \
        file ripgrep jq git sqlite3 curl ca-certificates \
        build-essential cmake ninja-build pkg-config gdb \
        nodejs npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer: only invalidated when the lockfile changes.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0 \
    uv sync --frozen --no-install-project --no-dev

# Project layer: install wahabot itself.
COPY src ./src
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0 \
    uv sync --frozen --no-dev

EXPOSE 8080
CMD ["wahabot", "serve"]
