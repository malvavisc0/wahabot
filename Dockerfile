# wahabot webhook server image; meant to be run via docker compose.
FROM astral/uv:python3.14-trixie

# OCI annotations. Docker build args set at build time (build-push-action
# passes IMAGE_REVISION=github.sha, IMAGE_CREATED, IMAGE_VERSION); their
# defaults below make a plain `docker build .` succeed too.
ARG IMAGE_TITLE="wahabot"
ARG IMAGE_DESCRIPTION="WhatsApp bot bridge on the WAHA HTTP API"
ARG IMAGE_LICENSES=MIT
ARG IMAGE_SOURCE="https://github.com/malvavisc0/wahabot"
ARG IMAGE_VERSION
ARG IMAGE_REVISION
ARG IMAGE_CREATED
LABEL org.opencontainers.image.title="${IMAGE_TITLE}" \
      org.opencontainers.image.description="${IMAGE_DESCRIPTION}" \
      org.opencontainers.image.licenses="${IMAGE_LICENSES}" \
      org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.version="${IMAGE_VERSION:-0.0.0}" \
      org.opencontainers.image.revision="${IMAGE_REVISION:-unknown}" \
      org.opencontainers.image.created="${IMAGE_CREATED:-unknown}"

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
# No --mount=type=cache for apt: buildx builds amd64+arm64 concurrently and
# they would share the same cache target, so apt's lock at /var/lib/apt/lists/lock
# is grabbed by two builds at once and the second fails (exit 100). Layer caching
# plus the GHA cache (type=gha) already keeps repeat builds fast.
RUN apt-get update \
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
