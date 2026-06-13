FROM python:3.11-slim

# espeak-ng for the eSpeak engine; build-essential for Coqui's C-extension
# deps that ship sdists without aarch64 wheels (e.g. monotonic-alignment-search).
RUN apt-get update && \
    apt-get install -y --no-install-recommends espeak-ng build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies only (project itself installed after source copy)
RUN uv sync --no-dev --frozen --no-install-project

# Copy application code
COPY tts_service/ tts_service/
COPY languages/ languages/
COPY web-ui/ web-ui/
COPY scripts/ scripts/

# Install the project into the venv now that source is present
RUN uv sync --no-dev --frozen

EXPOSE 8000

CMD ["scripts/entrypoint.sh"]
