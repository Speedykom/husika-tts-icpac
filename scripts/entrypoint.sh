#!/bin/sh
# Container entrypoint: pre-warm MMS models in the background, then start uvicorn.
# Pre-warming runs concurrently so /health comes up immediately while models
# stream into the hf_cache volume.
set -e

.venv/bin/python scripts/prewarm.py &

exec .venv/bin/uvicorn tts_service.api.server:app --host 0.0.0.0 --port 8000
