.PHONY: help install dev-install run test lint format clean

help:
	@echo "HUSIKA TTS — Sprint 1"
	@echo "====================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo "  make dev-install   Install with dev tools"
	@echo ""
	@echo "Run:"
	@echo "  make run           Start the API server (http://localhost:8181)"
	@echo "                     Test UI at http://localhost:8181/ui"
	@echo ""
	@echo "Test:"
	@echo "  make test          Run tests"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint          Run linters"
	@echo "  make format        Auto-format code"
	@echo "  make clean         Remove caches and build artifacts"

install:
	uv sync

dev-install:
	uv sync --extra dev

run:
	uv run uvicorn tts_service.api.server:app --host 0.0.0.0 --port 8181 --reload --env-file .env

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check tts_service/ tests/

format:
	uv run ruff format tts_service/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info
	@echo "Cleaned."
