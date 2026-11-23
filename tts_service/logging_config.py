"""Centralized JSON logging for the TTS service.

Emits one JSON object per log line to stdout so logs are machine-parseable by a
log aggregator. The root level is taken from the LOG_LEVEL env var (default
INFO). Structured fields are passed through the standard `extra=` mechanism,
e.g. `logger.info("rating saved", extra={"event": "rating_saved", ...})`.
"""

import json
import logging
import os
from datetime import datetime, timezone

# Standard LogRecord attributes — anything outside this set is treated as a
# caller-supplied structured field and merged into the JSON output.
_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
    "asctime",
    # uvicorn injects an ANSI-colored duplicate of the message — drop it.
    "color_message",
}


def _extract_extras(record: logging.LogRecord) -> dict:
    """Return the caller-supplied structured fields passed via `extra=`."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extract_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class SQLiteLogHandler(logging.Handler):
    """Persist each log record as a row in the shared SQLite `logs` table.

    The structured `event` field (if present) is stored in its own column for
    easy filtering; any remaining `extra=` fields are stored as a JSON blob.

    TODO: `emit` writes synchronously and opens a fresh connection per record.
    Since this runs inline on the request path, under load it adds latency and
    WAL write contention. Move to an async queue handler (e.g. a background
    thread draining `logging.handlers.QueueHandler`/`QueueListener`, or batched
    inserts on a reused connection) before this sees meaningful traffic.
    """

    def __init__(self, db) -> None:
        super().__init__()
        self._db = db

    def emit(self, record: logging.LogRecord) -> None:
        try:
            extras = _extract_extras(record)
            event = extras.pop("event", None)
            timestamp = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()
            with self._db.connect() as conn:
                conn.execute(
                    "INSERT INTO logs "
                    "(timestamp, level, logger, event, message, fields) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        timestamp,
                        record.levelname,
                        record.name,
                        event,
                        record.getMessage(),
                        json.dumps(extras, default=str) if extras else None,
                    ),
                )
        except Exception:
            self.handleError(record)


def configure_logging() -> None:
    """Configure root logging to emit JSON to stdout, honoring LOG_LEVEL."""
    level_name = os.environ.get("LOG_LEVEL", "info").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Persist our application logs (business events, warnings, errors) to the
    # shared SQLite database. Attached to the `tts_service` logger rather than
    # root so uvicorn access logs and third-party noise stay out of the table.
    from .db import _db

    app_logger = logging.getLogger("tts_service")
    app_logger.addHandler(SQLiteLogHandler(_db))

    # Route uvicorn's own loggers through our JSON handler instead of their
    # default text formatter, so all output is consistent.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
