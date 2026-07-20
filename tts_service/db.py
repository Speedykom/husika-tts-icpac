"""Shared SQLite database for Husika TTS.

All persistent application state (users, ratings) lives in a single SQLite
file so that backups, migrations, and inspection are trivial.

The module-level singleton `_db` is imported by the store modules. Override
the path via the `DB_PATH` environment variable before the process starts.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Default location: <project_root>/data/husika.db
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "husika.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS users (
    username         TEXT PRIMARY KEY,
    hashed_password  TEXT NOT NULL,
    is_admin         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS ratings (
    reviewer    TEXT    NOT NULL COLLATE NOCASE,
    language    TEXT    NOT NULL COLLATE NOCASE,
    phrase      TEXT    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment     TEXT,
    timestamp   TEXT    NOT NULL,
    audio_file  TEXT,
    PRIMARY KEY (reviewer, language, phrase)
);

CREATE TABLE IF NOT EXISTS logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    level      TEXT NOT NULL,
    logger     TEXT NOT NULL,
    event      TEXT,
    message    TEXT NOT NULL,
    fields     TEXT
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_event ON logs (event);
"""


def _resolve_db_path() -> Path:
    db_path = os.environ.get("DB_PATH")
    if db_path:
        return Path(db_path)
    return _DEFAULT_DB


class Database:
    """Thin wrapper around a SQLite file with a per-call connection factory."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else _resolve_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(_SCHEMA)

    def prune_logs(self, retention_days: int) -> int:
        """Delete `logs` rows older than ``retention_days`` and return the
        number removed.

        Keeps the shared database bounded so log volume doesn't grow forever.
        Timestamps are stored as ISO-8601 UTC strings, which sort
        lexicographically, so a string comparison on the indexed `timestamp`
        column is correct. A non-positive ``retention_days`` is a no-op.
        """
        if retention_days <= 0:
            return 0
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=retention_days)
        ).isoformat()
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff,))
            return cursor.rowcount

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# Module-level singleton
_db = Database(_resolve_db_path())
