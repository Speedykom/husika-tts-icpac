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
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Default location: <project_root>/data/husika.db
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "husika.db"

_USERS_TABLE = """\
CREATE TABLE IF NOT EXISTS users (
    username         TEXT PRIMARY KEY,
    hashed_password  TEXT NOT NULL,
    is_admin         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

# Ratings are append-only history: every submission is a new row, so a reviewer
# can rate the same phrase multiple times without overwriting earlier reviews.
# The index speeds up the "latest review for this reviewer/language/phrase"
# lookup used to pre-fill the UI.
_RATINGS_TABLE = """\
CREATE TABLE IF NOT EXISTS ratings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reviewer    TEXT    NOT NULL COLLATE NOCASE,
    language    TEXT    NOT NULL COLLATE NOCASE,
    phrase      TEXT    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment     TEXT,
    timestamp   TEXT    NOT NULL,
    audio_file  TEXT
);

CREATE INDEX IF NOT EXISTS idx_ratings_lookup
    ON ratings (reviewer, language, phrase);
"""

_SCHEMA = _USERS_TABLE + _RATINGS_TABLE


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
            self._migrate_ratings(conn)
            conn.executescript(_SCHEMA)

    def _migrate_ratings(self, conn: sqlite3.Connection) -> None:
        """Rebuild a legacy ratings table (composite PK, no `id`) into the
        append-only history schema, preserving existing rows."""
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(ratings)")]
        if not cols or "id" in cols:
            return  # fresh database or already migrated
        logger.info("Migrating ratings table to append-only history schema")
        conn.executescript(
            "ALTER TABLE ratings RENAME TO ratings_legacy;\n"
            + _RATINGS_TABLE
            + "INSERT INTO ratings "
            "(reviewer, language, phrase, rating, comment, timestamp, audio_file) "
            "SELECT reviewer, language, phrase, rating, comment, timestamp, audio_file "
            "FROM ratings_legacy;\n"
            "DROP TABLE ratings_legacy;\n"
        )

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
