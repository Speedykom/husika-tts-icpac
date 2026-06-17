"""SQLite-backed persistent store for audio quality ratings."""

from datetime import datetime, timezone
from typing import Optional

from ..db import _db


class RatingsStore:
    """Store that persists ratings to the shared SQLite database.

    Primary key: (reviewer, language, phrase) — submitting again updates the row.
    Reviewer and language comparisons are case-insensitive; phrase is exact-match.
    """

    def upsert(
        self,
        reviewer: str,
        language: str,
        phrase: str,
        rating: int,
        comment: Optional[str] = None,
        audio_file: Optional[str] = None,
    ) -> dict:
        """
        Insert or update a rating row.
        Returns the saved record.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        with _db.connect() as conn:
            conn.execute(
                (
                    "INSERT INTO ratings "
                    "(reviewer, language, phrase, rating, comment, timestamp, "
                    "audio_file) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(reviewer, language, phrase) DO UPDATE SET "
                    "rating = excluded.rating, "
                    "comment = excluded.comment, "
                    "timestamp = excluded.timestamp, "
                    "audio_file = COALESCE(excluded.audio_file, audio_file)"
                ),
                (reviewer, language, phrase, rating, comment, timestamp, audio_file),
            )
        return {
            "reviewer": reviewer,
            "language": language,
            "phrase": phrase,
            "rating": rating,
            "comment": comment,
            "timestamp": timestamp,
            "audio_file": audio_file,
        }

    def query(
        self,
        language: Optional[str] = None,
        reviewer: Optional[str] = None,
        phrase: Optional[str] = None,
    ) -> list[dict]:
        """Return all ratings, optionally filtered by language, reviewer, or phrase."""
        sql = (
            "SELECT reviewer, language, phrase, rating, comment, timestamp, audio_file"
            " FROM ratings WHERE 1=1"
        )
        params: list = []
        if language:
            sql += " AND language = ? COLLATE NOCASE"
            params.append(language)
        if reviewer:
            sql += " AND reviewer = ? COLLATE NOCASE"
            params.append(reviewer)
        if phrase:
            sql += " AND phrase = ?"
            params.append(phrase)
        with _db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def attach_audio(
        self, reviewer: str, language: str, phrase: str, filename: str
    ) -> bool:
        """
        Set audio_file on an existing rating row.
        Returns True if the row was found.
        """
        with _db.connect() as conn:
            cursor = conn.execute(
                """UPDATE ratings SET audio_file = ?
                   WHERE reviewer = ? COLLATE NOCASE
                     AND language = ? COLLATE NOCASE
                     AND phrase   = ?""",
                (filename, reviewer, language, phrase),
            )
            return cursor.rowcount > 0
