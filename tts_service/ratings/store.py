"""SQLite-backed persistent store for audio quality ratings."""

from datetime import datetime, timezone
from typing import Optional

from ..db import _db


class RatingsStore:
    """Store that persists ratings to the shared SQLite database.

    Ratings are append-only: every submission is a new row, so re-rating the
    same phrase keeps the full history instead of overwriting the earlier
    review. Reviewer and language comparisons are case-insensitive; phrase is
    exact-match.
    """

    def add(
        self,
        reviewer: str,
        language: str,
        phrase: str,
        rating: int,
        comment: Optional[str] = None,
        audio_file: Optional[str] = None,
    ) -> dict:
        """
        Insert a new rating row, preserving any previous reviews.
        Returns the saved record (including its generated ``id``).
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        with _db.connect() as conn:
            cursor = conn.execute(
                (
                    "INSERT INTO ratings "
                    "(reviewer, language, phrase, rating, comment, timestamp, "
                    "audio_file) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)"
                ),
                (reviewer, language, phrase, rating, comment, timestamp, audio_file),
            )
            row_id = cursor.lastrowid
        return {
            "id": row_id,
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
        """Return matching ratings newest-first (most recent review first),
        optionally filtered by language, reviewer, or phrase."""
        sql = (
            "SELECT id, reviewer, language, phrase, rating, comment, timestamp,"
            " audio_file FROM ratings WHERE 1=1"
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
        sql += " ORDER BY id DESC"
        with _db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def attach_audio(self, rating_id: int, filename: str) -> bool:
        """
        Set audio_file on the rating row with the given ``id`` (as returned by
        :meth:`add`). Returns True if the row was found.

        Targeting the exact id avoids the ambiguity of matching on
        reviewer/language/phrase, where two ratings submitted close together
        could otherwise attach the audio to the wrong row.
        """
        with _db.connect() as conn:
            cursor = conn.execute(
                "UPDATE ratings SET audio_file = ? WHERE id = ?",
                (filename, rating_id),
            )
            return cursor.rowcount > 0
