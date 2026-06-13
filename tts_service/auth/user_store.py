"""SQLite-backed user store with bcrypt password hashing."""

import logging
import os
from typing import Optional

import bcrypt

from ..db import _db

logger = logging.getLogger(__name__)

_DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")


class UserStore:
    """User store backed by the shared SQLite database.

    On first instantiation, a default admin account is seeded if no admin
    exists. Override the password via the ADMIN_PASSWORD environment variable.
    """

    def __init__(self) -> None:
        self._ensure_default_admin()

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Return the user record if credentials are valid, else None."""
        user = self.get_user(username)
        if user and bcrypt.checkpw(password.encode(), user["hashed_password"].encode()):
            return user
        return None

    def get_user(self, username: str) -> Optional[dict]:
        """Look up a user by username (case-insensitive). Returns None if missing."""
        with _db.connect() as conn:
            row = conn.execute(
                "SELECT username, hashed_password, is_admin, created_at"
                " FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
        if not row:
            return None
        return self._to_dict(row)

    # ------------------------------------------------------------------ #
    # Admin operations
    # ------------------------------------------------------------------ #

    def list_users(self) -> list[dict]:
        """Return all users (username, is_admin, created_at — no password hash)."""
        with _db.connect() as conn:
            rows = conn.execute(
                "SELECT username, is_admin, created_at FROM users ORDER BY created_at"
            ).fetchall()
        return [
            {
                "username": r["username"],
                "is_admin": bool(r["is_admin"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def create_user(self, username: str, password: str, is_admin: bool = False) -> dict:
        """Insert a new user. Raises ValueError if the username already exists."""
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        with _db.connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
            if existing:
                raise ValueError(f"Username '{username}' already exists")
            conn.execute(
                "INSERT INTO users (username, hashed_password, is_admin) "
                "VALUES (?, ?, ?)",
                (username, hashed, int(is_admin)),
            )
        return {"username": username, "is_admin": is_admin}

    def update_password(self, username: str, new_password: str) -> bool:
        """Set a new password for the given user. Returns False if user not found."""
        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        with _db.connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET hashed_password = ? "
                "WHERE username = ? COLLATE NOCASE",
                (hashed, username),
            )
            return cursor.rowcount > 0

    def delete_user(self, username: str) -> bool:
        """Delete a user. Returns False if the user did not exist."""
        with _db.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM users WHERE username = ? COLLATE NOCASE", (username,)
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_default_admin(self) -> None:
        with _db.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM users WHERE is_admin = 1"
            ).fetchone()[0]
            if count > 0:
                return
            hashed = bcrypt.hashpw(
                _DEFAULT_ADMIN_PASSWORD.encode(), bcrypt.gensalt()
            ).decode()
            conn.execute(
                "INSERT OR IGNORE INTO users (username, hashed_password, is_admin) "
                "VALUES ('admin', ?, 1)",
                (hashed,),
            )
        logger.warning(
            "Seeded default admin user: username=admin password=%s "
            "— change this immediately!",
            _DEFAULT_ADMIN_PASSWORD,
        )

    @staticmethod
    def _to_dict(row) -> dict:
        d = dict(row)
        d["is_admin"] = bool(d.get("is_admin", 0))
        return d
