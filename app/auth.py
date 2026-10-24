"""Email + password accounts for Drishti. No OTP, no email sending.

Signup creates the account at once and signs the user in; login checks the
password and signs the user in. Passwords are stored PBKDF2-hashed and
sessions are bearer tokens stored hashed, all in one SQLite file
(var/drishti.db). Accounts created by the old email+OTP build keep working:
their stored password hashes are checked the same way.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time

from .config import VAR_DIR

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _db_path() -> str:
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    return str(VAR_DIR / "drishti.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tokens(
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    )
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''")
    return conn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pw_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), 120_000).hex()
    return f"{salt}${dk}"


def _pw_ok(password: str, stored: str) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, _, dk = stored.partition("$")
    try:
        calc = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                   bytes.fromhex(salt), 120_000).hex()
    except ValueError:
        return False
    return hmac.compare_digest(calc, dk)


def _issue_token(conn, user_id: int, now: int) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO tokens(token_hash, user_id, created_at) VALUES(?, ?, ?)",
        (_hash(token), user_id, now))
    return token


def _user_dict(row) -> dict:
    return {"who": row["name"], "email": row["email"]}


def signup(email: str, password: str, name: str | None) -> tuple[bool, str, dict | None, str | None]:
    """Create the account and sign in at once. Returns (ok, msg, user, token)."""
    email = (email or "").strip().lower()
    name = (name or "").strip()
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address.", None, None
    if len(password or "") < 6:
        return False, "Password needs at least 6 characters.", None, None
    if len(name) < 2:
        return False, "Please enter your name.", None, None
    now = int(time.time())
    with _db() as conn:
        if conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
            return False, "This email already has an account. Log in instead.", None, None
        cur = conn.execute(
            "INSERT INTO users(name, email, password_hash, created_at)"
            " VALUES(?, ?, ?, ?)",
            (name, email, _pw_hash(password), now))
        user = {"who": name, "email": email}
        token = _issue_token(conn, cur.lastrowid, now)
    return True, "Account created.", user, token


def login(email: str, password: str) -> tuple[bool, str, dict | None, str | None]:
    """Check email + password and sign in. Returns (ok, msg, user, token)."""
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address.", None, None
    if len(password or "") < 6:
        return False, "Password needs at least 6 characters.", None, None
    now = int(time.time())
    with _db() as conn:
        user = conn.execute("SELECT id, name, email, password_hash FROM users"
                            " WHERE email = ?", (email,)).fetchone()
        if user is None:
            return False, "No account with this email. Create one first.", None, None
        if not _pw_ok(password, user["password_hash"]):
            return False, "Email or password is not right.", None, None
        token = _issue_token(conn, user["id"], now)
    return True, "Signed in.", _user_dict(user), token


def user_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT u.name, u.email FROM tokens t JOIN users u ON u.id = t.user_id"
            " WHERE t.token_hash = ?", (_hash(token),)).fetchone()
    if row is None:
        return None
    return {"who": row["name"], "email": row["email"]}


def logout(token: str | None) -> None:
    if not token:
        return
    with _db() as conn:
        conn.execute("DELETE FROM tokens WHERE token_hash = ?", (_hash(token),))
