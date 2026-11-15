"""PostgreSQL connection helpers for up_db."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import quote_plus, urlparse

import psycopg
from psycopg.rows import dict_row

from config import DATABASE_URL


def _resolve_database_url(url: str) -> str:
    """Fill in a password from the system keyring when the URL has none.

    Set DATABASE_KEYRING_SERVICE and, if needed, DATABASE_KEYRING_USER.
    When those are empty, the URL is used as written.
    """
    parsed = urlparse(url)
    if parsed.password:
        return url

    service = os.getenv("DATABASE_KEYRING_SERVICE", "").strip()
    account = os.getenv("DATABASE_KEYRING_USER", "").strip() or (parsed.username or "")
    if not service or not account:
        return url

    try:
        import keyring
    except ImportError:
        return url

    secret = keyring.get_password(service, account)
    if not secret:
        return url

    user = parsed.username or account
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    dbname = (parsed.path or "/up_db").lstrip("/") or "up_db"
    return (
        f"postgresql://{quote_plus(user)}:{quote_plus(secret)}"
        f"@{host}:{port}/{dbname}"
    )


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(_resolve_database_url(DATABASE_URL), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
