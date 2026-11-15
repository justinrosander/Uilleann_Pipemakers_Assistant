"""Create the measurement database and write DATABASE_URL into .env.

Run from the project directory:

    py scripts/setup_postgres.py

The script asks for the PostgreSQL administrator login, creates the
application database when it is missing, and applies db/schema.sql.
An existing database is left in place. The password is not printed.
"""

from __future__ import annotations

import getpass
import sys
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "db" / "schema.sql"
ENV_PATH = ROOT / ".env"


def ask(label: str, default: str) -> str:
    typed = input(f"{label} [{default}]: ").strip()
    return typed or default


def connect(user: str, password: str, host: str, port: int, dbname: str) -> psycopg.Connection:
    return psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        autocommit=True,
    )


def quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Use only letters, numbers, and underscores: {name}")
    return f'"{name}"'


def database_exists(conn: psycopg.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
    return row is not None


def apply_schema(conn: psycopg.Connection) -> None:
    script = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.execute(script)


def write_database_url(url: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    replaced = False
    updated = []
    for line in lines:
        if line.startswith("DATABASE_URL="):
            updated.append(f"DATABASE_URL={url}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        if updated and updated[-1] != "":
            updated.append("")
        updated.append(f"DATABASE_URL={url}")
    ENV_PATH.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"Schema file not found: {SCHEMA_PATH}")
        return 1
    print("PostgreSQL first-time setup. Press Enter to accept a default.")
    host = ask("Host", "localhost")
    port = int(ask("Port", "5432"))
    admin_db = ask("Maintenance database", "postgres")
    admin_user = ask("Admin user", "postgres")
    admin_password = getpass.getpass("Admin password: ")
    app_db = ask("Application database", "up_db")
    app_user = ask("Application user", admin_user)
    app_password = getpass.getpass("Application password (empty uses the admin password): ")
    if not app_password:
        app_password = admin_password
    if not admin_password:
        print("A password is required.")
        return 1

    try:
        admin = connect(admin_user, admin_password, host, port, admin_db)
    except Exception as exc:
        print(f"Could not connect to the maintenance database: {exc}")
        return 1
    try:
        if database_exists(admin, app_db):
            print(f"Database {app_db} already exists. Applying the schema only.")
        else:
            admin.execute(f"CREATE DATABASE {quote_ident(app_db)}")
            print(f"Created database {app_db}.")
    finally:
        admin.close()

    try:
        app = connect(app_user, app_password, host, port, app_db)
    except Exception as exc:
        print(f"Could not connect to {app_db}: {exc}")
        return 1
    try:
        apply_schema(app)
    finally:
        app.close()

    url = (
        f"postgresql://{quote_plus(app_user)}:{quote_plus(app_password)}"
        f"@{host}:{port}/{app_db}"
    )
    # Confirm the URL parses before writing it.
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.path:
        print("The connection URL could not be built.")
        return 1
    write_database_url(url)
    print(f"Schema is ready. DATABASE_URL was written to {ENV_PATH.name}.")
    print("The password is stored only in that local file, which Git ignores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
