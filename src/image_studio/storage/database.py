"""SQLite connection management and schema migrations."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

from image_studio.storage import migrations as migrations_package

_SCHEMA_VERSION = 2


def connect(database_file: Path) -> sqlite3.Connection:
    """Open the application database with WAL and foreign keys enabled.

    The connection is shared across the event-dispatch thread and API
    threads; callers serialize access through the repository lock.
    """
    database_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_file, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _migration_scripts() -> list[tuple[int, str]]:
    root = resources.files(migrations_package)
    scripts: list[tuple[int, str]] = []
    for entry in sorted(root.iterdir()):
        name = entry.name
        if name.endswith(".sql"):
            version = int(name.split("_", 1)[0])
            scripts.append((version, entry.read_text(encoding="utf-8")))
    return scripts


def _statements(script: str) -> list[str]:
    """Split a migration script into complete SQL statements."""
    statements: list[str] = []
    buffer = ""
    for character in script:
        buffer += character
        if character == ";" and sqlite3.complete_statement(buffer):
            statements.append(buffer)
            buffer = ""
    if buffer.strip():
        raise ValueError("migration script ends with an incomplete statement")
    return statements


def migrate(connection: sqlite3.Connection) -> int:
    """Apply pending migrations atomically; return the resulting schema version.

    Each migration runs as one explicit transaction that includes the
    ``PRAGMA user_version`` advancement, so a mid-script failure rolls back
    completely: no partial tables, version untouched, and the next startup
    can retry the same script.
    """
    current = connection.execute("PRAGMA user_version").fetchone()[0]
    for version, script in _migration_scripts():
        if version <= current:
            continue
        statements = _statements(script)
        previous_isolation = connection.isolation_level
        connection.isolation_level = None  # explicit transaction control
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in statements:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version={version}")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.isolation_level = previous_isolation
        current = version
    return current


def schema_version(connection: sqlite3.Connection) -> int:
    return connection.execute("PRAGMA user_version").fetchone()[0]
