from __future__ import annotations

from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

from mycelium_app.settings import settings


def _ensure_storage_dir() -> None:
    Path("storage").mkdir(parents=True, exist_ok=True)


_ensure_storage_dir()


def _normalize_database_url(url: str) -> str:
    """Normalize provider URLs to SQLAlchemy dialect URLs.

    Some providers (including Railway) expose Postgres URLs as `postgres://...`.
    SQLAlchemy expects `postgresql+psycopg://...` when using psycopg v3.
    """

    u = str(url or "").strip()
    if u.startswith("postgres://"):
        return "postgresql+psycopg://" + u[len("postgres://") :]
    if u.startswith("postgresql://") and "+" not in u.split("://", 1)[0]:
        # Prefer psycopg v3 driver.
        return "postgresql+psycopg://" + u[len("postgresql://") :]
    return u

engine = create_engine(
    _normalize_database_url(settings.database_url),
    connect_args={"check_same_thread": False} if str(settings.database_url).startswith("sqlite") else {},
)


def create_db_and_tables() -> None:
    from mycelium_app import models  # noqa: F401

    mode = str(getattr(settings, "db_migration_mode", "create_all") or "create_all").strip().lower()
    auto_create = bool(getattr(settings, "db_auto_create_tables", True))
    if not auto_create:
        if mode != "migrate":
            return

    # create_all is safe: creates missing tables without dropping existing ones
    try:
        SQLModel.metadata.create_all(engine)
    except Exception:
        import traceback
        traceback.print_exc()

    # Add missing columns to existing tables (SQLite ALTER TABLE)
    if str(settings.database_url).startswith("sqlite"):
        _migrate_sqlite_columns(engine)


def _migrate_sqlite_columns(eng) -> None:
    """Add missing columns to existing SQLite tables. Safe to run repeatedly."""
    import sqlite3
    url = str(settings.database_url).replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(url)
        cursor = conn.cursor()

        # Check existing columns and add missing ones
        migrations = [
            ("user", "gender", "TEXT DEFAULT ''"),
            # Backward-compatible autonomy schema upgrades.
            ("autonomyactionfeedback", "action_name", "TEXT DEFAULT ''"),
            ("autonomygoalstate", "last_7d_json", "TEXT DEFAULT '{}'"),
        ]
        for table, column, col_type in migrations:
            try:
                cursor.execute(f"SELECT {column} FROM {table} LIMIT 1")
            except sqlite3.OperationalError:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                except Exception:
                    pass

        conn.commit()
        conn.close()
    except Exception:
        pass


def get_session():
    with Session(engine) as session:
        yield session
