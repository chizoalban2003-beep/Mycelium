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
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
