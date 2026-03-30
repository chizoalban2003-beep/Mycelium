from __future__ import annotations

from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

from mycelium_app.settings import settings


def _ensure_storage_dir() -> None:
    Path("storage").mkdir(parents=True, exist_ok=True)


_ensure_storage_dir()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
