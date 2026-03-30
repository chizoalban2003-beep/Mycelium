from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mycelium_app.db import create_db_and_tables
from mycelium_app.routes.auth import router as auth_router
from mycelium_app.routes.game import router as game_router
from mycelium_app.routes.projects import router as projects_router
from mycelium_app.routes.tree import router as tree_router
from mycelium_app.settings import settings
from mycelium_app.web import router as web_router


app = FastAPI(title=settings.app_name)


@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(game_router)
app.include_router(projects_router)
app.include_router(tree_router)
app.include_router(web_router)

app.mount("/static", StaticFiles(directory="static"), name="static")
