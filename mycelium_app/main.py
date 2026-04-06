from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session

from mycelium_app.db import create_db_and_tables
from mycelium_app.db import engine
from mycelium_app.hive_empathy import queue_homeostasis_failure
from mycelium_app.homeostasis import list_recent_user_ids, tick_homeostasis
from mycelium_app.routes.auth import router as auth_router
from mycelium_app.routes.game import router as game_router
from mycelium_app.routes.growth import router as growth_router
from mycelium_app.routes.hive import router as hive_router
from mycelium_app.routes.homeostasis import router as homeostasis_router
from mycelium_app.routes.nexus import router as nexus_router
from mycelium_app.routes.predict import router as predict_router
from mycelium_app.routes.projects import router as projects_router
from mycelium_app.routes.reflection import router as reflection_router
from mycelium_app.routes.telemetry import router as telemetry_router
from mycelium_app.routes.tree import router as tree_router
from mycelium_app.settings import settings
from mycelium_app.web import router as web_router


app = FastAPI(title=settings.app_name)


async def _homeostasis_daemon() -> None:
    """Background daemon that periodically refreshes per-user HomeostasisState.

    This implements a minimal "Global Workspace" broadcast: mood/identity is
    recomputed on a cadence and persisted so other subsystems can read it.
    """

    # Delay a bit so startup is fast.
    await asyncio.sleep(1.0)
    tick_s = max(5, min(int(settings.nexus_homeostasis_tick_seconds), 3600))

    while True:
        try:
            with Session(engine) as session:
                user_ids = list_recent_user_ids(session, window_hours=24)
                for uid in user_ids[:200]:
                    try:
                        tick_homeostasis(session, user_id=int(uid), project_id=None)
                    except Exception as e:
                        # Homeostasis should never take the app down.
                        # Policy: degrade gracefully + ask parent (throttled).
                        try:
                            queue_homeostasis_failure(
                                session,
                                user_id=int(uid),
                                project_id=None,
                                device_id=str(settings.nexus_device_id or "local"),
                                error_type=type(e).__name__,
                                error_message=str(e),
                                min_interval_minutes=30,
                            )
                        except Exception:
                            pass
                        continue
        except Exception:
            pass

        await asyncio.sleep(float(tick_s))


@app.on_event("startup")
async def on_startup() -> None:
    create_db_and_tables()
    if bool(getattr(settings, "nexus_homeostasis_enabled", False)):
        asyncio.create_task(_homeostasis_daemon())


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(game_router)
app.include_router(hive_router)
app.include_router(homeostasis_router)
app.include_router(nexus_router)
app.include_router(growth_router)
app.include_router(telemetry_router)
app.include_router(reflection_router)
app.include_router(predict_router)
app.include_router(projects_router)
app.include_router(tree_router)
app.include_router(web_router)

app.mount("/static", StaticFiles(directory="static"), name="static")
