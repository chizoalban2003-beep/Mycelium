from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from mycelium_app.db import create_db_and_tables
from mycelium_app.db import engine
from mycelium_app.hive_empathy import queue_homeostasis_failure
from mycelium_app.hive_empathy import compute_wisdom_latest, stable_digest, summarize_kwargs_diff
from mycelium_app.homeostasis import list_recent_user_ids, tick_homeostasis
from mycelium_app.metric_snapshot import run_validation_shadow
from mycelium_app.models import NexusNudge, WisdomIntegrationState
from mycelium_app.routes.auth import router as auth_router
from mycelium_app.routes.game import router as game_router
from mycelium_app.routes.growth import router as growth_router
from mycelium_app.routes.hive import router as hive_router
from mycelium_app.routes.homeostasis import router as homeostasis_router
from mycelium_app.routes.identity import router as identity_router
from mycelium_app.routes.nexus import router as nexus_router
from mycelium_app.routes.nudges import router as nudges_router
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


async def _wisdom_nudge_daemon() -> None:
    """Background daemon that nudges users when new global wisdom arrives.

    This is the 'voice': when the child integrates new Hive wisdom, it
    proactively surfaces a short notification.
    """

    await asyncio.sleep(2.0)
    tick_s = 90.0

    while True:
        try:
            if not bool(settings.hive_enabled):
                await asyncio.sleep(tick_s)
                continue

            with Session(engine) as session:
                user_ids = list_recent_user_ids(session, window_hours=48)
                for uid in user_ids[:200]:
                    try:
                        latest = compute_wisdom_latest(
                            session,
                            project_id=None,
                            include_project_scoped=False,
                            limit=50,
                        )

                        new_kwargs = dict(latest.recommended_kwargs or {})
                        new_digest = stable_digest(new_kwargs)

                        q = select(WisdomIntegrationState).where(WisdomIntegrationState.user_id == int(uid))
                        q = q.where(WisdomIntegrationState.project_id.is_(None))
                        state = session.exec(q).first()

                        now = datetime.utcnow()
                        if state is None:
                            state = WisdomIntegrationState(
                                user_id=int(uid),
                                project_id=None,
                                last_wisdom_digest=str(new_digest),
                                last_wisdom_kwargs_json=json.dumps(new_kwargs, sort_keys=True, separators=(",", ":")),
                                updated_at=now,
                            )
                            session.add(state)
                            session.commit()
                            continue

                        old_digest = str(state.last_wisdom_digest or "")
                        if old_digest == str(new_digest):
                            continue

                        try:
                            old_kwargs = json.loads(state.last_wisdom_kwargs_json or "{}")
                            if not isinstance(old_kwargs, dict):
                                old_kwargs = {}
                        except Exception:
                            old_kwargs = {}

                        diff = summarize_kwargs_diff(old_kwargs, new_kwargs)
                        changed_keys = int(diff.get("changed_keys", 0) or 0)
                        max_rel = float(diff.get("max_rel_change", 0.0) or 0.0)

                        # Optional: Validation Shadow (empirical honesty).
                        shadow = run_validation_shadow(
                            session,
                            user_id=int(uid),
                            project_id=None,
                            target_col=str(getattr(settings, "nexus_validation_shadow_target_col", "") or ""),
                            baseline_kwargs=old_kwargs,
                            trial_kwargs=new_kwargs,
                            wisdom_digest=str(new_digest),
                        )

                        # Update integration state even if we don't nudge.
                        state.last_wisdom_digest = str(new_digest)
                        state.last_wisdom_kwargs_json = json.dumps(new_kwargs, sort_keys=True, separators=(",", ":"))
                        state.updated_at = now
                        session.add(state)

                        # 'Sovereign filter' for nudges: only speak when meaningful.
                        meaningful = (changed_keys >= 2) or (max_rel >= 0.10)
                        throttled = (
                            state.last_nudge_at is not None
                            and (now - state.last_nudge_at) < timedelta(hours=6)
                        )

                        # If we have an empirical benchmark, only speak when it improved.
                        empirical_ok = bool(shadow.ok and shadow.improvement_frac is not None)
                        min_imp = float(getattr(settings, "nexus_validation_shadow_min_improvement_frac", 0.02) or 0.0)
                        improved = empirical_ok and float(shadow.improvement_frac or 0.0) > float(min_imp)

                        should_nudge = (improved or (meaningful and not empirical_ok)) and not throttled

                        if should_nudge:
                            title = "New Hive wisdom integrated"
                            if improved:
                                pct = round(float(shadow.improvement_frac or 0.0) * 100.0, 1)
                                metric = str(shadow.metric_name or "metric")
                                msg = (
                                    f"Integrated Hive wisdom. Local validation improved by {pct}% "
                                    f"on {metric}."
                                )
                            else:
                                msg = (
                                    f"I learned an update from the Hive: {changed_keys} knob(s) changed "
                                    f"(max Δ≈{round(max_rel * 100.0)}%). You may see improved stability/accuracy."
                                )
                            nudge = NexusNudge(
                                created_by_user_id=int(uid),
                                project_id=None,
                                kind="wisdom_update",
                                title=title,
                                message=msg,
                                payload_json=json.dumps(
                                    {
                                        "diff": diff,
                                        "shadow": {
                                            "ok": bool(shadow.ok),
                                            "metric": shadow.metric_name,
                                            "baseline": shadow.baseline_value,
                                            "trial": shadow.trial_value,
                                            "improvement_frac": shadow.improvement_frac,
                                            "notes": shadow.notes,
                                        },
                                        "as_of": (latest.as_of.isoformat() + "Z") if latest.as_of else None,
                                        "n_whispers_used": int(latest.n_whispers_used),
                                        "digest": str(new_digest),
                                    },
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                            session.add(nudge)
                            state.last_nudge_at = now
                            session.add(state)

                        session.commit()
                    except Exception:
                        continue

        except Exception:
            pass

        await asyncio.sleep(tick_s)


@app.on_event("startup")
async def on_startup() -> None:
    create_db_and_tables()
    if bool(getattr(settings, "nexus_homeostasis_enabled", False)):
        asyncio.create_task(_homeostasis_daemon())
    # Wisdom nudges only make sense when Hive is enabled.
    if bool(getattr(settings, "hive_enabled", False)):
        asyncio.create_task(_wisdom_nudge_daemon())


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(game_router)
app.include_router(hive_router)
app.include_router(homeostasis_router)
app.include_router(nexus_router)
app.include_router(identity_router)
app.include_router(nudges_router)
app.include_router(growth_router)
app.include_router(telemetry_router)
app.include_router(reflection_router)
app.include_router(predict_router)
app.include_router(projects_router)
app.include_router(tree_router)
app.include_router(web_router)

app.mount("/static", StaticFiles(directory="static"), name="static")
