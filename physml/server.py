"""Stage 18 — FastAPI REST microservice for PhysML / myco.

Exposes four HTTP endpoints wrapping :class:`~physml.agent_api.PhysicsAgentSession`:

* ``POST /train``       — Train or retrain an agent on labelled data.
* ``POST /query``       — Predict for a new sample (or batch).
* ``POST /feedback``    — Provide a ground-truth label (online learning).
* ``GET  /report``      — Return session activity summary.
* ``WS   /ws/predict``  — Stage 72: real-time WebSocket prediction endpoint.

Usage
-----
Start the server (requires ``fastapi`` and ``uvicorn``):

::

    uvicorn physml.server:app --reload

Or programmatically:

::

    import uvicorn
    from physml.server import app
    uvicorn.run(app, host="0.0.0.0", port=8000)

Example requests:

::

    curl -X POST http://localhost:8000/train \\
         -H 'Content-Type: application/json' \\
         -d '{"user_id": "alice", "X": [[1,2],[3,4]], "y": [0, 1]}'

    curl -X POST http://localhost:8000/query \\
         -H 'Content-Type: application/json' \\
         -d '{"user_id": "alice", "X": [[1.5, 2.5]]}'

    curl http://localhost:8000/report?user_id=alice

Stage 72 — WebSocket real-time prediction:

::

    # Connect to ws://localhost:8000/ws/predict?user_id=alice
    # Send:  {"X": [[1.5, 2.5]]}
    # Recv:  {"prediction": [1], "confidence": 0.82, "user_id": "alice"}
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

from physml._log import configure_logging, get_logger

configure_logging()
_logger = get_logger(__name__)

_SESSION_TTL = 3600  # seconds before an idle session is evicted

# ---------------------------------------------------------------------------
# JWT auth helpers (stdlib-only, no PyJWT dependency)
# ---------------------------------------------------------------------------

import base64

_JWT_SECRET = os.environ.get("MYCELIUM_SECRET", "mycelium-dev-secret-change-me")
_JWT_EXPIRY = 86400  # 24 hours


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))


def _create_token(user_id: str) -> str:
    header = _b64_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64_encode(
        json.dumps({"sub": user_id, "exp": int(time.time()) + _JWT_EXPIRY}).encode()
    )
    sig_input = f"{header}.{payload}".encode()
    sig = hmac.new(_JWT_SECRET.encode(), sig_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64_encode(sig)}"


def _verify_token(token: str) -> str:
    """Verify token and return user_id, or raise ValueError."""
    try:
        header, payload, sig = token.split(".")
    except ValueError:
        raise ValueError("Malformed token")
    sig_input = f"{header}.{payload}".encode()
    expected = hmac.new(_JWT_SECRET.encode(), sig_input, hashlib.sha256).digest()
    if not hmac.compare_digest(_b64_decode(sig), expected):
        raise ValueError("Invalid signature")
    data = json.loads(_b64_decode(payload))
    if data.get("exp", 0) < time.time():
        raise ValueError("Token expired")
    return str(data["sub"])


# FastAPI and pydantic are optional dependencies — import lazily so the
# rest of physml is usable without them.
try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
    _bearer = HTTPBearer(auto_error=False)
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    HTTPException = RuntimeError  # type: ignore
    Depends = None  # type: ignore

    class BaseModel:  # type: ignore
        pass

    class HTTPAuthorizationCredentials:  # type: ignore
        credentials: str = ""


_sessions: dict[str, tuple[Any, float]] = {}  # user_id → (session, last_access_ts)


def _evict_stale_sessions() -> None:
    cutoff = time.time() - _SESSION_TTL
    stale = [uid for uid, (_, ts) in _sessions.items() if ts < cutoff]
    for uid in stale:
        _sessions.pop(uid, None)


def _get_or_create_session(user_id: str) -> Any:
    from physml.agent_api import PhysicsAgentSession
    _evict_stale_sessions()
    if user_id not in _sessions:
        _sessions[user_id] = (PhysicsAgentSession(user_id=user_id), time.time())
    else:
        session, _ = _sessions[user_id]
        _sessions[user_id] = (session, time.time())
    return _sessions[user_id][0]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    user_id: str
    X: list[list[float]]
    y: list[Any]


class QueryRequest(BaseModel):
    user_id: str
    X: list[list[float]]


class FeedbackRequest(BaseModel):
    user_id: str
    X: list[list[float]]
    y: list[Any]


class LoginRequest(BaseModel):
    user_id: str
    password: str = ""  # placeholder — extend with real auth as needed


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class GoalCreateRequest(BaseModel):
    description: str
    run_immediately: bool = False


class ScheduleCreateRequest(BaseModel):
    description: str
    schedule: str = "daily"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> Any:
    """Create and return the FastAPI application.

    Returns
    -------
    FastAPI app instance
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi and uvicorn are required for the REST API server. "
            "Install them with: pip install fastapi uvicorn"
        )

    import numpy as np
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse

    app = FastAPI(
        title="Mycelium API",
        description="REST + WebSocket API for the Mycelium local AI companion.",
        version="0.29.0",
    )

    # -----------------------------------------------------------------------
    # Auth dependency
    # -----------------------------------------------------------------------

    def get_current_user(
        creds: HTTPAuthorizationCredentials = Depends(_bearer),
    ) -> str:
        """Extract user_id from Bearer token, or return 'anonymous' if absent."""
        if creds is None:
            return "anonymous"
        try:
            return _verify_token(creds.credentials)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

    # -----------------------------------------------------------------------
    # Auth endpoints
    # -----------------------------------------------------------------------

    @app.post("/auth/token")
    def login(req: LoginRequest) -> dict:
        """Issue a JWT bearer token.

        In production, replace the stub password check with real credentials.
        """
        # Stub: any user_id with any (or empty) password is accepted locally.
        # Set MYCELIUM_SECRET env var to make tokens non-forgeable.
        token = _create_token(req.user_id)
        return {"access_token": token, "token_type": "bearer", "user_id": req.user_id}

    # -----------------------------------------------------------------------
    # Companion chat endpoint
    # -----------------------------------------------------------------------

    _companion: Any = None  # module-level singleton

    def _get_companion() -> Any:
        nonlocal _companion
        if _companion is None:
            try:
                from physml.companion import MyceliumCompanion
                _companion = MyceliumCompanion()
                _companion.start()
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail=f"Companion unavailable: {exc}"
                )
        return _companion

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        """Send a message to the Mycelium companion and get a response."""
        companion = _get_companion()
        try:
            response = companion.chat(req.message)
            return {"response": response, "user_id": req.user_id}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/companion/status")
    def companion_status() -> dict:
        """Return companion system status."""
        companion = _get_companion()
        return companion.status()

    @app.get("/", response_class=HTMLResponse)
    def web_ui() -> str:
        """Serve the built-in web chat UI."""
        return _WEB_UI_HTML

    @app.post("/train")
    def train(req: TrainRequest) -> dict:
        """Train (or retrain) the agent for a user."""
        session = _get_or_create_session(req.user_id)
        X = np.array(req.X, dtype=float)
        y = np.array(req.y)
        try:
            session.train(X, y)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {
            "status": "ok",
            "user_id": req.user_id,
            "n_samples": len(y),
        }

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        """Predict for a new sample or batch."""
        session = _get_or_create_session(req.user_id)
        if not session._fitted:
            raise HTTPException(
                status_code=400,
                detail=f"Agent for user {req.user_id!r} is not trained yet. "
                       "Call POST /train first.",
            )
        X = np.array(req.X, dtype=float)
        results = []
        for i in range(len(X)):
            r = session.query(X[i : i + 1])
            results.append({
                "prediction": r["prediction"].tolist() if hasattr(r["prediction"], "tolist") else r["prediction"],
                "confidence": float(r["confidence"]),
                "action": r["action"],
                "needs_label": r["needs_label"],
            })
        return {
            "user_id": req.user_id,
            "session_id": session.session_id,
            "n_queries": session.n_queries,
            "results": results,
        }

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        """Provide ground-truth labels for online learning."""
        session = _get_or_create_session(req.user_id)
        if not session._fitted:
            raise HTTPException(
                status_code=400,
                detail=f"Agent for user {req.user_id!r} is not trained yet.",
            )
        X = np.array(req.X, dtype=float)
        y = np.array(req.y)
        for i in range(len(X)):
            try:
                session.feedback(X[i : i + 1], y[i : i + 1])
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
        return {
            "status": "ok",
            "user_id": req.user_id,
            "n_feedbacks": session.n_feedbacks,
        }

    @app.get("/report")
    def report(user_id: str) -> dict:
        """Return a summary of a user's session."""
        if user_id not in _sessions:
            raise HTTPException(
                status_code=404,
                detail=f"No session found for user {user_id!r}.",
            )
        session, _ = _sessions[user_id]
        return session.report()

    @app.get("/metrics")
    def metrics() -> Any:
        """Prometheus text-format metrics endpoint (Stage 27).

        Exposes aggregate counters across all active sessions::

            physml_n_observations_total   — total observe() calls
            physml_oracle_calls_total     — total ask/reward calls
            physml_drift_events_total     — total drift-detection events
            physml_ask_rate               — mean ask-rate across sessions
            physml_active_sessions        — number of active user sessions

        Drop into any Grafana / Prometheus stack::

            scrape_configs:
              - job_name: physml
                static_configs:
                  - targets: ["localhost:8000"]
        """
        from fastapi.responses import PlainTextResponse

        n_obs = 0
        n_oracle = 0
        n_drift = 0
        ask_rates: list[float] = []

        for session, _ in _sessions.values():
            try:
                r = session.report()
                agent_r = r.get("agent", r)
                n_obs += int(agent_r.get("n_observations", 0))
                n_oracle += int(agent_r.get("n_rewards", 0))
                n_drift += int(agent_r.get("n_drifts_detected", 0))
                ask_rates.append(float(agent_r.get("ask_rate", 0.0)))
            except Exception:
                pass

        mean_ask_rate = float(np.mean(ask_rates)) if ask_rates else 0.0
        n_sessions = len(_sessions)

        # Goal-engine + scheduler metrics
        goals_pending = goals_active = goals_completed = goals_failed = goals_blocked = 0
        scheduler_total = scheduler_enabled = 0
        llm_calls = 0
        try:
            companion = _get_companion()
            if companion is not None:
                if getattr(companion, "goal_engine", None) is not None:
                    gs = companion.goal_engine.status()
                    goals_pending = gs.get("pending", 0)
                    goals_active = gs.get("active", 0)
                    goals_completed = gs.get("completed", 0)
                    goals_failed = gs.get("failed", 0)
                    goals_blocked = gs.get("blocked", 0)
                if getattr(companion, "scheduler", None) is not None:
                    ss = companion.scheduler.status()
                    scheduler_total = ss.get("total", 0)
                    scheduler_enabled = ss.get("enabled", 0)
                if getattr(companion, "llm", None) is not None:
                    llm_calls = int(getattr(companion.llm, "_call_count", 0))
        except Exception:
            pass

        lines = [
            "# HELP physml_n_observations_total Total observe() calls across all sessions",
            "# TYPE physml_n_observations_total counter",
            f"physml_n_observations_total {n_obs}",
            "# HELP physml_oracle_calls_total Total oracle (reward) calls across all sessions",
            "# TYPE physml_oracle_calls_total counter",
            f"physml_oracle_calls_total {n_oracle}",
            "# HELP physml_drift_events_total Total concept-drift events detected",
            "# TYPE physml_drift_events_total counter",
            f"physml_drift_events_total {n_drift}",
            "# HELP physml_ask_rate Mean ask-rate across active sessions",
            "# TYPE physml_ask_rate gauge",
            f"physml_ask_rate {mean_ask_rate:.6f}",
            "# HELP physml_active_sessions Number of active user sessions",
            "# TYPE physml_active_sessions gauge",
            f"physml_active_sessions {n_sessions}",
            "# HELP myco_goals_pending Goals waiting to be executed",
            "# TYPE myco_goals_pending gauge",
            f"myco_goals_pending {goals_pending}",
            "# HELP myco_goals_active Goals currently executing",
            "# TYPE myco_goals_active gauge",
            f"myco_goals_active {goals_active}",
            "# HELP myco_goals_completed_total Goals completed successfully",
            "# TYPE myco_goals_completed_total counter",
            f"myco_goals_completed_total {goals_completed}",
            "# HELP myco_goals_failed_total Goals that failed after all retries",
            "# TYPE myco_goals_failed_total counter",
            f"myco_goals_failed_total {goals_failed}",
            "# HELP myco_goals_blocked_total Goals blocked (max retries exceeded)",
            "# TYPE myco_goals_blocked_total counter",
            f"myco_goals_blocked_total {goals_blocked}",
            "# HELP myco_scheduler_total Total scheduled recurring goals",
            "# TYPE myco_scheduler_total gauge",
            f"myco_scheduler_total {scheduler_total}",
            "# HELP myco_scheduler_enabled Enabled scheduled recurring goals",
            "# TYPE myco_scheduler_enabled gauge",
            f"myco_scheduler_enabled {scheduler_enabled}",
            "# HELP myco_llm_calls_total Total LLM API calls made",
            "# TYPE myco_llm_calls_total counter",
            f"myco_llm_calls_total {llm_calls}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")

    # -----------------------------------------------------------------------
    # Stage 72 — Real-Time WebSocket prediction endpoint
    # -----------------------------------------------------------------------

    try:
        import asyncio as _asyncio
        from fastapi import WebSocket, WebSocketDisconnect
        import json as _json

        @app.websocket("/ws/predict")
        async def ws_predict(websocket: WebSocket, user_id: str = "default") -> None:
            """Real-time prediction via WebSocket.

            Protocol
            --------
            Client sends JSON messages of the form::

                {"X": [[f1, f2, ...]]}          # one or more samples

            Server replies with::

                {"prediction": [...], "confidence": [...],
                 "user_id": "...", "session_id": "..."}

            If the session is not yet trained, the server replies with::

                {"error": "not_trained"}

            The connection is kept alive until the client closes it.
            """
            await websocket.accept()
            try:
                while True:
                    raw = await websocket.receive_text()
                    try:
                        msg = _json.loads(raw)
                    except Exception:
                        await websocket.send_text(
                            _json.dumps({"error": "invalid_json"})
                        )
                        continue

                    x_list = msg.get("X")
                    if x_list is None:
                        await websocket.send_text(
                            _json.dumps({"error": "missing_X"})
                        )
                        continue

                    session = _get_or_create_session(user_id)
                    if not session._fitted:
                        await websocket.send_text(
                            _json.dumps({"error": "not_trained"})
                        )
                        continue

                    try:
                        X = np.array(x_list, dtype=float)
                        predictions = []
                        confidences = []
                        for i in range(len(X)):
                            r = await _asyncio.to_thread(session.query, X[i : i + 1])
                            pred = r["prediction"]
                            conf = float(r["confidence"])
                            predictions.append(
                                pred.tolist()
                                if hasattr(pred, "tolist")
                                else pred
                            )
                            confidences.append(conf)

                        await websocket.send_text(
                            _json.dumps(
                                {
                                    "prediction": predictions,
                                    "confidence": confidences,
                                    "user_id": user_id,
                                    "session_id": session.session_id,
                                }
                            )
                        )
                    except Exception as exc:
                        await websocket.send_text(
                            _json.dumps({"error": str(exc)})
                        )
            except WebSocketDisconnect:
                pass

    except ImportError:
        # WebSocket support not available (older fastapi / missing dependency)
        pass

    # -----------------------------------------------------------------------
    # Goals REST API (Stage 140)
    # -----------------------------------------------------------------------

    @app.get("/goals")
    def list_goals(status: str = None) -> dict:
        """List all goals, optionally filtered by status.

        Query params: ``?status=pending|active|completed|failed|blocked|cancelled``
        """
        companion = _get_companion()
        from physml.goal_engine import GoalStatus
        filt = None
        if status:
            try:
                filt = GoalStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown status {status!r}")
        goals = companion.goal_engine.goals(filt)
        return {
            "total": len(goals),
            "status_filter": status,
            "goals": [g.to_dict() for g in goals],
        }

    @app.post("/goals", status_code=201)
    def create_goal(req: GoalCreateRequest) -> dict:
        """Queue (or immediately run) a new goal."""
        companion = _get_companion()
        goal_id = companion.goal_engine.add_goal(
            req.description,
            run_immediately=req.run_immediately,
        )
        goal = companion.goal_engine.get(goal_id)
        return {
            "id": goal_id,
            "status": goal.status.value if goal else "queued",
            "message": f"Goal {'executed' if req.run_immediately else 'queued'}: {req.description[:60]}",
        }

    @app.get("/goals/{goal_id}")
    def get_goal(goal_id: str) -> dict:
        """Get a single goal by ID."""
        companion = _get_companion()
        goal = companion.goal_engine.get(goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail=f"Goal {goal_id!r} not found")
        return goal.to_dict()

    @app.delete("/goals/{goal_id}")
    def cancel_goal(goal_id: str) -> dict:
        """Cancel a pending or active goal."""
        companion = _get_companion()
        cancelled = companion.goal_engine.cancel_goal(goal_id)
        if not cancelled:
            raise HTTPException(
                status_code=404,
                detail=f"Goal {goal_id!r} not found or already terminal",
            )
        return {"id": goal_id, "status": "cancelled"}

    # -----------------------------------------------------------------------
    # Schedules REST API
    # -----------------------------------------------------------------------

    @app.get("/schedules")
    def list_schedules() -> dict:
        """List all scheduled recurring goals."""
        companion = _get_companion()
        return companion.scheduler.status()

    @app.post("/schedules", status_code=201)
    def create_schedule(req: ScheduleCreateRequest) -> dict:
        """Register a new recurring goal."""
        companion = _get_companion()
        try:
            sid = companion.scheduler.add(req.description, schedule=req.schedule)
            return {
                "id": sid,
                "description": req.description,
                "schedule": req.schedule,
                "message": f"Scheduled: {req.description[:60]!r} — {req.schedule}",
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/schedules/{schedule_id}")
    def remove_schedule(schedule_id: str) -> dict:
        """Remove a scheduled goal."""
        companion = _get_companion()
        removed = companion.scheduler.remove(schedule_id)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule {schedule_id!r} not found",
            )
        return {"id": schedule_id, "status": "removed"}

    # -----------------------------------------------------------------------
    # Streaming chat (SSE) — Stage 141
    # -----------------------------------------------------------------------

    try:
        import asyncio as _asyncio
        from fastapi.responses import StreamingResponse as _StreamingResponse
        import json as _json_sse

        @app.post("/chat/stream")
        async def chat_stream(req: ChatRequest) -> _StreamingResponse:
            """Stream a chat response as Server-Sent Events.

            Each event is a JSON object::

                data: {"token": "Hello"}
                data: {"token": " world"}
                data: [DONE]

            The client should append tokens as they arrive.
            """
            companion = _get_companion()

            async def _generate():
                try:
                    async for chunk in companion.chat_stream(req.message):
                        payload = _json_sse.dumps({"token": chunk})
                        yield f"data: {payload}\n\n"
                except Exception as exc:
                    err = _json_sse.dumps({"error": str(exc)})
                    yield f"data: {err}\n\n"
                finally:
                    yield "data: [DONE]\n\n"

            return _StreamingResponse(
                _generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    except ImportError:
        pass

    # -----------------------------------------------------------------------
    # Daily digest endpoint — Stage 142
    # -----------------------------------------------------------------------

    @app.get("/digest")
    def get_digest() -> dict:
        """Return the daily activity digest."""
        companion = _get_companion()
        text = companion.daily_digest()
        return {"digest": text}

    # -----------------------------------------------------------------------
    # Voice loop endpoints — Stage 145
    # -----------------------------------------------------------------------

    @app.get("/voice/status")
    def voice_status() -> dict:
        """Return voice loop status."""
        companion = _get_companion()
        vl = getattr(companion, "voice_loop", None)
        return {
            "running": vl is not None and getattr(vl, "_running", False),
            "available": True,
        }

    @app.post("/voice/start")
    def voice_start(wake_word: str = "", record_seconds: float = 5.0) -> dict:
        """Start the voice interaction loop."""
        companion = _get_companion()
        msg = companion.start_voice(
            wake_word=wake_word or None,
            record_seconds=record_seconds,
        )
        return {"message": msg}

    @app.post("/voice/stop")
    def voice_stop() -> dict:
        """Stop the voice interaction loop."""
        companion = _get_companion()
        msg = companion.stop_voice()
        return {"message": msg}

    # -----------------------------------------------------------------------
    # CommBridge status endpoint — Stage 143
    # -----------------------------------------------------------------------

    @app.get("/comm/status")
    def comm_status() -> dict:
        """Return communication channel configuration status."""
        companion = _get_companion()
        cb = getattr(companion, "comm_bridge", None)
        if cb is None:
            return {"error": "CommBridge not initialised"}
        return cb.status()

    # -----------------------------------------------------------------------
    # DesktopBridge status endpoint — Stage 144
    # -----------------------------------------------------------------------

    @app.get("/desktop/status")
    def desktop_status() -> dict:
        """Return desktop automation capability status."""
        companion = _get_companion()
        db = getattr(companion, "desktop_bridge", None)
        if db is None:
            return {"error": "DesktopBridge not initialised"}
        return db.status()

    return app


# Expose a module-level ``app`` that can be referenced by uvicorn.
# Created lazily so importing physml.server does not fail when fastapi is absent.
try:
    app = create_app()
except ImportError:
    app = None  # type: ignore


# ---------------------------------------------------------------------------
# Built-in Web Chat UI (Stage 128)
# ---------------------------------------------------------------------------

_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mycelium — Local AI Companion</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#0f1117;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
  header{padding:12px 20px;background:#1a1d2e;border-bottom:1px solid #2d3148;
         display:flex;align-items:center;gap:10px}
  header h1{font-size:1.05rem;font-weight:600;color:#a78bfa;flex:1}
  .hbtn{background:none;border:1px solid #2d3148;color:#94a3b8;border-radius:6px;
        padding:4px 10px;font-size:.75rem;cursor:pointer;transition:all .15s}
  .hbtn:hover{border-color:#a78bfa;color:#a78bfa}
  #main{display:flex;flex:1;overflow:hidden}
  #chat-panel{flex:1;display:flex;flex-direction:column;overflow:hidden}
  #goals-panel{width:260px;background:#111827;border-left:1px solid #1e2235;
               display:flex;flex-direction:column;transition:width .2s}
  #goals-panel.hidden{width:0;overflow:hidden;border:none}
  #goals-hdr{padding:10px 14px;background:#1a1d2e;border-bottom:1px solid #2d3148;
             font-size:.8rem;font-weight:600;color:#a78bfa;display:flex;align-items:center;gap:6px}
  #goals-list{flex:1;overflow-y:auto;padding:8px}
  .goal-item{padding:7px 9px;margin-bottom:5px;border-radius:7px;background:#1e2235;
             font-size:.78rem;line-height:1.4;border-left:3px solid #374151}
  .goal-item.completed{border-color:#10b981}
  .goal-item.failed,.goal-item.blocked{border-color:#ef4444}
  .goal-item.active{border-color:#f59e0b}
  .goal-item.pending{border-color:#6366f1}
  .goal-status{font-size:.68rem;color:#64748b;margin-bottom:2px;text-transform:uppercase}
  #add-goal{padding:8px;border-top:1px solid #1e2235}
  #goal-input{width:100%;background:#0f1117;border:1px solid #2d3148;border-radius:6px;
              color:#e2e8f0;padding:6px 8px;font-size:.78rem;outline:none}
  #goal-input:focus{border-color:#6366f1}
  #chat{flex:1;overflow-y:auto;padding:14px 18px;display:flex;flex-direction:column;gap:9px}
  .msg{max-width:74%;padding:10px 13px;border-radius:12px;line-height:1.55;font-size:.88rem;
       white-space:pre-wrap;word-break:break-word}
  .user{align-self:flex-end;background:#4f46e5;color:#fff;border-bottom-right-radius:2px}
  .agent{align-self:flex-start;background:#1e2235;color:#cbd5e1;border-bottom-left-radius:2px}
  .agent strong{color:#a78bfa}
  footer{padding:10px 14px;background:#1a1d2e;border-top:1px solid #2d3148;display:flex;gap:7px}
  #input{flex:1;background:#0f1117;border:1px solid #2d3148;border-radius:8px;
         color:#e2e8f0;padding:9px 13px;font-size:.88rem;outline:none;resize:none;
         height:40px;max-height:120px;overflow:auto}
  #input:focus{border-color:#4f46e5}
  button{background:#4f46e5;color:#fff;border:none;border-radius:8px;
         padding:0 16px;cursor:pointer;font-size:.88rem;font-weight:500;
         transition:background .15s;white-space:nowrap}
  button:hover{background:#4338ca}
  button:disabled{background:#374151;cursor:not-allowed}
  .thinking{color:#64748b;font-style:italic;font-size:.82rem;padding:5px 13px;align-self:flex-start}
  #digest-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;
                align-items:center;justify-content:center}
  #digest-modal.show{display:flex}
  #digest-box{background:#1a1d2e;border:1px solid #2d3148;border-radius:12px;
              padding:20px 24px;max-width:520px;width:90%;max-height:70vh;overflow-y:auto}
  #digest-box h2{color:#a78bfa;font-size:1rem;margin-bottom:12px}
  #digest-text{font-size:.83rem;white-space:pre-wrap;line-height:1.6;color:#cbd5e1}
  #digest-close{margin-top:14px;background:#374151;padding:6px 16px;font-size:.82rem}
</style>
</head>
<body>
<header>
  <h1>&#x1F344; Mycelium</h1>
  <button class="hbtn" onclick="toggleGoals()">Goals</button>
  <button class="hbtn" onclick="showDigest()">Digest</button>
</header>
<div id="main">
  <div id="chat-panel">
    <div id="chat">
      <div class="msg agent"><strong>Mycelium</strong><br>Hello! I am your local AI companion. Everything runs on your device — nothing leaves. Ask me to predict, train on a CSV, run a goal, or just chat.</div>
    </div>
    <footer>
      <textarea id="input" placeholder="Type a message… (Enter to send)" rows="1"></textarea>
      <button id="send" onclick="sendMsg()">Send</button>
    </footer>
  </div>
  <div id="goals-panel" class="hidden">
    <div id="goals-hdr">
      &#x26A1; Goals
      <span id="goals-count" style="margin-left:auto;color:#64748b;font-weight:400">—</span>
    </div>
    <div id="goals-list"></div>
    <div id="add-goal">
      <input id="goal-input" placeholder="Add goal…" onkeydown="if(event.key==='Enter')addGoal()">
    </div>
  </div>
</div>
<div id="digest-modal">
  <div id="digest-box">
    <h2>&#x1F4C5; Daily Digest</h2>
    <div id="digest-text">Loading…</div>
    <button id="digest-close" onclick="closeDigest()">Close</button>
  </div>
</div>
<script>
const chat=document.getElementById('chat');
const input=document.getElementById('input');
const btn=document.getElementById('send');
let goalsVisible=false;

function escHtml(t){return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');}
function addMsg(text,cls,streaming){
  const d=document.createElement('div');
  d.className='msg '+cls;
  if(cls==='agent'){d.innerHTML='<strong>Mycelium</strong><br>'+(streaming?'':escHtml(text));}
  else{d.textContent=text;}
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;
  return d;
}
async function sendMsg(){
  const text=input.value.trim();if(!text)return;
  input.value='';input.style.height='40px';btn.disabled=true;
  addMsg(text,'user');
  const thk=document.createElement('div');thk.className='thinking';thk.textContent='Thinking…';
  chat.appendChild(thk);chat.scrollTop=chat.scrollHeight;
  const agentDiv=addMsg('','agent',true);
  const contentSpan=agentDiv.querySelector('br').nextSibling||agentDiv.appendChild(document.createTextNode(''));
  let buf='';
  try{
    const r=await fetch('/chat/stream',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,user_id:'web_user'})});
    if(!r.ok){
      const e=await r.json();
      thk.remove();agentDiv.innerHTML='<strong>Mycelium</strong><br>'+escHtml(e.detail||'Error');
      btn.disabled=false;input.focus();return;
    }
    thk.remove();
    const reader=r.body.getReader();const dec=new TextDecoder();
    let partial='';
    while(true){
      const {done,value}=await reader.read();if(done)break;
      partial+=dec.decode(value,{stream:true});
      const lines=partial.split('\\n');partial=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        const raw=line.slice(6);
        if(raw==='[DONE]')break;
        try{const {token,error}=JSON.parse(raw);
          if(error){buf+='[Error: '+error+']';}else{buf+=token;}
          agentDiv.innerHTML='<strong>Mycelium</strong><br>'+escHtml(buf);
          chat.scrollTop=chat.scrollHeight;
        }catch(e){}
      }
    }
    if(!buf)agentDiv.innerHTML='<strong>Mycelium</strong><br>(no response)';
  }catch(e){thk.remove();agentDiv.innerHTML='<strong>Mycelium</strong><br>Connection error: '+e;}
  btn.disabled=false;input.focus();
  if(goalsVisible)refreshGoals();
}
input.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}
  setTimeout(()=>{input.style.height='40px';input.style.height=Math.min(input.scrollHeight,120)+'px';},0);
});
function toggleGoals(){
  goalsVisible=!goalsVisible;
  document.getElementById('goals-panel').classList.toggle('hidden',!goalsVisible);
  if(goalsVisible)refreshGoals();
}
async function refreshGoals(){
  try{
    const r=await fetch('/goals');const data=await r.json();
    const list=document.getElementById('goals-list');list.innerHTML='';
    document.getElementById('goals-count').textContent=data.total;
    const goals=data.goals||[];
    if(!goals.length){list.innerHTML='<div style="color:#64748b;font-size:.78rem;padding:10px">No goals yet.</div>';return;}
    // Show most recent first
    goals.slice().reverse().slice(0,20).forEach(g=>{
      const d=document.createElement('div');
      d.className='goal-item '+g.status;
      const done=g.steps?g.steps.filter(s=>s.status==='ok').length:0;
      const total=g.steps?g.steps.length:0;
      d.innerHTML='<div class="goal-status">'+g.status+(total?' '+done+'/'+total:'')+'</div>'+
                  escHtml(g.description.slice(0,70));
      list.appendChild(d);
    });
  }catch(e){document.getElementById('goals-list').textContent='Error loading goals.';}
}
async function addGoal(){
  const inp=document.getElementById('goal-input');
  const desc=inp.value.trim();if(!desc)return;
  inp.value='';
  try{
    await fetch('/goals',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({description:desc,run_immediately:false})});
    refreshGoals();
  }catch(e){alert('Error adding goal: '+e);}
}
async function showDigest(){
  document.getElementById('digest-modal').classList.add('show');
  document.getElementById('digest-text').textContent='Loading…';
  try{
    const r=await fetch('/digest');const data=await r.json();
    document.getElementById('digest-text').textContent=data.digest||'No digest available.';
  }catch(e){document.getElementById('digest-text').textContent='Error: '+e;}
}
function closeDigest(){document.getElementById('digest-modal').classList.remove('show');}
</script>
</body>
</html>"""
