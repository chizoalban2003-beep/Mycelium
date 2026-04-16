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

from typing import Any

# FastAPI and pydantic are optional dependencies — import lazily so the
# rest of physml is usable without them.
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    HTTPException = RuntimeError  # type: ignore

    class BaseModel:  # type: ignore
        pass


_sessions: dict[str, Any] = {}  # user_id → PhysicsAgentSession


def _get_or_create_session(user_id: str) -> Any:
    from physml.agent_api import PhysicsAgentSession
    if user_id not in _sessions:
        _sessions[user_id] = PhysicsAgentSession(user_id=user_id)
    return _sessions[user_id]


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

    app = FastAPI(
        title="PhysML / myco API",
        description="REST microservice for the Mycelium autonomous learning agent.",
        version="1.0.0",
    )

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
        session = _sessions[user_id]
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

        for session in _sessions.values():
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
        ]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")

    # -----------------------------------------------------------------------
    # Stage 72 — Real-Time WebSocket prediction endpoint
    # -----------------------------------------------------------------------

    try:
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
                            r = session.query(X[i : i + 1])
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

    return app


# Expose a module-level ``app`` that can be referenced by uvicorn.
# Created lazily so importing physml.server does not fail when fastapi is absent.
try:
    app = create_app()
except ImportError:
    app = None  # type: ignore
