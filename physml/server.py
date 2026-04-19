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
  header{padding:14px 24px;background:#1a1d2e;border-bottom:1px solid #2d3148;
         display:flex;align-items:center;gap:12px}
  header h1{font-size:1.1rem;font-weight:600;color:#a78bfa}
  header span{font-size:.75rem;color:#64748b;background:#1e2235;
              padding:2px 8px;border-radius:999px}
  #chat{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:10px}
  .msg{max-width:72%;padding:10px 14px;border-radius:12px;line-height:1.55;font-size:.9rem;
       white-space:pre-wrap;word-break:break-word}
  .user{align-self:flex-end;background:#4f46e5;color:#fff;border-bottom-right-radius:2px}
  .agent{align-self:flex-start;background:#1e2235;color:#cbd5e1;border-bottom-left-radius:2px}
  .agent strong{color:#a78bfa}
  footer{padding:12px 16px;background:#1a1d2e;border-top:1px solid #2d3148;
         display:flex;gap:8px}
  #input{flex:1;background:#0f1117;border:1px solid #2d3148;border-radius:8px;
         color:#e2e8f0;padding:10px 14px;font-size:.9rem;outline:none;resize:none;
         height:42px;max-height:120px;overflow:auto}
  #input:focus{border-color:#4f46e5}
  button{background:#4f46e5;color:#fff;border:none;border-radius:8px;
         padding:0 18px;cursor:pointer;font-size:.9rem;font-weight:500;
         transition:background .15s}
  button:hover{background:#4338ca}
  button:disabled{background:#374151;cursor:not-allowed}
  .thinking{color:#64748b;font-style:italic;font-size:.85rem;padding:6px 14px}
</style>
</head>
<body>
<header>
  <h1>&#x1F344; Mycelium</h1>
  <span>local AI companion</span>
</header>
<div id="chat">
  <div class="msg agent"><strong>Mycelium</strong><br>Hello! I am your local AI companion. Everything runs on your device — your data never leaves. Try asking me to predict, train on a CSV, or just chat!</div>
</div>
<footer>
  <textarea id="input" placeholder="Type a message…" rows="1"></textarea>
  <button id="send" onclick="sendMsg()">Send</button>
</footer>
<script>
const chat=document.getElementById('chat');
const input=document.getElementById('input');
const btn=document.getElementById('send');
function addMsg(text,cls){
  const d=document.createElement('div');
  d.className='msg '+cls;
  if(cls==='agent'){d.innerHTML='<strong>Mycelium</strong><br>'+escHtml(text);}
  else{d.textContent=text;}
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
  return d;
}
function escHtml(t){return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');}
async function sendMsg(){
  const text=input.value.trim();
  if(!text)return;
  input.value='';input.style.height='42px';
  btn.disabled=true;
  addMsg(text,'user');
  const thinking=document.createElement('div');
  thinking.className='thinking';thinking.textContent='Thinking…';
  chat.appendChild(thinking);chat.scrollTop=chat.scrollHeight;
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,user_id:'web_user'})});
    const data=await r.json();
    thinking.remove();
    addMsg(data.response||data.detail||'Error','agent');
  }catch(e){thinking.remove();addMsg('Connection error: '+e,'agent');}
  btn.disabled=false;input.focus();
}
input.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}
  setTimeout(()=>{input.style.height='42px';input.style.height=Math.min(input.scrollHeight,120)+'px';},0);
});
</script>
</body>
</html>"""
