"""Stage 7 — User-facing stateful agent session.

:class:`PhysicsAgentSession` is the top-level API for deploying the
autonomous agent in a production setting.  Each user gets their own
session object, which:

* Persists the fitted :class:`~physml.estimator.PhysicsPredictor` between
  Python sessions (via ``joblib``).
* Wraps the low-level :class:`~physml.agent.PhysicsAgent` behind a simple
  ``query / feedback / report`` interface.
* Tracks session-level metadata (``user_id``, ``session_id``,
  ``last_seen_timestamp``, ``n_queries``).

Typical usage
-------------
::

    session = PhysicsAgentSession(user_id="alice")
    # First call — model must be pre-trained or seeded via train()
    session.train(X_seed, y_seed)

    result = session.query(X_new)
    if result["needs_label"]:
        session.feedback(X_new, y_true)

    print(session.report())
    session.save()   # persist to disk

    # Next Python session — restore instantly
    session2 = PhysicsAgentSession.load("alice")
    result2 = session2.query(X_new2)
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

import numpy as np


class PhysicsAgentSession:
    """Stateful, per-user agent session with persistence.

    Parameters
    ----------
    user_id : str
        Unique identifier for this user / model.  Used as the default file
        name when persisting to ``model_dir``.
    model_dir : str or Path, default "~/.physml_agents"
        Directory where sessions are saved / loaded.
    predictor_kwargs : dict or None
        Keyword arguments forwarded to :class:`~physml.estimator.PhysicsPredictor`
        when a new predictor is created.  Ignored if a predictor is provided
        directly via ``predictor=``.
    predictor : PhysicsPredictor or None
        Pre-built predictor.  When ``None`` (default), a new
        ``PhysicsPredictor(backend="neural")`` is created.
    uncertainty_threshold : float, default 0.35
        Forwarded to :class:`~physml.agent.PhysicsAgent`.
    ewc_lambda : float, default 0.4
        Forwarded to :class:`~physml.agent.PhysicsAgent`.
    """

    def __init__(
        self,
        user_id: str,
        model_dir: str | Path = "~/.physml_agents",
        predictor_kwargs: dict[str, Any] | None = None,
        predictor: Any = None,
        uncertainty_threshold: float = 0.35,
        ewc_lambda: float = 0.4,
    ) -> None:
        self.user_id = str(user_id)
        self.model_dir = Path(model_dir).expanduser()
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.ewc_lambda = float(ewc_lambda)

        # Build or accept predictor
        if predictor is not None:
            self._predictor = predictor
        else:
            from physml.estimator import PhysicsPredictor
            kwargs = dict(predictor_kwargs or {})
            kwargs.setdefault("backend", "neural")
            kwargs.setdefault("n_cycles", 20)
            self._predictor = PhysicsPredictor(**kwargs)

        self._agent: Any = None  # built lazily after first fit
        self._fitted: bool = False

        # Session-level metadata
        self.session_id: str = str(uuid.uuid4())
        self.created_at: str = _now()
        self.last_seen: str = _now()
        self.n_queries: int = 0
        self.n_feedbacks: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: Any, y: Any) -> "PhysicsAgentSession":
        """Fit the underlying predictor on seed data.

        Must be called at least once before :meth:`query`.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        self
        """
        self._predictor.fit(np.atleast_2d(X), np.atleast_1d(y))
        self._fitted = True
        self._build_agent()
        self.last_seen = _now()
        return self

    # ------------------------------------------------------------------
    # Query / feedback / report
    # ------------------------------------------------------------------

    def query(self, X: Any) -> dict[str, Any]:
        """Predict for a new sample (or batch).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        dict with keys:
            prediction  — model output (None if action=="ask").
            confidence  — float in [0, 1].
            action      — "predict" | "abstain" | "ask".
            needs_label — bool shortcut.
            session_id  — current session UUID.
            n_queries   — total queries this session.
        """
        if not self._fitted:
            raise RuntimeError(
                "Session is not trained yet.  Call train(X_seed, y_seed) first."
            )
        self.n_queries += 1
        self.last_seen = _now()
        from physml.agent import AgentAction
        action: AgentAction = self._agent.observe(X)
        return {
            "prediction": action.prediction,
            "confidence": action.confidence,
            "action": action.action,
            "needs_label": action.needs_label,
            "session_id": self.session_id,
            "n_queries": self.n_queries,
            "metadata": action.metadata,
        }

    def feedback(self, X: Any, y_true: Any) -> "PhysicsAgentSession":
        """Provide a ground-truth label to improve the model.

        Parameters
        ----------
        X : array-like
        y_true : array-like

        Returns
        -------
        self
        """
        if not self._fitted:
            raise RuntimeError(
                "Session is not trained yet.  Call train(X_seed, y_seed) first."
            )
        self.n_feedbacks += 1
        self.last_seen = _now()
        self._agent.reward(X, y_true, immediate=True)
        return self

    def report(self) -> dict[str, Any]:
        """Return a summary of this session.

        Returns
        -------
        dict with keys:
            user_id, session_id, created_at, last_seen,
            n_queries, n_feedbacks, agent_report (sub-dict).
        """
        agent_report = self._agent.report() if self._agent is not None else {}
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "n_queries": self.n_queries,
            "n_feedbacks": self.n_feedbacks,
            "agent_report": agent_report,
        }

    # ------------------------------------------------------------------
    # Persistence (Stage 6/7)
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the session to disk.

        Parameters
        ----------
        path : str, Path, or None
            If None, saves to ``<model_dir>/<user_id>.pkl``.

        Returns
        -------
        Path — the file path used.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for session persistence") from exc
        save_path = Path(path) if path is not None else self._default_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Store runtime state metadata for traceability
        self._predictor.runtime_state_.metadata.update({
            "user_id": self.user_id,
            "session_id": self.session_id,
            "last_seen_timestamp": self.last_seen,
        })
        joblib.dump(self, str(save_path))
        return save_path

    @classmethod
    def load(
        cls,
        user_id_or_path: str | Path,
        model_dir: str | Path = "~/.physml_agents",
    ) -> "PhysicsAgentSession":
        """Load a previously saved session.

        Parameters
        ----------
        user_id_or_path : str or Path
            Either a ``user_id`` (file looked up in ``model_dir``) or an
            explicit file path.
        model_dir : str or Path
            Ignored when ``user_id_or_path`` looks like a file path.

        Returns
        -------
        PhysicsAgentSession
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for session persistence") from exc

        p = Path(user_id_or_path)
        if not p.exists():
            # Treat as user_id
            p = Path(model_dir).expanduser() / f"{user_id_or_path}.pkl"
        obj = joblib.load(str(p))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected PhysicsAgentSession, got {type(obj)}")
        # Refresh session bookkeeping after loading
        obj.last_seen = _now()
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_agent(self) -> None:
        from physml.agent import PhysicsAgent
        self._agent = PhysicsAgent(
            self._predictor,
            uncertainty_threshold=self.uncertainty_threshold,
            ewc_lambda=self.ewc_lambda,
        )

    def _default_path(self) -> Path:
        return self.model_dir / f"{self.user_id}.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
