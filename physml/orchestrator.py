"""Stage 46 — AgentOrchestrator: multi-specialist coordinator.

Provides :class:`AgentOrchestrator`, a coordinator that routes incoming
requests to the most appropriate specialist agent based on embedding similarity
and past performance.

Architecture
------------
::

    AgentOrchestrator
    ├── physics_specialist  (MyceliumAgent — numeric/tabular data)
    ├── tool_specialist     (AutonomousLoop — text goal + tool calls)
    └── fallback            (default agent when nothing matches well)

Routing logic
~~~~~~~~~~~~~
1. Featurize the input (via :class:`~physml.featurizer.Featurizer`).
2. Compute cosine similarity between the input embedding and each
   specialist's prototype vector (set during :meth:`register_specialist`).
3. Boost scores with empirical success rates from
   :class:`~physml.memory.EpisodicMemory` if attached.
4. Dispatch to the highest-scoring specialist.

The orchestrator itself is stateless across requests — all state is owned
by the individual specialists and the shared episodic memory.

Design rationale
----------------
The missing piece for LLM-competitiveness is *composability*: a single
ML predictor only handles one task type.  An orchestrator lets Mycelium
combine its physics-inspired numeric predictor, its tool-calling loop, and
future specialists (vision, code, etc.) under one roof — mirroring how
multi-modal LLMs route across specialised sub-networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from physml.featurizer import Featurizer
    from physml.memory import EpisodicMemory


@dataclass
class Specialist:
    """A named specialist agent registered with an :class:`AgentOrchestrator`.

    Attributes
    ----------
    name : str
        Unique specialist identifier.
    description : str
        Human-readable description of what this specialist handles.
    handler : Callable[[Any], Any]
        Function that processes a request and returns a response.
    prototype : np.ndarray or None
        Representative feature vector for routing.  Set automatically by
        :meth:`AgentOrchestrator.register_specialist` if not provided.
    """

    name: str
    description: str
    handler: Callable[[Any], Any]
    prototype: np.ndarray | None = None


@dataclass
class OrchestratorResult:
    """Result of a single :meth:`AgentOrchestrator.route` call.

    Attributes
    ----------
    specialist_name : str
        Name of the chosen specialist.
    confidence : float
        Routing confidence score (0 — 1).
    response : Any
        Output from the specialist's handler.
    ranked_alternatives : list[str]
        Other specialists considered, in descending order of score.
    """

    specialist_name: str
    confidence: float
    response: Any
    ranked_alternatives: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """Multi-specialist routing coordinator.

    Parameters
    ----------
    featurizer : Featurizer
        Fitted featurizer used to embed inputs and specialist descriptions.
    memory : EpisodicMemory or None, default None
        Shared episodic memory.  When provided, specialist success rates are
        inferred from stored (action=specialist_name, outcome) pairs.
    memory_weight : float, default 0.25
        Weight of memory-derived success rate in the final routing score.
    min_confidence : float, default 0.0
        Minimum confidence required to dispatch.  When all specialists score
        below this, the fallback specialist is used (if registered).
    """

    def __init__(
        self,
        featurizer: "Featurizer",
        memory: "EpisodicMemory | None" = None,
        memory_weight: float = 0.25,
        min_confidence: float = 0.0,
    ) -> None:
        self.featurizer = featurizer
        self.memory = memory
        self.memory_weight = float(np.clip(memory_weight, 0.0, 1.0))
        self.min_confidence = float(min_confidence)

        self._specialists: dict[str, Specialist] = {}
        self._fallback: Specialist | None = None

        self._n_routes: int = 0
        self._route_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_specialist(
        self,
        specialist: Specialist,
        *,
        fallback: bool = False,
    ) -> "AgentOrchestrator":
        """Register a specialist with the orchestrator.

        Parameters
        ----------
        specialist : Specialist
        fallback : bool, default False
            When ``True`` this specialist is also set as the fallback
            used when all other specialists score below ``min_confidence``.

        Returns
        -------
        self
        """
        # Auto-compute prototype from description if not provided
        if specialist.prototype is None:
            try:
                specialist.prototype = self.featurizer.transform([specialist.description])[0]
            except Exception:
                specialist.prototype = np.zeros(
                    self.featurizer.output_dim, dtype=np.float32
                )

        self._specialists[specialist.name] = specialist
        if fallback:
            self._fallback = specialist
        return self

    def set_fallback(self, name: str) -> "AgentOrchestrator":
        """Set an already-registered specialist as the fallback by name."""
        if name not in self._specialists:
            raise KeyError(f"Specialist '{name}' is not registered.")
        self._fallback = self._specialists[name]
        return self

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, request: Any) -> OrchestratorResult:
        """Route *request* to the most appropriate specialist.

        Parameters
        ----------
        request : str, np.ndarray, or any
            The input to process.  Strings are featurized directly.
            Arrays are assumed pre-featurized (1-D float32).
            Other types are converted via ``str(request)``.

        Returns
        -------
        OrchestratorResult

        Raises
        ------
        RuntimeError
            If no specialists are registered.
        """
        if not self._specialists:
            raise RuntimeError("No specialists registered. Call register_specialist() first.")

        request_vec = self._featurize_request(request)
        success_rates = self._compute_success_rates()
        scores = self._score_specialists(request_vec, success_rates)

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        best_name = ranked[0]
        best_score = float(scores[best_name])

        if best_score < self.min_confidence and self._fallback is not None:
            best_name = self._fallback.name
            best_score = float(scores.get(best_name, 0.0))

        alternatives = [n for n in ranked if n != best_name]
        specialist = self._specialists[best_name]

        try:
            response = specialist.handler(request)
        except Exception as exc:
            response = {"error": str(exc)}

        # Update routing stats
        self._n_routes += 1
        self._route_counts[best_name] = self._route_counts.get(best_name, 0) + 1

        # Store in memory (action = specialist name, outcome = 1.0 placeholder)
        if self.memory is not None:
            try:
                self.memory.store(
                    context=request_vec,
                    action=best_name,
                    outcome=1.0,
                )
            except Exception:
                pass

        return OrchestratorResult(
            specialist_name=best_name,
            confidence=best_score,
            response=response,
            ranked_alternatives=alternatives,
        )

    def report(self) -> dict:
        """Summary of routing activity."""
        return {
            "n_routes": self._n_routes,
            "route_counts": dict(self._route_counts),
            "registered_specialists": list(self._specialists.keys()),
            "fallback": self._fallback.name if self._fallback else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _featurize_request(self, request: Any) -> np.ndarray:
        if isinstance(request, np.ndarray):
            return request.astype(np.float32).ravel()
        text = request if isinstance(request, str) else str(request)
        try:
            return self.featurizer.transform([text])[0]
        except Exception:
            return np.zeros(self.featurizer.output_dim, dtype=np.float32)

    def _score_specialists(
        self,
        request_vec: np.ndarray,
        success_rates: dict[str, float],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, spec in self._specialists.items():
            sim = self._cosine(request_vec, spec.prototype)
            sr = success_rates.get(name, 0.5)
            scores[name] = (1.0 - self.memory_weight) * sim + self.memory_weight * sr
        return scores

    def _cosine(self, a: np.ndarray, b: np.ndarray | None) -> float:
        if b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _compute_success_rates(self) -> dict[str, float]:
        if self.memory is None or len(self.memory) == 0:
            return {}
        actions = list(self.memory._actions)
        outcomes = list(self.memory._outcomes)
        totals: dict[str, list[float]] = {}
        for act, out in zip(actions, outcomes):
            totals.setdefault(act, []).append(float(out))
        return {name: float(np.mean(vals)) for name, vals in totals.items()}
