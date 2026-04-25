"""Stage 102 — TraceRecorder: structured execution-trace recording.

Captures a time-ordered log of agent events (observations, actions,
rewards, errors) during an autonomous run.  Supports replay, filtering,
export to plain dicts, and summary statistics.

Classes
-------
ExecutionTrace
    A single recorded event in the agent's execution history.
TraceRecorder
    Accumulates :class:`ExecutionTrace` entries and exposes query helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ExecutionTrace
# ---------------------------------------------------------------------------


@dataclass
class ExecutionTrace:
    """One recorded event in the execution log.

    Attributes
    ----------
    event_id : int
        Auto-assigned sequential identifier.
    event_type : str
        Category such as ``"observe"``, ``"action"``, ``"reward"``,
        ``"error"``, or any application-defined string.
    timestamp : float
        Unix timestamp when the event was recorded.
    payload : dict
        Arbitrary event data.
    agent_id : str
        Identifier of the agent that emitted the event.
    """

    event_id: int
    event_type: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "payload": dict(self.payload),
        }


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Records structured execution traces from one or more agents.

    Parameters
    ----------
    max_size : int or None, default None
        When set, the recorder retains only the *most recent* ``max_size``
        events, discarding older ones automatically.
    agent_id : str, default "default"
        Default agent identifier appended to events when none is provided.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        agent_id: str = "default",
    ) -> None:
        self.max_size = max_size
        self.agent_id = agent_id
        self._traces: List[ExecutionTrace] = []
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> ExecutionTrace:
        """Append one event to the trace log.

        Parameters
        ----------
        event_type : str
            Category of the event.
        payload : dict, optional
            Arbitrary data associated with the event.
        agent_id : str, optional
            Overrides the recorder's default *agent_id*.
        timestamp : float, optional
            Override the auto-generated Unix timestamp.

        Returns
        -------
        ExecutionTrace
            The newly created trace entry.
        """
        trace = ExecutionTrace(
            event_id=self._counter,
            event_type=event_type,
            timestamp=timestamp if timestamp is not None else time.time(),
            payload=dict(payload or {}),
            agent_id=agent_id or self.agent_id,
        )
        self._traces.append(trace)
        self._counter += 1
        if self.max_size is not None and len(self._traces) > self.max_size:
            self._traces = self._traces[-self.max_size :]
        return trace

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def filter(
        self,
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[ExecutionTrace]:
        """Return traces matching the given filters.

        Parameters
        ----------
        event_type : str or None
            Only return traces of this event type.
        agent_id : str or None
            Only return traces from this agent.
        """
        result = self._traces
        if event_type is not None:
            result = [t for t in result if t.event_type == event_type]
        if agent_id is not None:
            result = [t for t in result if t.agent_id == agent_id]
        return result

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics over all recorded traces.

        Returns
        -------
        dict with keys:
            ``total`` — total event count,
            ``by_type`` — mapping of event_type → count,
            ``by_agent`` — mapping of agent_id → count,
            ``duration`` — wall-clock span (last − first timestamp), or 0.
        """
        by_type: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}
        for t in self._traces:
            by_type[t.event_type] = by_type.get(t.event_type, 0) + 1
            by_agent[t.agent_id] = by_agent.get(t.agent_id, 0) + 1
        duration = 0.0
        if len(self._traces) >= 2:
            duration = self._traces[-1].timestamp - self._traces[0].timestamp
        return {
            "total": len(self._traces),
            "by_type": by_type,
            "by_agent": by_agent,
            "duration": duration,
        }

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Export all traces as a list of plain dicts."""
        return [t.to_dict() for t in self._traces]

    def clear(self) -> None:
        """Remove all recorded events."""
        self._traces = []
        self._counter = 0

    def __len__(self) -> int:
        return len(self._traces)

    def __iter__(self):  # type: ignore[override]
        return iter(self._traces)
