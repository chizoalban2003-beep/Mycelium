"""Stage 91 — AgentMemory: episodic and semantic memory store.

Provides the agent with a structured memory system that separates
short-term *episodic* events (timestamped observations/actions) from
long-term *semantic* facts (key-value knowledge).

Classes
-------
MemoryEntry
    A single episodic memory record.
AgentMemory
    Manages episodic and semantic memory with retrieval helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """One episodic memory.

    Attributes
    ----------
    timestamp : float
        Unix time when this entry was recorded.
    observation : Any
        The raw observation or state snapshot.
    action : Any
        The action taken (or *None* for passive observations).
    reward : float
        Reward received (defaults to 0.0).
    tag : str
        Optional label for retrieval filtering.
    """

    timestamp: float
    observation: Any
    action: Any = None
    reward: float = 0.0
    tag: str = ""


class AgentMemory:
    """Episodic + semantic memory for an autonomous agent.

    Parameters
    ----------
    max_episodic : int
        Maximum number of episodic entries to retain (FIFO eviction).
        Use ``-1`` for unlimited.

    Attributes
    ----------
    episodic : list[MemoryEntry]
        Recent episodic memories.
    semantic : dict[str, Any]
        Long-term key-value semantic facts.
    """

    def __init__(self, max_episodic: int = 1000) -> None:
        self.max_episodic = max_episodic
        self.episodic: List[MemoryEntry] = []
        self.semantic: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Episodic
    # ------------------------------------------------------------------
    def record(
        self,
        observation: Any,
        action: Any = None,
        reward: float = 0.0,
        tag: str = "",
    ) -> MemoryEntry:
        """Record an episodic memory.

        Parameters
        ----------
        observation : Any
            Current observation.
        action : Any, optional
            Action taken.
        reward : float, optional
            Reward received.
        tag : str, optional
            Label for later retrieval.

        Returns
        -------
        MemoryEntry
            The newly created entry.
        """
        entry = MemoryEntry(
            timestamp=time.time(),
            observation=observation,
            action=action,
            reward=reward,
            tag=tag,
        )
        self.episodic.append(entry)
        if self.max_episodic > 0 and len(self.episodic) > self.max_episodic:
            self.episodic = self.episodic[-self.max_episodic :]
        return entry

    def recall(self, tag: str = "", n: int = 10) -> List[MemoryEntry]:
        """Return the *n* most-recent episodic entries matching *tag*.

        Parameters
        ----------
        tag : str
            If non-empty, only entries with this tag are returned.
        n : int
            Maximum number of entries to return.

        Returns
        -------
        list[MemoryEntry]
        """
        entries = self.episodic if not tag else [e for e in self.episodic if e.tag == tag]
        return entries[-n:]

    def total_reward(self) -> float:
        """Sum of all episodic rewards."""
        return sum(e.reward for e in self.episodic)

    # ------------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------------
    def remember(self, key: str, value: Any) -> None:
        """Store a semantic fact."""
        self.semantic[key] = value

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a semantic fact."""
        return self.semantic.get(key, default)

    def forget(self, key: str) -> bool:
        """Delete a semantic fact.  Returns *True* if the key existed."""
        if key in self.semantic:
            del self.semantic[key]
            return True
        return False

    # ------------------------------------------------------------------
    def clear_episodic(self) -> None:
        """Discard all episodic memories."""
        self.episodic.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AgentMemory(episodic={len(self.episodic)}, "
            f"semantic_keys={len(self.semantic)})"
        )
