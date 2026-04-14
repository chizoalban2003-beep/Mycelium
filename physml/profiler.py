"""Stage 56 — AgentProfiler: lightweight runtime profiling and instrumentation.

Provides a context manager for timing code blocks, a memory-delta tracker,
and a summary report so developers can identify performance bottlenecks in
the agent pipeline.
"""

from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


@dataclass
class ProfileEntry:
    """One recorded timing + memory measurement."""

    name: str
    elapsed_s: float
    memory_delta_kb: float
    call_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ProfileEntry(name={self.name!r}, "
            f"elapsed_s={self.elapsed_s:.4f}, "
            f"memory_delta_kb={self.memory_delta_kb:.1f}, "
            f"calls={self.call_count})"
        )


class AgentProfiler:
    """Collect timing and memory-usage measurements across agent operations.

    Usage
    -----
    >>> profiler = AgentProfiler()
    >>> with profiler.profile("fit"):
    ...     agent.fit(X, y)
    >>> profiler.report()

    Parameters
    ----------
    track_memory : bool
        Whether to use :mod:`tracemalloc` for memory deltas (adds overhead).
    """

    def __init__(self, track_memory: bool = True) -> None:
        self.track_memory = bool(track_memory)
        self._records: List[ProfileEntry] = []
        self._totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"elapsed_s": 0.0, "memory_delta_kb": 0.0, "calls": 0}
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def profile(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[None, None, None]:
        """Context manager that records timing (and optionally memory) for *name*."""
        if self.track_memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            mem_delta = 0.0
            if self.track_memory:
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_delta = peak / 1024.0  # bytes → KB
            entry = ProfileEntry(
                name=name,
                elapsed_s=elapsed,
                memory_delta_kb=mem_delta,
                metadata=metadata or {},
            )
            self._records.append(entry)
            t = self._totals[name]
            t["elapsed_s"] += elapsed
            t["memory_delta_kb"] += mem_delta
            t["calls"] += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, top_n: int = 10) -> Dict[str, Any]:
        """Return aggregated summary sorted by total elapsed time."""
        rows = []
        for name, totals in self._totals.items():
            rows.append(
                {
                    "name": name,
                    "total_elapsed_s": round(totals["elapsed_s"], 6),
                    "total_memory_kb": round(totals["memory_delta_kb"], 2),
                    "calls": int(totals["calls"]),
                    "avg_elapsed_s": round(
                        totals["elapsed_s"] / totals["calls"], 6
                    ),
                }
            )
        rows.sort(key=lambda r: r["total_elapsed_s"], reverse=True)
        return {
            "top_entries": rows[:top_n],
            "total_calls": sum(r["calls"] for r in rows),
            "total_elapsed_s": round(sum(r["total_elapsed_s"] for r in rows), 6),
        }

    def top_bottlenecks(self, n: int = 3) -> List[str]:
        """Return names of the *n* slowest operations."""
        report = self.report(top_n=n)
        return [e["name"] for e in report["top_entries"]]

    def reset(self) -> None:
        self._records = []
        self._totals = defaultdict(
            lambda: {"elapsed_s": 0.0, "memory_delta_kb": 0.0, "calls": 0}
        )

    # ------------------------------------------------------------------
    # Raw access
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[ProfileEntry]:
        return list(self._records)

    def total_elapsed(self, name: str) -> float:
        return self._totals[name]["elapsed_s"]

    def call_count(self, name: str) -> int:
        return int(self._totals[name]["calls"])

    def __len__(self) -> int:
        return len(self._records)
