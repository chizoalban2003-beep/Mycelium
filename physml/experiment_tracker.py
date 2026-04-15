"""Stage 78 — ExperimentTracker: lightweight ML experiment tracking.

Tracks training runs with parameters, metrics, and artefact paths without
requiring external services (no MLflow dependency).  All data is stored
in-memory and can be persisted to a local JSON file.

Classes
-------
Run
    A single experiment run with params, metrics, and artefacts.
ExperimentTracker
    Manages multiple runs; supports comparison and JSON export.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Run:
    """A single experiment run.

    Attributes
    ----------
    run_id : str
        Unique identifier (UUID4 hex prefix).
    name : str
        Human-readable run name.
    params : dict
        Hyperparameters logged for this run.
    metrics : dict
        Scalar metrics (e.g. ``{"accuracy": 0.92, "loss": 0.18}``).
    artefacts : list[str]
        File paths of artefacts (models, plots, data dumps) attached to this run.
    tags : dict
        Arbitrary string key/value metadata.
    start_time : float
        Unix timestamp when the run was created.
    end_time : float or None
        Unix timestamp when :meth:`end` was called, or None if still active.
    status : str
        ``"running"`` / ``"finished"`` / ``"failed"``.
    """

    run_id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artefacts: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "running"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self.params[key] = value

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple hyperparameters at once."""
        self.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        """Log a scalar metric."""
        self.metrics[key] = float(value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log multiple scalar metrics at once."""
        for k, v in metrics.items():
            self.metrics[k] = float(v)

    def log_artefact(self, path: str) -> None:
        """Register a file path as an artefact of this run."""
        self.artefacts.append(str(path))

    def set_tag(self, key: str, value: str) -> None:
        """Attach a string tag."""
        self.tags[key] = str(value)

    def end(self, status: str = "finished") -> None:
        """Mark the run as finished."""
        self.end_time = time.time()
        self.status = status

    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration in seconds, or None if still running."""
        if self.end_time is None:
            return None
        return round(self.end_time - self.start_time, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "params": self.params,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "artefacts": self.artefacts,
            "tags": self.tags,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
        }


class ExperimentTracker:
    """Lightweight ML experiment tracker.

    Manages a collection of :class:`Run` objects and provides comparison,
    filtering, and JSON export utilities.

    Parameters
    ----------
    experiment_name : str, default "default"
        Human-readable label for this experiment.
    persist_path : str or Path or None
        If provided, auto-save to this JSON file after every run is ended.

    Example
    -------
    >>> from physml.experiment_tracker import ExperimentTracker
    >>> tracker = ExperimentTracker("my_experiment")
    >>> run = tracker.start_run("baseline")
    >>> run.log_params({"lr": 0.01, "epochs": 50})
    >>> run.log_metric("accuracy", 0.88)
    >>> tracker.end_run()
    >>> tracker.best_run("accuracy").name
    'baseline'
    """

    def __init__(
        self,
        experiment_name: str = "default",
        *,
        persist_path: "str | Path | None" = None,
    ) -> None:
        self.experiment_name = str(experiment_name)
        self.persist_path = Path(persist_path) if persist_path is not None else None

        self._runs: list[Run] = []
        self._active_run: Run | None = None

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, name: str = "") -> Run:
        """Create and activate a new run.

        If a run is already active it is ended automatically (status=
        ``"finished"``) before the new one starts.
        """
        if self._active_run is not None:
            self._active_run.end("finished")

        run_id = uuid.uuid4().hex[:8]
        if not name:
            name = f"run_{len(self._runs)}"
        run = Run(run_id=run_id, name=name)
        self._runs.append(run)
        self._active_run = run
        return run

    def end_run(self, status: str = "finished") -> Run | None:
        """End the currently active run and persist if a path is configured."""
        if self._active_run is None:
            return None
        self._active_run.end(status)
        run = self._active_run
        self._active_run = None
        if self.persist_path is not None:
            self.save(self.persist_path)
        return run

    @property
    def active_run(self) -> Run | None:
        """The currently active run, or None."""
        return self._active_run

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def runs(self) -> list[Run]:
        """All runs (including active)."""
        return list(self._runs)

    def get_run(self, run_id: str) -> Run | None:
        """Fetch a run by its ID."""
        for r in self._runs:
            if r.run_id == run_id:
                return r
        return None

    def best_run(self, metric: str, *, higher_is_better: bool = True) -> Run | None:
        """Return the run with the best value for *metric*.

        Parameters
        ----------
        metric : str
            Key in ``run.metrics``.
        higher_is_better : bool, default True
        """
        candidates = [r for r in self._runs if metric in r.metrics]
        if not candidates:
            return None
        return (max if higher_is_better else min)(
            candidates, key=lambda r: r.metrics[metric]
        )

    def compare(self, metric: str) -> list[dict[str, Any]]:
        """Return a leaderboard sorted by *metric* (descending).

        Parameters
        ----------
        metric : str

        Returns
        -------
        list[dict]
            Each entry has ``run_id``, ``name``, and the metric value.
        """
        candidates = [
            {"run_id": r.run_id, "name": r.name, metric: r.metrics[metric]}
            for r in self._runs
            if metric in r.metrics
        ]
        return sorted(candidates, key=lambda d: d[metric], reverse=True)

    def filter_by_tag(self, key: str, value: str) -> list[Run]:
        """Return all runs where ``tags[key] == value``."""
        return [r for r in self._runs if r.tags.get(key) == value]

    def summary(self) -> dict[str, Any]:
        """High-level tracker summary."""
        return {
            "experiment_name": self.experiment_name,
            "n_runs": len(self._runs),
            "n_finished": sum(r.status == "finished" for r in self._runs),
            "n_running": sum(r.status == "running" for r in self._runs),
            "n_failed": sum(r.status == "failed" for r in self._runs),
            "active_run": self._active_run.run_id if self._active_run else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: "str | Path") -> None:
        """Persist all runs to *path* as JSON."""
        data = {
            "experiment_name": self.experiment_name,
            "runs": [r.as_dict() for r in self._runs],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: "str | Path") -> "ExperimentTracker":
        """Restore an :class:`ExperimentTracker` from a JSON file."""
        raw = json.loads(Path(path).read_text())
        tracker = cls(raw.get("experiment_name", "default"))
        for rd in raw.get("runs", []):
            run = Run(
                run_id=rd["run_id"],
                name=rd["name"],
                params=rd.get("params", {}),
                metrics=rd.get("metrics", {}),
                artefacts=rd.get("artefacts", []),
                tags=rd.get("tags", {}),
                start_time=rd.get("start_time", 0.0),
                end_time=rd.get("end_time"),
                status=rd.get("status", "finished"),
            )
            tracker._runs.append(run)
        return tracker
