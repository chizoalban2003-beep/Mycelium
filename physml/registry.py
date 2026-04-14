"""Stage 29 — Lightweight model registry for PhysML / myco.

Provides :class:`ModelRegistry` — a simple JSONL-backed run tracker that logs
metadata for every ``myco.fit()`` call, enabling reproducibility and rollback
without a heavy MLflow dependency.

Each run record contains:

* ``run_id``        — unique UUID
* ``timestamp``     — ISO-8601 UTC timestamp
* ``dataset_hash``  — SHA-256 of ``(X, y)`` concatenated bytes
* ``n_samples``     — number of training rows
* ``n_features``    — number of columns in *X*
* ``temperature``   — calibration temperature (from Stage 13)
* ``oracle_calls``  — number of reward() calls at logging time
* ``final_accuracy``— last test accuracy from the predictor (when available)
* ``tags``          — arbitrary user-supplied metadata dict
* ``agent_path``    — path to the saved agent pickle (if ``save_agent=True``)

Usage
-----
::

    from physml import myco
    from physml.registry import ModelRegistry

    reg = ModelRegistry("experiments.jsonl")

    agent = myco()
    agent.fit(X_train, y_train)

    run_id = reg.log(agent, X_train, y_train, tags={"dataset": "iris"})
    print(run_id)

    df = reg.list_runs()
    print(df)

    # Restore agent from a previous run
    loaded = reg.load_agent(run_id)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Lightweight JSONL-backed model registry.

    Parameters
    ----------
    path : str or Path
        Path to the ``.jsonl`` file used to store run records.  Created on
        first :meth:`log` call if it does not exist.
    """

    def __init__(self, path: str | Path = "physml_runs.jsonl") -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        agent: Any,
        X: Any,
        y: Any,
        *,
        tags: dict[str, Any] | None = None,
        save_agent: bool = True,
    ) -> str:
        """Record a training run and optionally save the agent to disk.

        Parameters
        ----------
        agent : MyceliumAgent or PhysicsAgent
            A *fitted* agent.
        X : array-like of shape (n_samples, n_features)
            Training features used in the fit.
        y : array-like of shape (n_samples,)
            Training targets used in the fit.
        tags : dict, optional
            Arbitrary key/value metadata (e.g. ``{"dataset": "iris", "seed": 42}``).
        save_agent : bool, default True
            If True, save the agent to a ``.pkl`` file alongside the registry
            (using :meth:`~physml.mycelium_agent.MyceliumAgent.save` when
            available, or joblib otherwise).  The path is recorded in the run.

        Returns
        -------
        str
            A unique ``run_id`` (UUID4 hex).
        """
        import numpy as np

        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y)

        run_id = uuid.uuid4().hex
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        dataset_hash = _hash_data(X_arr, y_arr)
        n_samples, n_features = X_arr.shape

        # Extract metadata from the agent
        temperature = float(getattr(agent, "temperature_", 1.0))
        oracle_calls = 0
        final_accuracy = None

        report = {}
        try:
            report = agent.report()
        except Exception:
            pass

        agent_report = report.get("agent", report)
        oracle_calls = int(agent_report.get("n_rewards", 0))

        # Try to extract final accuracy from the predictor's last result
        predictor = getattr(agent, "_predictor", getattr(agent, "predictor", None))
        if predictor is not None:
            result = getattr(predictor, "result_", None)
            if result is not None:
                final_accuracy = getattr(result, "test_accuracy", None)
                if final_accuracy is None:
                    final_accuracy = getattr(result, "r2_score", None)

        agent_path = None
        if save_agent:
            pkl_dir = self.path.parent / "agents"
            pkl_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = pkl_dir / f"{run_id}.pkl"
            try:
                save_fn = getattr(agent, "save", None)
                if callable(save_fn):
                    save_fn(pkl_path)
                else:
                    import joblib
                    joblib.dump(agent, str(pkl_path))
                agent_path = str(pkl_path)
            except Exception:
                agent_path = None

        record: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "dataset_hash": dataset_hash,
            "n_samples": n_samples,
            "n_features": n_features,
            "temperature": temperature,
            "oracle_calls": oracle_calls,
            "final_accuracy": final_accuracy,
            "tags": tags or {},
            "agent_path": agent_path,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return run_id

    def list_runs(self) -> Any:
        """Return all runs as a pandas DataFrame (or list of dicts).

        Returns
        -------
        pandas.DataFrame if pandas is available, else list[dict]
        """
        records = self._load_records()
        try:
            import pandas as pd
            return pd.DataFrame(records)
        except ImportError:
            return records

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return the record for a specific run.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        dict

        Raises
        ------
        KeyError
            If no run with the given ``run_id`` exists.
        """
        for record in self._load_records():
            if record.get("run_id") == run_id:
                return record
        raise KeyError(f"No run found with run_id={run_id!r}")

    def load_agent(self, run_id: str) -> Any:
        """Load the saved agent for a previous run.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        MyceliumAgent (or whatever was saved)

        Raises
        ------
        KeyError
            If no run with the given ``run_id`` exists.
        ValueError
            If the run has no saved agent path.
        """
        import joblib

        record = self.get_run(run_id)
        agent_path = record.get("agent_path")
        if not agent_path:
            raise ValueError(
                f"Run {run_id!r} has no saved agent.  "
                "Re-log with save_agent=True."
            )
        return joblib.load(agent_path)

    def delete_run(self, run_id: str) -> None:
        """Remove a run record (and its saved agent file if present).

        Parameters
        ----------
        run_id : str
        """
        records = self._load_records()
        kept = []
        deleted_path = None
        for r in records:
            if r.get("run_id") == run_id:
                deleted_path = r.get("agent_path")
            else:
                kept.append(r)
        if deleted_path:
            try:
                Path(deleted_path).unlink(missing_ok=True)
            except Exception:
                pass
        with self.path.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def __repr__(self) -> str:
        n = len(self._load_records())
        return f"ModelRegistry(path={str(self.path)!r}, n_runs={n})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_data(*arrays: Any) -> str:
    """Return a hex SHA-256 digest of the concatenated byte representations."""
    import numpy as np

    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.asarray(arr).tobytes())
    return h.hexdigest()[:16]  # 16 hex chars is plenty for a run identifier
