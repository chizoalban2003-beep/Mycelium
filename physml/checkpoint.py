"""Stage 50 — AgentCheckpoint: full agent state serialization / resume.

Provides one-call save and load for :class:`~physml.mycelium_agent.MyceliumAgent`
(and any object with a compatible attribute layout) using :mod:`joblib`
(already a transitive dependency via scikit-learn) with optional gzip
compression.

The checkpoint stores a manifest dict so you can inspect what was saved and
validate version compatibility before loading.

Usage
-----
::

    from physml.checkpoint import AgentCheckpoint

    # Save
    ckpt = AgentCheckpoint.save(agent, "mycelium.ckpt")

    # Load — returns a ready-to-use MyceliumAgent
    agent2 = AgentCheckpoint.load("mycelium.ckpt")

    # Inspect without loading the full agent
    meta = AgentCheckpoint.inspect("mycelium.ckpt")
    print(meta["stage"], meta["n_observations"])
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import joblib

_CHECKPOINT_VERSION = "1.0"
# All checkpoint format versions this release can read.  Add new entries here
# when the format changes to preserve backward compatibility.
_SUPPORTED_VERSIONS = {"1.0"}


class AgentCheckpoint:
    """Utility class for saving and loading agent state.

    All methods are class-methods / static-methods — no instantiation needed.

    The checkpoint format is a dict stored by joblib::

        {
            "version":        str,          # checkpoint format version
            "timestamp":      float,        # unix timestamp
            "stage":          str,          # last known stage tag
            "n_observations": int,
            "agent":          MyceliumAgent # the full agent object
        }
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def save(
        cls,
        agent: Any,
        path: str | Path,
        compress: int = 3,
    ) -> Path:
        """Persist *agent* to *path*.

        Parameters
        ----------
        agent : MyceliumAgent or compatible
        path : str | Path
            File path.  ``.ckpt`` or ``.pkl`` extension recommended.
        compress : int, default 3
            joblib compression level (0 = none, 9 = max).

        Returns
        -------
        Path
            Resolved absolute path of the written checkpoint.
        """
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        n_obs = cls._get_n_observations(agent)
        stage = cls._get_stage_tag(agent)

        manifest = {
            "version": _CHECKPOINT_VERSION,
            "timestamp": time.time(),
            "stage": stage,
            "n_observations": n_obs,
            "agent": agent,
        }
        joblib.dump(manifest, path, compress=compress)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Any:
        """Load and return the agent stored in *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the checkpoint version is incompatible.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        manifest = joblib.load(path)
        cls._validate_manifest(manifest)
        return manifest["agent"]

    @classmethod
    def inspect(cls, path: str | Path) -> dict[str, Any]:
        """Return the manifest metadata without loading the full agent.

        The ``"agent"`` key is replaced with a placeholder string so that the
        full object graph is not deserialised.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        manifest = joblib.load(path)
        meta = {k: v for k, v in manifest.items() if k != "agent"}
        meta["agent"] = "<not loaded>"
        meta["file_size_bytes"] = path.stat().st_size
        return meta

    @classmethod
    def save_bytes(cls, agent: Any, compress: int = 3) -> bytes:
        """Serialize *agent* to an in-memory bytes buffer (no file I/O)."""
        n_obs = cls._get_n_observations(agent)
        stage = cls._get_stage_tag(agent)
        manifest = {
            "version": _CHECKPOINT_VERSION,
            "timestamp": time.time(),
            "stage": stage,
            "n_observations": n_obs,
            "agent": agent,
        }
        buf = io.BytesIO()
        joblib.dump(manifest, buf, compress=compress)
        return buf.getvalue()

    @classmethod
    def load_bytes(cls, data: bytes) -> Any:
        """Load an agent from a bytes buffer."""
        buf = io.BytesIO(data)
        manifest = joblib.load(buf)
        cls._validate_manifest(manifest)
        return manifest["agent"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_n_observations(agent: Any) -> int:
        try:
            return int(agent._agent.n_observations_)
        except AttributeError:
            pass
        try:
            return int(agent.n_observations_)
        except AttributeError:
            return 0

    @staticmethod
    def _get_stage_tag(agent: Any) -> str:
        cls_name = type(agent).__name__
        return f"{cls_name}@stage46+"

    @staticmethod
    def _validate_manifest(manifest: Any) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("Checkpoint is not a valid manifest dict.")
        if manifest.get("version") not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported checkpoint version {manifest.get('version')!r}. "
                f"Supported versions: {sorted(_SUPPORTED_VERSIONS)}."
            )
        if "agent" not in manifest:
            raise ValueError("Checkpoint has no 'agent' key.")
