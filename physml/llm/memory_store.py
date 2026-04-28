"""physml.llm.memory_store — Persistent key-value user memory.

:class:`UserMemory` stores named facts about the user (name, preferences,
key info) in a JSON file (~/.mycelium/user_memory.json) and can inject them
as context into system prompts.

Usage::

    from physml.llm.memory_store import UserMemory

    mem = UserMemory()
    mem.remember("name", "Alex")
    mem.remember("preferred_language", "Python")

    print(mem.recall("name"))            # "Alex"
    print(mem.inject_into_prompt())      # formatted string for system prompts
    print(mem.summary())                 # {"name": "Alex", ...}
    mem.forget("name")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


class UserMemory:
    """Persistent user facts: name, preferences, key info.

    Backed by a JSON file at *path* (default: ``~/.mycelium/user_memory.json``).
    All reads and writes are synchronous and atomic (write-to-temp + rename is
    avoided for simplicity; the file is small).

    Parameters
    ----------
    path : str or Path, optional
        Path to the JSON storage file.  Defaults to
        ``~/.mycelium/user_memory.json``.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = "~/.mycelium/user_memory.json"
        self._path = Path(path).expanduser()
        self._data: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        """Store a key-value fact about the user.

        Parameters
        ----------
        key : str
            Fact name (e.g. ``"name"``, ``"preferred_language"``).
        value : str
            The value to store.
        """
        self._data[str(key)] = str(value)
        self._save()

    def recall(self, key: str) -> Optional[str]:
        """Return the stored value for *key*, or ``None`` if not found.

        Parameters
        ----------
        key : str
            Fact name to look up.

        Returns
        -------
        str or None
        """
        return self._data.get(str(key))

    def forget(self, key: str) -> None:
        """Remove *key* from memory (no-op if not present).

        Parameters
        ----------
        key : str
            Fact name to remove.
        """
        self._data.pop(str(key), None)
        self._save()

    def inject_into_prompt(self) -> str:
        """Return a formatted string for injection into system prompts.

        Returns an empty string when no facts are stored.

        Returns
        -------
        str
            Multi-line string of ``Key: Value`` pairs, or empty string.
        """
        if not self._data:
            return ""
        lines = ["User facts:"]
        for k, v in self._data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def summary(self) -> Dict[str, str]:
        """Return a shallow copy of all stored facts.

        Returns
        -------
        dict
        """
        return dict(self._data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load facts from disk (no-op if file does not exist)."""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._data = {str(k): str(v) for k, v in data.items()}
            except Exception:
                self._data = {}

    def _save(self) -> None:
        """Persist facts to disk, creating parent directories as needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"UserMemory(path={str(self._path)!r}, facts={len(self._data)})"
