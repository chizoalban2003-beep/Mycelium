"""physml.conversation_store — Persistent conversation history with search.

:class:`ConversationStore` persists multi-turn conversation history to disk as
JSON and supports:

* Appending new user/assistant turns.
* Loading history from disk on startup (survives restarts).
* Searching past turns by keyword (substring or TF-IDF similarity).
* Trimming to a maximum number of turns.
* Exporting history as a list of dicts suitable for the Anthropic messages API.

This fills the gap of a "memory layer" — the companion can recall what was
said in previous sessions and pass relevant context to the LLM.

Usage::

    from physml.conversation_store import ConversationStore

    store = ConversationStore(path="~/.mycelium/conversation.json")

    store.add("user", "What is my sales forecast?")
    store.add("assistant", "Based on your data, sales will likely grow 12% next month.")

    # Persist to disk
    store.save()

    # Reload in a new session
    store2 = ConversationStore(path="~/.mycelium/conversation.json")
    print(len(store2))  # number of turns

    # Search relevant history
    results = store2.search("sales forecast", k=3)
    for r in results:
        print(r["role"], r["content"][:60])

    # Export for Anthropic API
    messages = store2.to_messages(max_turns=20)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------


class ConversationStore:
    """Persistent conversation history with keyword search.

    Parameters
    ----------
    path : str
        Path to the JSON file where history is stored.  Created automatically.
    max_turns : int
        Maximum number of turns to keep in memory and on disk.  Oldest turns
        are evicted when the limit is exceeded.
    """

    def __init__(
        self,
        path: str = "~/.mycelium/conversation.json",
        max_turns: int = 1000,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_turns = max_turns
        self._turns: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Core mutations
    # ------------------------------------------------------------------

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a new turn.

        Parameters
        ----------
        role : str
            ``"user"`` or ``"assistant"``.
        content : str
            The message text.
        metadata : dict, optional
            Extra metadata (intent, timestamp, model, etc.).

        Returns
        -------
        dict
            The newly appended turn record.
        """
        turn: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            turn["metadata"] = metadata

        self._turns.append(turn)

        # Evict oldest turns if over limit
        if len(self._turns) > self.max_turns:
            excess = len(self._turns) - self.max_turns
            self._turns = self._turns[excess:]

        return turn

    def clear(self) -> None:
        """Remove all turns from memory (does not delete the file)."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write history to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {"version": 1, "turns": self._turns},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            _logger.debug("ConversationStore: saved %d turns to %s", len(self._turns), self.path)
        except Exception as exc:
            _logger.warning("ConversationStore.save failed: %s", exc)

    def _load(self) -> None:
        """Load history from disk if the file exists."""
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            turns = data.get("turns", data) if isinstance(data, dict) else data
            self._turns = list(turns)[-self.max_turns:]
            _logger.debug(
                "ConversationStore: loaded %d turns from %s", len(self._turns), self.path
            )
        except Exception as exc:
            _logger.warning("ConversationStore._load failed: %s", exc)
            self._turns = []

    def reload(self) -> None:
        """Re-read history from disk (discards in-memory unsaved changes)."""
        self._load()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for turns relevant to *query*.

        First tries TF-IDF similarity; falls back to simple substring /
        keyword matching if scikit-learn is unavailable.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Maximum number of results.
        role : str or None
            Filter to ``"user"`` or ``"assistant"`` turns only.

        Returns
        -------
        list of turn dicts, ordered by relevance (highest first).
        """
        turns = self._turns
        if role is not None:
            turns = [t for t in turns if t.get("role") == role]
        if not turns:
            return []

        query_lower = query.lower()

        try:
            return self._search_tfidf(query_lower, turns, k)
        except Exception:
            return self._search_keyword(query_lower, turns, k)

    def _search_tfidf(
        self, query: str, turns: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        corpus = [t.get("content", "") for t in turns]
        if len(corpus) == 0:
            return []

        vec = TfidfVectorizer(max_features=500, stop_words="english")
        try:
            mat = vec.fit_transform(corpus)
            q_vec = vec.transform([query])
            sims = cosine_similarity(q_vec, mat)[0]
            top_idx = np.argsort(sims)[::-1][:k]
            return [turns[i] for i in top_idx if sims[i] > 0]
        except ValueError:
            return self._search_keyword(query, turns, k)

    def _search_keyword(
        self, query: str, turns: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for turn in turns:
            text = turn.get("content", "").lower()
            text_words = set(re.findall(r"\w+", text))
            if not text_words:
                continue
            overlap = len(query_words & text_words) / max(len(query_words), 1)
            if overlap > 0:
                scored.append((overlap, turn))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_messages(
        self,
        max_turns: Optional[int] = None,
        include_metadata: bool = False,
    ) -> List[Dict[str, str]]:
        """Export history as a list of Anthropic-API-compatible message dicts.

        Parameters
        ----------
        max_turns : int or None
            If set, only the last *max_turns* turns are included.
        include_metadata : bool
            Whether to embed metadata into the content (for debugging).

        Returns
        -------
        list of ``{"role": str, "content": str}`` dicts.
        """
        turns = self._turns
        if max_turns is not None:
            turns = turns[-max_turns:]

        msgs = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if include_metadata and turn.get("metadata"):
                content = f"{content} [meta={turn['metadata']}]"
            msgs.append({"role": role, "content": content})
        return msgs

    def to_llm_messages(self, max_turns: int = 20) -> List[Any]:
        """Export history as ``LLMMessage`` objects for :class:`~physml.llm_integration.LLMIntegration`."""
        from physml.llm_integration import LLMMessage

        turns = self._turns[-max_turns:]
        return [LLMMessage(role=t.get("role", "user"), content=t.get("content", "")) for t in turns]

    # ------------------------------------------------------------------
    # Iteration / length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._turns)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._turns[idx]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a statistics summary of the conversation history."""
        n = len(self._turns)
        if n == 0:
            return {"total_turns": 0, "user_turns": 0, "assistant_turns": 0}

        user_turns = sum(1 for t in self._turns if t.get("role") == "user")
        asst_turns = sum(1 for t in self._turns if t.get("role") == "assistant")
        earliest = self._turns[0].get("timestamp", 0)
        latest = self._turns[-1].get("timestamp", 0)

        return {
            "total_turns": n,
            "user_turns": user_turns,
            "assistant_turns": asst_turns,
            "earliest_timestamp": earliest,
            "latest_timestamp": latest,
            "path": str(self.path),
        }

    def __repr__(self) -> str:
        return f"ConversationStore(turns={len(self._turns)}, path={str(self.path)!r})"
