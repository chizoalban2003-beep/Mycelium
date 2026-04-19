"""Stage 126 — VectorMemory: semantic memory with local embeddings.

Upgrades the basic KNN episodic memory (Stage 33) with semantic vector
search so the companion can retrieve *contextually relevant* past
conversations rather than just numerically similar feature vectors.

Embedding backends (in priority order)
---------------------------------------
1. **sentence-transformers** — State-of-the-art local embeddings (``all-MiniLM-L6-v2``).
2. **TF-IDF** (sklearn) — Lightweight bag-of-words vectors; no GPU needed.
3. **BM25** — Keyword frequency ranking; minimal dependencies.

All backends are optional. The system selects the best available one
at construction time with full graceful fallback.

Usage
-----
::

    from physml.vector_memory import VectorMemory

    mem = VectorMemory(max_entries=1000)
    mem.add("User asked about sales forecasting for Q3.")
    mem.add("Model trained on retail_data.csv, 1200 rows.")
    mem.add("User prefers concise answers.")

    results = mem.search("How should I forecast revenue?", k=3)
    for r in results:
        print(r.text, r.score)

    mem.save("~/.mycelium/vector_memory.json")
    mem.load("~/.mycelium/vector_memory.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single stored memory.

    Attributes
    ----------
    text : str
        The stored text.
    timestamp : float
        Unix timestamp when the entry was added.
    metadata : dict
        Optional metadata (speaker, intent, etc.).
    """

    text: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieved memory with relevance score.

    Attributes
    ----------
    entry : MemoryEntry
    score : float
        Cosine similarity or relevance score in [0, 1].
    rank : int
        1-indexed rank among results.
    """

    entry: MemoryEntry
    score: float
    rank: int = 1

    @property
    def text(self) -> str:
        return self.entry.text


# ---------------------------------------------------------------------------
# VectorMemory
# ---------------------------------------------------------------------------


class VectorMemory:
    """Semantic memory store with vector search.

    Parameters
    ----------
    max_entries : int, default 2000
        Maximum number of stored entries (FIFO eviction when exceeded).
    backend : str, default "auto"
        ``"sentence_transformers"``, ``"tfidf"``, ``"bm25"``, or ``"auto"``.
    model_name : str
        sentence-transformers model name (used when backend is sentence_transformers).
    persist_path : str or None
        If given, auto-load on init and auto-save on :meth:`add`.
    """

    def __init__(
        self,
        max_entries: int = 2000,
        backend: str = "auto",
        model_name: str = "all-MiniLM-L6-v2",
        persist_path: Optional[str] = None,
    ) -> None:
        self.max_entries = max_entries
        self._preferred_backend = backend
        self._model_name = model_name
        self.persist_path = Path(persist_path).expanduser() if persist_path else None

        self._entries: List[MemoryEntry] = []
        self._embeddings: Any = None  # numpy array of shape (n, dim)
        self._model: Any = None
        self._vectorizer: Any = None
        self._tfidf_matrix: Any = None
        self._active_backend: str = "none"

        self._init_backend()

        if self.persist_path and self.persist_path.exists():
            try:
                self.load(str(self.persist_path))
            except Exception as e:
                _logger.warning("VectorMemory: could not load from %s: %s", persist_path, e)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        pref = self._preferred_backend
        if pref in ("sentence_transformers", "auto"):
            if self._try_init_st():
                return
        if pref in ("tfidf", "auto"):
            if self._try_init_tfidf():
                return
        if pref in ("bm25", "auto"):
            if self._try_init_bm25():
                return
        self._active_backend = "linear_scan"
        _logger.info("VectorMemory: using linear text scan (no embedding backend)")

    def _try_init_st(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name)
            self._active_backend = "sentence_transformers"
            _logger.info("VectorMemory: using sentence-transformers (%s)", self._model_name)
            return True
        except ImportError:
            return False

    def _try_init_tfidf(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            self._vectorizer = TfidfVectorizer(max_features=4096, sublinear_tf=True)
            self._active_backend = "tfidf"
            _logger.info("VectorMemory: using TF-IDF backend")
            return True
        except ImportError:
            return False

    def _try_init_bm25(self) -> bool:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore  # noqa: F401

            self._active_backend = "bm25"
            _logger.info("VectorMemory: using BM25 backend")
            return True
        except ImportError:
            return False

    @property
    def active_backend(self) -> str:
        """The embedding backend currently in use."""
        return self._active_backend

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Add a text entry to memory.

        Parameters
        ----------
        text : str
        metadata : dict, optional

        Returns
        -------
        MemoryEntry
        """
        entry = MemoryEntry(text=text, metadata=metadata or {})
        self._entries.append(entry)

        # Evict oldest if over capacity
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
            self._embeddings = None  # invalidate cache

        # Incrementally update embeddings for ST backend
        if self._active_backend == "sentence_transformers" and self._model is not None:
            import numpy as np

            new_emb = self._model.encode([text], normalize_embeddings=True)
            if self._embeddings is None or len(self._embeddings) != len(self._entries) - 1:
                self._embeddings = new_emb
            else:
                self._embeddings = np.vstack([self._embeddings, new_emb])

        # Invalidate TF-IDF matrix (rebuilt on next search)
        elif self._active_backend == "tfidf":
            self._tfidf_matrix = None

        if self.persist_path:
            try:
                self.save(str(self.persist_path))
            except Exception:
                pass

        return entry

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Retrieve the *k* most semantically relevant memories.

        Parameters
        ----------
        query : str
        k : int, default 5

        Returns
        -------
        list of SearchResult
        """
        if not self._entries:
            return []

        k = min(k, len(self._entries))

        if self._active_backend == "sentence_transformers":
            return self._search_st(query, k)
        elif self._active_backend == "tfidf":
            return self._search_tfidf(query, k)
        elif self._active_backend == "bm25":
            return self._search_bm25(query, k)
        else:
            return self._search_linear(query, k)

    def clear(self) -> None:
        """Remove all stored entries."""
        self._entries.clear()
        self._embeddings = None
        self._tfidf_matrix = None

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Backend-specific search
    # ------------------------------------------------------------------

    def _search_st(self, query: str, k: int) -> List[SearchResult]:
        import numpy as np

        if self._embeddings is None or len(self._embeddings) != len(self._entries):
            texts = [e.text for e in self._entries]
            self._embeddings = self._model.encode(texts, normalize_embeddings=True)

        q_emb = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q_emb.T).flatten()
        top_k = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(entry=self._entries[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(top_k)
        ]

    def _search_tfidf(self, query: str, k: int) -> List[SearchResult]:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        texts = [e.text for e in self._entries]
        if self._tfidf_matrix is None or self._tfidf_matrix.shape[0] != len(texts):
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_k = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(entry=self._entries[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(top_k)
        ]

    def _search_bm25(self, query: str, k: int) -> List[SearchResult]:
        import numpy as np
        from rank_bm25 import BM25Okapi  # type: ignore

        tokenized_corpus = [e.text.lower().split() for e in self._entries]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        top_k = np.argsort(scores)[::-1][:k]
        max_score = max(scores) if scores.max() > 0 else 1.0
        return [
            SearchResult(
                entry=self._entries[i],
                score=float(scores[i]) / max_score,
                rank=rank + 1,
            )
            for rank, i in enumerate(top_k)
        ]

    def _search_linear(self, query: str, k: int) -> List[SearchResult]:
        """Fallback: score by word overlap (Jaccard-like)."""
        query_words = set(query.lower().split())
        scored = []
        for entry in self._entries:
            entry_words = set(entry.text.lower().split())
            if not query_words and not entry_words:
                score = 0.0
            else:
                intersection = query_words & entry_words
                union = query_words | entry_words
                score = len(intersection) / len(union) if union else 0.0
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(entry=e, score=s, rank=i + 1)
            for i, (s, e) in enumerate(scored[:k])
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all entries to a JSON file.

        Parameters
        ----------
        path : str
        """
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "text": e.text,
                "timestamp": e.timestamp,
                "metadata": e.metadata,
            }
            for e in self._entries
        ]
        p.write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> int:
        """Load entries from a JSON file.

        Parameters
        ----------
        path : str

        Returns
        -------
        int
            Number of entries loaded.
        """
        p = Path(path).expanduser()
        if not p.exists():
            return 0
        data = json.loads(p.read_text())
        self._entries = [
            MemoryEntry(
                text=d["text"],
                timestamp=d.get("timestamp", time.time()),
                metadata=d.get("metadata", {}),
            )
            for d in data
        ]
        self._embeddings = None
        self._tfidf_matrix = None
        _logger.info("VectorMemory: loaded %d entries from %s", len(self._entries), path)
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"VectorMemory("
            f"entries={len(self._entries)}, "
            f"backend={self._active_backend!r})"
        )
