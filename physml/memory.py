"""Stage 33 — EpisodicMemory: numpy kNN episode store for memory-augmented inference.

Stores ``(context, action, outcome)`` triples and retrieves the *k* most
similar episodes by cosine similarity.  The retrieved episodes can be used to
augment input feature vectors before prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class EpisodicMemory:
    """Fixed-capacity episodic memory backed by numpy arrays.

    Episodes are stored as ``(context_vector, action_string, outcome_float)``
    triples.  When capacity is exceeded the oldest episode is evicted (FIFO).

    Parameters
    ----------
    capacity : int, default 1000
        Maximum number of episodes stored simultaneously.
    n_neighbors : int, default 3
        Default *k* for :meth:`retrieve` and :meth:`augment_features`.
    feature_dim : int or None, default None
        Expected dimensionality of context vectors (informational only).
    """

    def __init__(
        self,
        capacity: int = 1000,
        n_neighbors: int = 3,
        feature_dim: int | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.n_neighbors = int(n_neighbors)
        self.feature_dim = feature_dim

        self._contexts: list[np.ndarray] = []
        self._actions: list[str] = []
        self._outcomes: list[float] = []
        self._action_vocab: dict[str, int] = {}  # action → index for encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, context: np.ndarray, action: str, outcome: float) -> None:
        """Store an episode triple.

        When the store exceeds :attr:`capacity`, the oldest episode is dropped.

        Parameters
        ----------
        context : np.ndarray
            1-D context vector.
        action : str
            Action taken in this episode.
        outcome : float
            Scalar outcome / reward received.
        """
        ctx = np.atleast_1d(np.asarray(context, dtype=np.float32)).ravel()
        act = str(action)
        out = float(outcome)

        # Evict oldest if at capacity
        if len(self._contexts) >= self.capacity:
            self._contexts.pop(0)
            self._actions.pop(0)
            self._outcomes.pop(0)

        self._contexts.append(ctx)
        self._actions.append(act)
        self._outcomes.append(out)

        if act not in self._action_vocab:
            self._action_vocab[act] = len(self._action_vocab)

    def retrieve(self, query: np.ndarray, k: int | None = None) -> list[dict]:
        """Return the *k* nearest episodes ranked by cosine similarity.

        Parameters
        ----------
        query : np.ndarray
            1-D query vector.
        k : int or None
            Number of neighbours to return.  Defaults to :attr:`n_neighbors`.

        Returns
        -------
        list[dict]
            Each dict has keys: ``context``, ``action``, ``outcome``,
            ``similarity``.
        """
        if not self._contexts:
            return []

        k = self.n_neighbors if k is None else int(k)
        k = min(k, len(self._contexts))

        q = np.atleast_1d(np.asarray(query, dtype=np.float32)).ravel()
        contexts = np.stack(self._contexts)  # (N, d)

        q_norm = float(np.linalg.norm(q)) + 1e-8
        c_norms = np.linalg.norm(contexts, axis=1) + 1e-8
        sims = (contexts @ q) / (c_norms * q_norm)

        top_k = np.argsort(sims)[::-1][:k]
        return [
            {
                "context": self._contexts[i],
                "action": self._actions[i],
                "outcome": self._outcomes[i],
                "similarity": float(sims[i]),
            }
            for i in top_k
        ]

    def augment_features(self, X: np.ndarray) -> np.ndarray:
        """Append episodic memory features to each row of *X*.

        For each row, retrieves :attr:`n_neighbors` nearest episodes and
        appends ``n_neighbors * 2`` extra columns:
        ``(outcome_0, action_enc_0, outcome_1, action_enc_1, ...)``
        where ``action_enc_i`` is the normalised action index.

        If the memory is empty, *X* is returned unchanged.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features + n_neighbors * 2)
        """
        if not self._contexts:
            return X

        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        n_vocab = max(1, len(self._action_vocab))
        extra_rows: list[np.ndarray] = []

        for row in X:
            neighbors = self.retrieve(row, k=self.n_neighbors)
            # Pad to exactly n_neighbors entries
            while len(neighbors) < self.n_neighbors:
                neighbors.append({"outcome": 0.0, "action": "", "similarity": 0.0, "context": row})

            extra: list[float] = []
            for nb in neighbors[: self.n_neighbors]:
                outcome = float(nb["outcome"])
                action_idx = self._action_vocab.get(nb["action"], 0)
                action_enc = action_idx / n_vocab  # normalised to [0, 1)
                extra.extend([outcome, action_enc])

            extra_rows.append(np.array(extra, dtype=np.float32))

        extra_arr = np.stack(extra_rows)  # (n, n_neighbors * 2)
        return np.concatenate([X, extra_arr], axis=1).astype(np.float32)

    def __len__(self) -> int:
        """Return the number of stored episodes."""
        return len(self._contexts)
