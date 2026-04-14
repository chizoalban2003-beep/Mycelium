"""Stage 30 — Featurizer: converts raw inputs to fixed-length float32 vectors.

Supports three input types detected automatically from the first sample:
* **text** (``str``) — character n-gram (n=3,4) hashing trick → TruncatedSVD
* **dict** — JSON-serialised, then same text pipeline
* **numeric** (``list[float]`` or ``np.ndarray`` rows) — StandardScaler + PCA

All outputs are float32 arrays of shape ``(n_samples, output_dim)``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


class Featurizer:
    """Convert heterogeneous raw inputs into fixed-length float32 numpy vectors.

    Parameters
    ----------
    output_dim : int, default 64
        Dimensionality of the output embedding.
    hash_features : int, default 2048
        Size of the intermediate hashing space used for text / dict inputs.
    """

    def __init__(self, output_dim: int = 64, hash_features: int = 2048) -> None:
        self.output_dim = int(output_dim)
        self.hash_features = int(hash_features)

        self._kind: str | None = None  # "text" | "dict" | "numeric"
        self._scaler: StandardScaler | None = None
        self._svd: TruncatedSVD | None = None
        self._pca: PCA | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, samples: list) -> "Featurizer":
        """Infer input type from *samples* and fit the internal transform.

        Parameters
        ----------
        samples : list
            Non-empty list of str, dict, or array-like rows.

        Returns
        -------
        self
        """
        if not samples:
            raise ValueError("samples must be non-empty")

        self._kind = self._infer_kind(samples)

        if self._kind in ("text", "dict"):
            texts = self._to_texts(samples)
            X_hash = self._texts_to_hash_matrix(texts)  # (n, hash_features)
            n_components = max(1, min(self.output_dim, X_hash.shape[0] - 1, X_hash.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=n_components, random_state=42)
            self._svd.fit(X_hash)
        else:
            X = np.atleast_2d(np.array(samples, dtype=np.float32))
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            if X.shape[1] > self.output_dim:
                n_components = max(1, min(self.output_dim, X.shape[0] - 1, X.shape[1]))
                self._pca = PCA(n_components=n_components, random_state=42)
                self._pca.fit(X_scaled)

        self._fitted = True
        return self

    def transform(self, samples: list) -> np.ndarray:
        """Transform *samples* into float32 array of shape ``(n, output_dim)``.

        Parameters
        ----------
        samples : list

        Returns
        -------
        np.ndarray, shape (n_samples, output_dim), dtype float32
        """
        if not self._fitted:
            raise RuntimeError("Featurizer is not fitted yet. Call fit() first.")

        if self._kind in ("text", "dict"):
            texts = self._to_texts(samples)
            X_hash = self._texts_to_hash_matrix(texts)
            X_out = self._svd.transform(X_hash)
        else:
            X = np.atleast_2d(np.array(samples, dtype=np.float32))
            X_scaled = self._scaler.transform(X)
            if self._pca is not None:
                X_out = self._pca.transform(X_scaled)
            else:
                X_out = X_scaled

        return self._pad_or_truncate(X_out)

    def fit_transform(self, samples: list) -> np.ndarray:
        """Fit and transform in a single step.

        Returns
        -------
        np.ndarray, shape (n_samples, output_dim), dtype float32
        """
        return self.fit(samples).transform(samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_kind(self, samples: list) -> str:
        """Return 'text', 'dict', or 'numeric' based on the first sample."""
        s = samples[0]
        if isinstance(s, str):
            return "text"
        if isinstance(s, dict):
            return "dict"
        return "numeric"

    def _to_texts(self, samples: list) -> list[str]:
        if self._kind == "dict":
            return [json.dumps(s, sort_keys=True, default=str) for s in samples]
        return [str(s) for s in samples]

    def _char_ngrams(self, text: str, ns: tuple[int, ...] = (3, 4)) -> list[str]:
        """Extract character n-grams of lengths *ns* from *text*."""
        text = text.lower()
        grams: list[str] = []
        for n in ns:
            for i in range(len(text) - n + 1):
                grams.append(text[i : i + n])
        return grams

    def _hash_ngrams(self, grams: list[str]) -> np.ndarray:
        """Map n-grams to a count vector of length ``hash_features``."""
        vec = np.zeros(self.hash_features, dtype=np.float32)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16) % self.hash_features
            vec[h] += 1.0
        return vec

    def _texts_to_hash_matrix(self, texts: list[str]) -> np.ndarray:
        """Return (n, hash_features) float32 matrix from a list of texts."""
        rows = [self._hash_ngrams(self._char_ngrams(t)) for t in texts]
        return np.array(rows, dtype=np.float32)

    def _pad_or_truncate(self, X: np.ndarray) -> np.ndarray:
        """Ensure output has exactly ``output_dim`` columns."""
        n, d = X.shape
        if d < self.output_dim:
            pad = np.zeros((n, self.output_dim - d), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)
        elif d > self.output_dim:
            X = X[:, : self.output_dim]
        return X.astype(np.float32)
