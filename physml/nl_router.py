"""Stage 106 — NaturalLanguageRouter: text command → action routing.

Converts free-text user commands into structured ``RoutedAction`` objects
without requiring a cloud LLM.  Uses a two-stage pipeline:

1. **Intent classification** — TF-IDF + cosine similarity against a set of
   registered intent templates (falls back to fuzzy keyword matching when
   scikit-learn is unavailable).
2. **Entity extraction** — regex-based extraction of common entity types
   (file paths, numbers, date strings, quoted strings, key=value pairs).

This enables ``MyceliumSystem`` (Stage 100) to accept natural-language goals
from users and route them to the correct tool or subsystem.

Usage
-----
::

    from physml.nl_router import NaturalLanguageRouter, Intent

    router = NaturalLanguageRouter()
    router.register(Intent("predict", ["predict this", "run inference on", "what is"]))
    router.register(Intent("train",   ["train on", "learn from", "fit the model to"]))
    router.register(Intent("report",  ["show stats", "give me a report", "how is the model"]))

    result = router.route("predict the outcome for these values: 1.2 3.4 5.6")
    print(result.intent)    # "predict"
    print(result.entities)  # {"numbers": [1.2, 3.4, 5.6]}
    print(result.confidence) # 0.87
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """A named intent with example utterances for matching.

    Parameters
    ----------
    name : str
        Unique intent identifier (e.g. ``"predict"``, ``"train"``).
    examples : list of str
        Example phrases that trigger this intent.
    metadata : dict, optional
        Arbitrary extra data forwarded into ``RoutedAction.metadata``.
    """
    name: str
    examples: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutedAction:
    """Result of routing a text command.

    Attributes
    ----------
    intent : str
        Matched intent name, or ``"unknown"`` if no match.
    confidence : float
        Match score in [0, 1].
    entities : dict
        Extracted entities keyed by type
        (``"numbers"``, ``"paths"``, ``"quoted"``, ``"kv"``).
    raw_text : str
        The original input text.
    metadata : dict
        Forwarded from the matched :class:`Intent`.
    """
    intent: str
    confidence: float
    entities: Dict[str, Any]
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NaturalLanguageRouter
# ---------------------------------------------------------------------------

class NaturalLanguageRouter:
    """Route free-text commands to registered intents.

    Parameters
    ----------
    min_confidence : float, default 0.15
        Minimum cosine similarity required to return a non-``"unknown"``
        intent.  Lowering this increases recall; raising it increases
        precision.
    use_tfidf : bool, default True
        When ``True``, attempt to use TF-IDF vectorisation for similarity
        scoring.  Falls back to keyword matching if scikit-learn is
        unavailable.
    """

    def __init__(
        self,
        min_confidence: float = 0.15,
        use_tfidf: bool = True,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.use_tfidf = bool(use_tfidf)
        self._intents: List[Intent] = []
        self._vectorizer: Any = None
        self._intent_matrix: Any = None
        self._dirty: bool = True  # needs re-fit after register()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, intent: Intent) -> "NaturalLanguageRouter":
        """Register a new :class:`Intent`.

        Parameters
        ----------
        intent : Intent

        Returns
        -------
        self
        """
        self._intents.append(intent)
        self._dirty = True
        return self

    def register_many(self, intents: Sequence[Intent]) -> "NaturalLanguageRouter":
        """Register multiple intents at once."""
        for i in intents:
            self._intents.append(i)
        self._dirty = True
        return self

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, text: str) -> RoutedAction:
        """Route *text* to the best-matching intent.

        Parameters
        ----------
        text : str
            Raw user input (any language, but English works best).

        Returns
        -------
        RoutedAction
        """
        text = str(text).strip()
        entities = _extract_entities(text)

        if not self._intents:
            return RoutedAction("unknown", 0.0, entities, text)

        if self._dirty:
            self._fit()

        confidence, best_name, meta = self._score(text)

        if confidence < self.min_confidence:
            return RoutedAction("unknown", confidence, entities, text)

        return RoutedAction(best_name, confidence, entities, text, meta)

    # ------------------------------------------------------------------
    # Internal: vectorisation + scoring
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Build or rebuild the intent match model."""
        all_examples: List[str] = []
        self._example_map: List[str] = []  # example → intent name
        self._example_meta: List[Dict[str, Any]] = []
        for intent in self._intents:
            for ex in intent.examples:
                all_examples.append(ex.lower())
                self._example_map.append(intent.name)
                self._example_meta.append(intent.metadata)

        if self.use_tfidf:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                import numpy as np
                self._vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                )
                self._intent_matrix = self._vectorizer.fit_transform(all_examples)
                self._use_tfidf_active = True
            except Exception:
                self._use_tfidf_active = False
        else:
            self._use_tfidf_active = False

        self._dirty = False

    def _score(self, text: str) -> tuple[float, str, Dict[str, Any]]:
        """Return (confidence, intent_name, metadata) for *text*."""
        query = text.lower()

        if self._use_tfidf_active and self._vectorizer is not None:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._intent_matrix)[0]
            best_idx = int(np.argmax(sims))
            return float(sims[best_idx]), self._example_map[best_idx], self._example_meta[best_idx]

        # Keyword fallback: count word overlaps
        best_score = 0.0
        best_name = "unknown"
        best_meta: Dict[str, Any] = {}
        q_words = set(re.findall(r"\w+", query))
        for ex, name, meta in zip(
            [e.lower() for intent in self._intents for e in intent.examples],
            self._example_map,
            self._example_meta,
        ):
            ex_words = set(re.findall(r"\w+", ex))
            if not ex_words:
                continue
            overlap = len(q_words & ex_words) / len(ex_words)
            if overlap > best_score:
                best_score = overlap
                best_name = name
                best_meta = meta
        return best_score, best_name, best_meta

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_intents(self) -> List[str]:
        """Return registered intent names."""
        return [i.name for i in self._intents]

    def __repr__(self) -> str:
        return (
            f"NaturalLanguageRouter("
            f"n_intents={len(self._intents)}, "
            f"min_confidence={self.min_confidence})"
        )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _extract_entities(text: str) -> Dict[str, Any]:
    """Extract common entity types from *text*.

    Returns
    -------
    dict with keys:
        ``"numbers"``  — list of float
        ``"paths"``    — list of str (Unix/Windows paths)
        ``"quoted"``   — list of str (double- or single-quoted substrings)
        ``"kv"``       — dict of str→str (key=value pairs)
    """
    entities: Dict[str, Any] = {}

    # Numbers (int and float, including negative)
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if numbers:
        entities["numbers"] = numbers

    # File paths (absolute Unix, Windows drive, or relative with extension)
    paths = re.findall(
        r"(?:/[\w./-]+|[A-Za-z]:\\[\w\\. /-]+|[\w./\\-]+\.(?:csv|json|pkl|txt|py|xlsx|tsv|parquet))",
        text,
    )
    if paths:
        entities["paths"] = paths

    # Quoted strings
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    flat_quoted = [q[0] or q[1] for q in quoted]
    if flat_quoted:
        entities["quoted"] = flat_quoted

    # key=value or key: value pairs
    kv_pairs = re.findall(r"(\w+)\s*[=:]\s*([\"']?[\w./]+[\"']?)", text)
    if kv_pairs:
        entities["kv"] = {k: v.strip("'\"") for k, v in kv_pairs}

    return entities
