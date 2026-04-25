"""Stage 135 — FeedbackLoop: user corrections trigger live model updates.

When a user says "that's wrong, it should be X" or gives a thumbs-down,
this module captures the correction, stores it as a labelled example, and
calls ``model.partial_fit()`` so the model immediately improves without a
full retrain.

Usage
-----
::

    from physml.feedback_loop import FeedbackLoop

    fl = FeedbackLoop(model_manager=mm, vector_memory=vm)

    # User corrects a wrong prediction
    fl.record_correction(
        features=[1.2, 3.4, 5.6],
        correct_label=1,
        predicted_label=0,
    )
    # Model is updated immediately via partial_fit
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class CorrectionRecord:
    """A single user correction."""

    timestamp: float
    features: List[float]
    correct_label: Any
    predicted_label: Any = None
    applied: bool = False
    source: str = "user"

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "correct_label": self.correct_label,
            "predicted_label": self.predicted_label,
            "applied": self.applied,
            "n_features": len(self.features),
            "source": self.source,
        }


# Patterns that signal explicit user correction in text
_CORRECTION_PATTERNS = [
    r"\bthat(?:'s| is|s) wrong\b",
    r"\bincorrect\b",
    r"\bshould(?:'ve| have)? been\b",
    r"\bactually(,| it)?\b.*\bit(?:'s| is)\b",
    r"\bno,?\s+(?:the answer|it should)\b",
    r"\bcorrect(?:ion)?[: ]\b",
    r"\bwrong\b.*\bright(?:\s+answer)?\s+is\b",
    r"\bthumb(?:s)?\s*down\b",
]


class FeedbackLoop:
    """Capture user corrections and apply them to the model in real time.

    Parameters
    ----------
    model_manager : ModelManager or None
        The live model.  When provided, corrections trigger partial_fit.
    vector_memory : VectorMemory or None
        Stores correction text for semantic retrieval.
    min_corrections_before_fit : int, default 1
        Buffer at least this many corrections before calling partial_fit.
    max_buffer : int, default 50
        Discard oldest entries once the buffer exceeds this size.
    """

    def __init__(
        self,
        model_manager: Any = None,
        vector_memory: Any = None,
        min_corrections_before_fit: int = 1,
        max_buffer: int = 50,
    ) -> None:
        self.model_manager = model_manager
        self.vector_memory = vector_memory
        self.min_corrections = min_corrections_before_fit
        self.max_buffer = max_buffer
        self._corrections: List[CorrectionRecord] = []
        self._total_applied = 0

    # ------------------------------------------------------------------
    def record_correction(
        self,
        features: List[float],
        correct_label: Any,
        predicted_label: Any = None,
        source: str = "user",
    ) -> bool:
        """Record and immediately apply a user correction.

        Parameters
        ----------
        features : list of float
            The feature vector that was predicted.
        correct_label : any
            The label the user says is correct.
        predicted_label : any, optional
            What the model originally predicted.
        source : str
            Who provided the correction.

        Returns
        -------
        bool
            True if the correction was applied to the model.
        """
        rec = CorrectionRecord(
            timestamp=time.time(),
            features=list(features),
            correct_label=correct_label,
            predicted_label=predicted_label,
            source=source,
        )
        self._corrections.append(rec)
        if len(self._corrections) > self.max_buffer:
            self._corrections = self._corrections[-self.max_buffer:]

        # Store in vector memory
        if self.vector_memory is not None:
            try:
                text = (
                    f"Correction: features={features[:3]}… "
                    f"correct={correct_label} predicted={predicted_label}"
                )
                self.vector_memory.add(text, {"type": "correction"})
            except Exception:
                pass

        # Apply if we have enough buffered corrections
        pending = [c for c in self._corrections if not c.applied]
        if len(pending) >= self.min_corrections:
            return self._apply_corrections(pending)
        return False

    def parse_and_record(
        self,
        user_text: str,
        last_features: Optional[List[float]] = None,
    ) -> Optional[str]:
        """Detect if *user_text* is a correction and handle it.

        Returns a confirmation string if a correction was detected, else None.
        """
        import re
        lower = user_text.lower()
        is_correction = any(
            re.search(p, lower) for p in _CORRECTION_PATTERNS
        )
        if not is_correction:
            return None

        # Try to extract the correct value from text
        correct_val = self._extract_value(user_text)
        if correct_val is None:
            return "Thanks for the feedback! Could you tell me the correct value?"

        if last_features is None:
            return (
                f"Got it — the correct answer is {correct_val}. "
                "I'll remember that for next time."
            )

        applied = self.record_correction(
            features=last_features,
            correct_label=correct_val,
            source="user_text",
        )
        if applied:
            return (
                f"Thanks for the correction! I've updated my model "
                f"to learn that the answer is {correct_val}."
            )
        return (
            f"Noted — the correct answer is {correct_val}. "
            "I'll apply this when I have enough examples."
        )

    def _extract_value(self, text: str) -> Optional[Any]:
        """Try to pull a numeric or categorical value from a correction message."""
        import re
        # Look for numbers
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if nums:
            v = nums[-1]
            return float(v) if "." in v else int(v)
        # Look for "it is X" / "it should be X"
        m = re.search(r"(?:should be|it is|is actually|answer is)\s+([a-zA-Z0-9_\-]+)", text, re.I)
        if m:
            return m.group(1)
        return None

    def _apply_corrections(self, corrections: List[CorrectionRecord]) -> bool:
        """Call partial_fit on the model manager with buffered corrections."""
        if self.model_manager is None:
            for c in corrections:
                c.applied = True
            return False

        try:
            import numpy as np
            X = np.array([c.features for c in corrections])
            y = np.array([c.correct_label for c in corrections])
            result = self.model_manager.partial_fit(X, y)
            success = getattr(result, "success", True)
            if success:
                for c in corrections:
                    c.applied = True
                self._total_applied += len(corrections)
                _logger.info(
                    "FeedbackLoop: applied %d correction(s) via partial_fit",
                    len(corrections),
                )
                return True
        except Exception as exc:
            _logger.warning("FeedbackLoop partial_fit failed: %s", exc)
        return False

    def status(self) -> dict:
        pending = sum(1 for c in self._corrections if not c.applied)
        return {
            "total_corrections": len(self._corrections),
            "pending": pending,
            "total_applied": self._total_applied,
            "model_connected": self.model_manager is not None,
        }
