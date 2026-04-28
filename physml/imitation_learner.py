"""physml.imitation_learner — Learn to perform tasks by watching the user.

:class:`ImitationLearner` trains a lightweight policy model on
:class:`~physml.macro_recorder.MacroSequence` recordings.  Given the
current screen context it predicts the most likely next action —
enabling Mycelium to offer proactive automation suggestions.

Architecture
------------
* Features: one-hot action-type + normalised x/y + app-name hash + key/text hash
* Model: scikit-learn ``HistGradientBoostingClassifier`` (same as the core engine)
* Training: standard supervised — (context_window → next_action)
* Inference: returns top-k predicted next actions with confidence scores

Usage::

    from physml.macro_recorder import MacroRecorder, ActionStep, ActionType
    from physml.imitation_learner import ImitationLearner

    recorder = MacroRecorder()
    seq = recorder.record_text_sequence("open_doc", [
        {"action_type": "click", "x": 100, "y": 50, "app_name": "Finder"},
        {"action_type": "double_click", "x": 200, "y": 300, "app_name": "Finder"},
        {"action_type": "type_text", "text": "report.pdf", "app_name": "Finder"},
    ])

    learner = ImitationLearner()
    learner.add_sequence(seq)
    learner.fit()

    suggestions = learner.predict_next(context_app="Finder", context_action="click")
    for s in suggestions:
        print(s["action_type"], f"{s['confidence']:.0%}")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from physml._log import get_logger
from physml.macro_recorder import ActionStep, ActionType, MacroSequence

_logger = get_logger(__name__)

# Known action types for one-hot encoding
_ACTION_TYPES = [
    ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK,
    ActionType.KEY_PRESS, ActionType.KEY_RELEASE, ActionType.TYPE_TEXT,
    ActionType.SCROLL, ActionType.DRAG, ActionType.WINDOW_CHANGE, ActionType.PAUSE,
]
_ACTION_IDX = {a: i for i, a in enumerate(_ACTION_TYPES)}

_SCREEN_W = 1920
_SCREEN_H = 1080
_CONTEXT_WINDOW = 3   # how many previous steps to use as context


@dataclass
class ActionSuggestion:
    """Predicted next action.

    Attributes
    ----------
    action_type : str
        Predicted action type.
    confidence : float
        Model confidence [0, 1].
    x, y : int or None
        Predicted coordinates (averaged from training examples).
    text : str or None
        Predicted text (most common in training).
    app_name : str
        Predicted app context.
    """

    action_type: str
    confidence: float
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    app_name: str = "unknown"


class ImitationLearner:
    """Train a policy model on macro recordings and suggest next actions.

    Parameters
    ----------
    context_window : int
        How many prior steps to use as features for next-action prediction.
    min_sequences : int
        Minimum number of sequences required before fitting.
    """

    def __init__(
        self,
        context_window: int = _CONTEXT_WINDOW,
        min_sequences: int = 1,
    ) -> None:
        self.context_window = context_window
        self.min_sequences = min_sequences
        self._sequences: List[MacroSequence] = []
        self._model: Any = None
        self._is_fitted = False
        self._label_map: Dict[int, str] = {}
        # Store training examples keyed by (predicted_action_type) for coordinate averaging
        self._coord_store: Dict[str, List[tuple]] = {}
        self._text_store: Dict[str, List[str]] = {}
        self._app_store: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_sequence(self, seq: MacroSequence) -> None:
        """Add a :class:`MacroSequence` to the training corpus."""
        self._sequences.append(seq)
        self._is_fitted = False

    def add_sequences(self, seqs: List[MacroSequence]) -> None:
        for s in seqs:
            self.add_sequence(s)

    def fit(self) -> bool:
        """Train the policy model on all added sequences.

        Returns ``True`` on success, ``False`` when not enough data.
        """
        if len(self._sequences) < self.min_sequences:
            _logger.info("ImitationLearner: need %d sequences, have %d", self.min_sequences, len(self._sequences))
            return False

        X, y = self._build_dataset()
        if len(X) < 4:
            _logger.info("ImitationLearner: not enough samples (%d)", len(X))
            return False

        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._model = HistGradientBoostingClassifier(max_iter=50, random_state=42)
            self._model.fit(X, y)
            classes = self._model.classes_
            self._label_map = {i: _ACTION_TYPES[c] if c < len(_ACTION_TYPES) else "unknown"
                               for i, c in enumerate(classes)}
            self._is_fitted = True
            _logger.info("ImitationLearner: fitted on %d samples from %d sequences", len(X), len(self._sequences))
            return True
        except Exception as exc:
            _logger.warning("ImitationLearner.fit error: %s", exc)
            return False

    def predict_next(
        self,
        context_steps: Optional[List[ActionStep]] = None,
        context_app: str = "unknown",
        context_action: str = ActionType.CLICK,
        top_k: int = 3,
    ) -> List[ActionSuggestion]:
        """Predict the top-k most likely next actions.

        Parameters
        ----------
        context_steps : list[ActionStep] or None
            The most recent steps (up to ``context_window``). When ``None``,
            a synthetic context from ``context_app`` + ``context_action`` is used.
        context_app : str
            Current active application (used when ``context_steps`` is None).
        context_action : str
            Last performed action type (used when ``context_steps`` is None).
        top_k : int
            Number of suggestions to return.

        Returns
        -------
        list[ActionSuggestion]
        """
        if not self._is_fitted or self._model is None:
            return self._heuristic_suggestions(context_action, context_app, top_k)

        # Build feature vector from context
        if context_steps is None:
            dummy = ActionStep(action_type=context_action, app_name=context_app)
            context_steps = [dummy]

        x = self._featurize_context(context_steps[-self.context_window:])
        x_arr = np.array([x])

        try:
            proba = self._model.predict_proba(x_arr)[0]
            top_indices = np.argsort(proba)[::-1][:top_k]
            suggestions = []
            for idx in top_indices:
                atype = self._label_map.get(int(idx), "unknown")
                conf = float(proba[idx])
                if conf < 0.01:
                    continue
                coords = self._coord_store.get(atype, [])
                avg_x = int(np.mean([c[0] for c in coords])) if coords else None
                avg_y = int(np.mean([c[1] for c in coords])) if coords else None
                texts = self._text_store.get(atype, [])
                common_text = max(set(texts), key=texts.count) if texts else None
                apps = self._app_store.get(atype, [])
                common_app = max(set(apps), key=apps.count) if apps else "unknown"
                suggestions.append(ActionSuggestion(
                    action_type=atype, confidence=conf,
                    x=avg_x, y=avg_y, text=common_text, app_name=common_app,
                ))
            return suggestions
        except Exception as exc:
            _logger.debug("ImitationLearner.predict_next error: %s", exc)
            return []

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def sequence_count(self) -> int:
        return len(self._sequences)

    def status(self) -> Dict[str, Any]:
        return {
            "sequences": len(self._sequences),
            "fitted": self._is_fitted,
            "context_window": self.context_window,
        }

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def _build_dataset(self) -> tuple:
        X, y = [], []
        for seq in self._sequences:
            steps = seq.steps
            for i in range(self.context_window, len(steps)):
                context = steps[max(0, i - self.context_window):i]
                target = steps[i]
                x = self._featurize_context(context)
                label = _ACTION_IDX.get(target.action_type, len(_ACTION_TYPES) - 1)
                X.append(x)
                y.append(label)
                # Store coord/text/app for coordinate averaging
                atype = target.action_type
                if target.x is not None and target.y is not None:
                    self._coord_store.setdefault(atype, []).append((target.x, target.y))
                if target.text:
                    self._text_store.setdefault(atype, []).append(target.text)
                self._app_store.setdefault(atype, []).append(target.app_name)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def _featurize_context(self, steps: List[ActionStep]) -> List[float]:
        """Convert a context window into a fixed-size float feature vector."""
        feats: List[float] = []
        # Pad / truncate to context_window
        padded = ([None] * (self.context_window - len(steps))) + list(steps)
        for step in padded:
            if step is None:
                feats.extend([0.0] * (len(_ACTION_TYPES) + 6))
                continue
            # One-hot action type
            oh = [0.0] * len(_ACTION_TYPES)
            idx = _ACTION_IDX.get(step.action_type, 0)
            oh[idx] = 1.0
            feats.extend(oh)
            # Normalised coords
            feats.append((step.x or 0) / _SCREEN_W)
            feats.append((step.y or 0) / _SCREEN_H)
            # App hash
            feats.append(self._str_hash(step.app_name))
            # Key/text hash
            feats.append(self._str_hash(step.key or step.text or ""))
            # Timestamp delta (normalised)
            feats.append(0.0)  # placeholder
            feats.append(0.0)  # placeholder
        return feats

    @staticmethod
    def _str_hash(s: str) -> float:
        """Map string to [0, 1] via MD5."""
        if not s:
            return 0.0
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        return (h % 100_000) / 100_000.0

    def _heuristic_suggestions(
        self, context_action: str, context_app: str, top_k: int
    ) -> List[ActionSuggestion]:
        """Simple heuristic fallback when model is not fitted."""
        # Most common action after each action type
        _NEXT = {
            ActionType.CLICK: [ActionType.TYPE_TEXT, ActionType.CLICK, ActionType.SCROLL],
            ActionType.DOUBLE_CLICK: [ActionType.TYPE_TEXT, ActionType.KEY_PRESS],
            ActionType.TYPE_TEXT: [ActionType.KEY_PRESS, ActionType.CLICK],
            ActionType.KEY_PRESS: [ActionType.CLICK, ActionType.TYPE_TEXT],
            ActionType.SCROLL: [ActionType.CLICK, ActionType.SCROLL],
        }
        candidates = _NEXT.get(context_action, [ActionType.CLICK, ActionType.TYPE_TEXT])
        return [
            ActionSuggestion(action_type=c, confidence=0.5 - 0.1 * i, app_name=context_app)
            for i, c in enumerate(candidates[:top_k])
        ]
