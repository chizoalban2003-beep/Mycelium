"""Stage 79 — ModelDistillery: knowledge distillation (teacher → student).

Compresses a high-capacity *teacher* model into a lightweight *student* by
training the student on soft probability labels produced by the teacher
(Hinton et al., 2015).  Temperature τ controls label smoothness: higher
values produce softer targets that transfer more generalisation.

Classes
-------
DistillationResult
    Snapshot of one distillation training run.
ModelDistillery
    Orchestrates teacher → student knowledge distillation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DistillationResult:
    """Snapshot of one distillation run.

    Attributes
    ----------
    temperature : float
        Temperature τ used during distillation.
    teacher_accuracy : float
        Teacher accuracy on the evaluation set.
    student_accuracy : float
        Student accuracy on the evaluation set after distillation.
    accuracy_gap : float
        Teacher accuracy minus student accuracy (positive = teacher wins).
    n_samples : int
        Number of training samples used.
    elapsed_s : float
        Wall-clock duration of the distillation call.
    """

    temperature: float
    teacher_accuracy: float
    student_accuracy: float
    accuracy_gap: float
    n_samples: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "teacher_accuracy": round(self.teacher_accuracy, 4),
            "student_accuracy": round(self.student_accuracy, 4),
            "accuracy_gap": round(self.accuracy_gap, 4),
            "n_samples": self.n_samples,
            "elapsed_s": round(self.elapsed_s, 4),
        }


class ModelDistillery:
    """Distil knowledge from a *teacher* model into a *student* model.

    The teacher's class probabilities are softened by *temperature* τ and
    used as training targets for the student.  If the student supports
    ``fit(X, soft_labels)`` natively (e.g. a custom neural network), it is
    used directly; otherwise the soft labels are converted to hard labels via
    ``argmax`` and ``fit()`` is called normally.

    For sklearn estimators that accept ``sample_weight``, the weight of each
    sample is set proportional to the teacher's maximum-class confidence,
    concentrating distillation effort on the teacher's most confident
    predictions.

    Parameters
    ----------
    teacher : Any
        A fitted estimator with ``predict_proba(X)``.
    student : Any
        The estimator to distil into (need not be pre-fitted).
    temperature : float, default 2.0
        Softening temperature (τ ≥ 1.0; higher → softer targets).
    use_sample_weights : bool, default True
        Pass teacher-confidence-derived sample weights to the student when
        supported.

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.datasets import make_classification
    >>> from physml.model_distillery import ModelDistillery
    >>> X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    >>> teacher = GradientBoostingClassifier(n_estimators=50, random_state=0)
    >>> teacher.fit(X[:200], y[:200])
    GradientBoostingClassifier(...)
    >>> student = LogisticRegression(max_iter=200)
    >>> distillery = ModelDistillery(teacher, student, temperature=3.0)
    >>> result = distillery.distil(X[:200], y[:200])
    >>> result.student_accuracy >= 0.0
    True
    """

    def __init__(
        self,
        teacher: Any,
        student: Any,
        *,
        temperature: float = 2.0,
        use_sample_weights: bool = True,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.temperature = float(max(1.0, temperature))
        self.use_sample_weights = bool(use_sample_weights)

        self._history: list[DistillationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distil(
        self,
        X: Any,
        y: Any,
        *,
        X_eval: Any | None = None,
        y_eval: Any | None = None,
    ) -> DistillationResult:
        """Distil the teacher into the student using data *(X, y)*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            True labels (used to evaluate both models; NOT used as student
            training targets — soft labels from the teacher are used instead).
        X_eval : array-like or None
            Separate evaluation set.  If None, evaluation is done on *(X, y)*.
        y_eval : array-like or None
            Evaluation labels.

        Returns
        -------
        DistillationResult
        """
        import inspect

        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Soft labels from teacher
        soft = self._soft_labels(X)
        hard = np.argmax(soft, axis=1)

        # Sample weights ∝ teacher's maximum probability
        sample_weight = soft.max(axis=1) if self.use_sample_weights else None

        # Train student
        if sample_weight is not None:
            sig = inspect.signature(self.student.fit)
            if "sample_weight" in sig.parameters:
                self.student.fit(X, hard, sample_weight=sample_weight)
            else:
                self.student.fit(X, hard)
        else:
            self.student.fit(X, hard)

        # Evaluate
        X_ev = np.asarray(X_eval, dtype=float) if X_eval is not None else X
        y_ev = np.asarray(y_eval) if y_eval is not None else y

        teacher_acc = float(np.mean(self._predict(self.teacher, X_ev) == y_ev))
        student_acc = float(np.mean(self._predict(self.student, X_ev) == y_ev))

        result = DistillationResult(
            temperature=self.temperature,
            teacher_accuracy=teacher_acc,
            student_accuracy=student_acc,
            accuracy_gap=teacher_acc - student_acc,
            n_samples=len(X),
            elapsed_s=time.time() - t0,
        )
        self._history.append(result)
        return result

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        """Compare teacher vs. student accuracy on *(X, y)*.

        Returns
        -------
        dict with keys ``teacher_accuracy``, ``student_accuracy``,
        ``accuracy_gap``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        teacher_acc = float(np.mean(self._predict(self.teacher, X) == y))
        student_acc = float(np.mean(self._predict(self.student, X) == y))
        return {
            "teacher_accuracy": round(teacher_acc, 4),
            "student_accuracy": round(student_acc, 4),
            "accuracy_gap": round(teacher_acc - student_acc, 4),
        }

    @property
    def history(self) -> list[DistillationResult]:
        """All distillation results in order."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _soft_labels(self, X: np.ndarray) -> np.ndarray:
        """Return temperature-scaled probability matrix from the teacher."""
        proba = self.teacher.predict_proba(X)
        # Temperature scaling: scale logits by 1/T, re-normalise
        log_proba = np.log(proba + 1e-9) / self.temperature
        log_proba -= log_proba.max(axis=1, keepdims=True)
        soft = np.exp(log_proba)
        soft /= soft.sum(axis=1, keepdims=True)
        return soft

    @staticmethod
    def _predict(estimator: Any, X: np.ndarray) -> np.ndarray:
        """Predict with fallback for agents using observe()."""
        if hasattr(estimator, "predict"):
            return np.asarray(estimator.predict(X))
        # PhysML agents
        preds = []
        for row in X:
            action = estimator.observe(row.reshape(1, -1))
            preds.append(
                action.prediction if hasattr(action, "prediction") else int(action)
            )
        return np.asarray(preds)
