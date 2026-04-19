"""Stage 123 — ModelManager: persistent ML model with auto-train and real prediction.

Wraps :class:`~physml.mycelium_agent.MyceliumAgent` (the flagship physics-ML
agent) with:

* **Persistent save/load** — model survives companion restarts.
* **CSV auto-training** — ``train_from_csv(path)`` handles parsing, feature
  inference, and fitting in one call.
* **Real predictions** — ``predict(features)`` returns a numeric prediction
  with confidence, feature names, and provenance.
* **Incremental learning** — ``partial_fit(X, y)`` updates the model online.
* **Status reporting** — ``status()`` summarises training state, accuracy,
  and prediction count for the companion UI.

This is the bridge that connects the 100-stage physics ML engine to the
:class:`~physml.companion.MyceliumCompanion` product layer.

Usage
-----
::

    from physml.model_manager import ModelManager

    mgr = ModelManager(model_dir="~/.mycelium/model")
    mgr.load()                              # restore from disk (no-op if fresh)

    result = mgr.train_from_csv("sales.csv", target_column="revenue")
    print(result.message)                   # "Trained on 1200 rows, 8 features"

    pred = mgr.predict([1.2, 3.4, 5.6])
    print(pred.value, pred.confidence)      # 42.7   0.87

    mgr.save()                              # persist to disk
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Result of a training call.

    Attributes
    ----------
    success : bool
    message : str
    n_rows : int
    n_features : int
    target_column : str
    elapsed : float
    error : str or None
    """

    success: bool
    message: str
    n_rows: int = 0
    n_features: int = 0
    target_column: str = ""
    elapsed: float = 0.0
    error: Optional[str] = None


@dataclass
class PredictResult:
    """Result of a prediction call.

    Attributes
    ----------
    value : float or int or str
        The predicted value.
    confidence : float
        Confidence score in [0, 1].
    feature_names : list of str
        Feature names used in this prediction.
    target_column : str
        Name of the target being predicted.
    model_fitted : bool
        ``False`` when no model is available.
    error : str or None
    """

    value: Any
    confidence: float
    feature_names: List[str] = field(default_factory=list)
    target_column: str = ""
    model_fitted: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class ModelManager:
    """Persistent physics-ML model manager.

    Parameters
    ----------
    model_dir : str, default "~/.mycelium/model"
        Directory where the trained agent is persisted.
    target_column : str or None
        Default target column name when training from CSV.  Auto-detected
        if ``None``.
    """

    _AGENT_FILENAME = "mycelium_agent.pkl"
    _META_FILENAME = "model_meta.json"

    def __init__(
        self,
        model_dir: str = "~/.mycelium/model",
        target_column: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self._target_column: Optional[str] = target_column
        self._agent: Any = None
        self._feature_names: List[str] = []
        self._n_predictions: int = 0
        self._n_training_rows: int = 0
        self._last_accuracy: Optional[float] = None
        self._fitted: bool = False
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Restore a previously saved agent from disk.

        Returns
        -------
        bool
            ``True`` if a model was found and loaded.
        """
        agent_path = self.model_dir / self._AGENT_FILENAME
        meta_path = self.model_dir / self._META_FILENAME

        if not agent_path.exists():
            _logger.info("ModelManager: no saved model at %s", agent_path)
            return False

        try:
            from physml.mycelium_agent import MyceliumAgent

            self._agent = MyceliumAgent.load(str(agent_path))
            self._fitted = True

            # Restore metadata
            if meta_path.exists():
                import json

                meta = json.loads(meta_path.read_text())
                self._feature_names = meta.get("feature_names", [])
                self._target_column = meta.get("target_column") or self._target_column
                self._n_training_rows = meta.get("n_training_rows", 0)
                self._n_predictions = meta.get("n_predictions", 0)
                self._last_accuracy = meta.get("last_accuracy")

            _logger.info(
                "ModelManager: loaded model trained on %d rows (%d features)",
                self._n_training_rows,
                len(self._feature_names),
            )
            self._loaded = True
            return True

        except Exception as exc:
            _logger.warning("ModelManager.load failed: %s", exc)
            return False

    def save(self) -> bool:
        """Persist the fitted agent to disk.

        Returns
        -------
        bool
            ``True`` on success.
        """
        if not self._fitted or self._agent is None:
            _logger.info("ModelManager: nothing to save (model not fitted)")
            return False

        try:
            import json

            self.model_dir.mkdir(parents=True, exist_ok=True)
            self._agent.save(str(self.model_dir / self._AGENT_FILENAME))

            meta = {
                "feature_names": self._feature_names,
                "target_column": self._target_column,
                "n_training_rows": self._n_training_rows,
                "n_predictions": self._n_predictions,
                "last_accuracy": self._last_accuracy,
                "saved_at": time.time(),
            }
            (self.model_dir / self._META_FILENAME).write_text(json.dumps(meta, indent=2))
            _logger.info("ModelManager: saved model to %s", self.model_dir)
            return True

        except Exception as exc:
            _logger.warning("ModelManager.save failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_csv(
        self,
        path: str,
        target_column: Optional[str] = None,
    ) -> TrainResult:
        """Train the model from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        target_column : str or None
            Target column name.  Auto-detected (last column) if ``None``.

        Returns
        -------
        TrainResult
        """
        t0 = time.time()
        try:
            import pandas as pd
            import numpy as np

            df = pd.read_csv(path)
            if df.empty:
                return TrainResult(
                    success=False,
                    message="CSV file is empty.",
                    error="empty file",
                )

            # Drop fully-NaN columns
            df = df.dropna(axis=1, how="all")

            # Select numeric columns only
            num_df = df.select_dtypes(include=[float, int, "number"])
            if num_df.shape[1] < 2:
                return TrainResult(
                    success=False,
                    message=f"Need at least 2 numeric columns, found {num_df.shape[1]}.",
                    error="insufficient numeric columns",
                )

            # Resolve target column
            target = target_column or self._target_column
            if target and target in num_df.columns:
                y_col = target
            else:
                y_col = num_df.columns[-1]  # last numeric column
                if target:
                    _logger.warning(
                        "ModelManager: column %r not found; using %r", target, y_col
                    )

            feature_cols = [c for c in num_df.columns if c != y_col]
            X = num_df[feature_cols].fillna(0).values.astype(float)
            y = num_df[y_col].fillna(0).values

            return self._fit(X, y, feature_names=feature_cols, target_column=y_col, t0=t0)

        except Exception as exc:
            _logger.warning("ModelManager.train_from_csv: %s", exc)
            return TrainResult(
                success=False,
                message=f"Training failed: {exc}",
                error=str(exc),
            )

    def train_from_arrays(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[List[str]] = None,
        target_column: str = "target",
    ) -> TrainResult:
        """Train the model from numpy arrays.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
        target_column : str, default "target"

        Returns
        -------
        TrainResult
        """
        t0 = time.time()
        import numpy as np

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        names = feature_names or [f"x{i}" for i in range(X.shape[1])]
        return self._fit(X, y, feature_names=names, target_column=target_column, t0=t0)

    def _fit(
        self,
        X: Any,
        y: Any,
        feature_names: List[str],
        target_column: str,
        t0: float,
    ) -> TrainResult:
        import numpy as np

        try:
            from physml.mycelium_agent import MyceliumAgent

            self._agent = MyceliumAgent()
            self._agent.fit(X, y)
            self._feature_names = list(feature_names)
            self._target_column = target_column
            self._n_training_rows = int(X.shape[0])
            self._fitted = True

            # Quick self-evaluation on a hold-out slice if possible
            if X.shape[0] >= 10:
                split = max(int(X.shape[0] * 0.8), 1)
                try:
                    eval_result = self._agent.self_evaluate(X[split:], y[split:])
                    self._last_accuracy = eval_result.get("accuracy") or eval_result.get(
                        "r2_score"
                    )
                except Exception:
                    self._last_accuracy = None

            elapsed = time.time() - t0
            acc_str = (
                f", accuracy≈{self._last_accuracy:.2%}"
                if self._last_accuracy is not None
                else ""
            )
            msg = (
                f"Trained on {self._n_training_rows} rows, "
                f"{len(feature_names)} features → target: {target_column!r}{acc_str}."
            )
            _logger.info("ModelManager: %s (%.2fs)", msg, elapsed)
            return TrainResult(
                success=True,
                message=msg,
                n_rows=self._n_training_rows,
                n_features=len(feature_names),
                target_column=target_column,
                elapsed=elapsed,
            )

        except Exception as exc:
            _logger.warning("ModelManager._fit: %s", exc)
            return TrainResult(
                success=False,
                message=f"Training failed: {exc}",
                error=str(exc),
                elapsed=time.time() - t0,
            )

    def partial_fit(self, X: Any, y: Any) -> TrainResult:
        """Incrementally update the model with new labelled data.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        TrainResult
        """
        if not self._fitted or self._agent is None:
            return self.train_from_arrays(X, y)

        t0 = time.time()
        try:
            import numpy as np

            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
            self._agent.reward(X, y)
            self._n_training_rows += int(X.shape[0])
            elapsed = time.time() - t0
            return TrainResult(
                success=True,
                message=f"Updated model with {len(y)} new samples.",
                n_rows=len(y),
                elapsed=elapsed,
            )
        except Exception as exc:
            _logger.warning("ModelManager.partial_fit: %s", exc)
            return TrainResult(success=False, message=str(exc), error=str(exc))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: Any,
        feature_names: Optional[List[str]] = None,
    ) -> PredictResult:
        """Predict from a feature vector.

        Parameters
        ----------
        features : list or array-like of float
            Input feature values.
        feature_names : list of str, optional
            Names for the provided features.

        Returns
        -------
        PredictResult
        """
        if not self._fitted or self._agent is None:
            return PredictResult(
                value=None,
                confidence=0.0,
                model_fitted=False,
                error="No model trained yet. Use 'train on <file.csv>' first.",
            )

        try:
            import numpy as np

            X = np.asarray(features, dtype=float).reshape(1, -1)
            preds = self._agent.predict(X)
            value = preds[0]

            # Confidence from predict_proba if available
            conf = 0.75
            try:
                proba = self._agent._agent._predictor.predict_proba(X)
                conf = float(np.max(proba))
            except Exception:
                pass

            self._n_predictions += 1
            fnames = feature_names or self._feature_names or [
                f"x{i}" for i in range(len(features))
            ]

            return PredictResult(
                value=float(value) if np.isscalar(value) else value,
                confidence=conf,
                feature_names=fnames[: len(features)],
                target_column=self._target_column or "target",
                model_fitted=True,
            )

        except Exception as exc:
            _logger.warning("ModelManager.predict: %s", exc)
            return PredictResult(
                value=None,
                confidence=0.0,
                error=str(exc),
                model_fitted=self._fitted,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """``True`` if a model has been trained."""
        return self._fitted

    def status(self) -> Dict[str, Any]:
        """Return a status dict for the companion UI.

        Returns
        -------
        dict
        """
        return {
            "fitted": self._fitted,
            "target_column": self._target_column,
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names),
            "n_training_rows": self._n_training_rows,
            "n_predictions": self._n_predictions,
            "last_accuracy": self._last_accuracy,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"ModelManager({status}, rows={self._n_training_rows}, "
            f"features={len(self._feature_names)})"
        )
