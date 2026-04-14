"""Stage 9 — Multi-task physics engine with a shared trunk and per-task heads.

Architecture
------------
One shared :class:`~physml.neural_engine.NeuralPhysicsEngine` trunk provides
the attention-transformed feature representation that all tasks share.  Each
task gets its own lightweight MLP head trained on top of the shared
representation.

The Mycelium metaphor: the trunk is the main fungal network; per-task heads
are the individual hyphae growing into different substrates (targets).

Usage
-----
::

    from physml.multitask_engine import MultiTaskPhysicsEngine

    engine = MultiTaskPhysicsEngine()

    # Train the shared trunk on the first (or most representative) task
    engine.fit_trunk(X_sales, y_revenue, is_classifier=False)

    # Fine-tune task-specific heads
    engine.fit_task("churn", X_users, y_churn, is_classifier=True)
    engine.fit_task("revenue", X_sales, y_revenue, is_classifier=False)
    engine.fit_task("segment", X_users, y_segment, is_classifier=True)

    # Predict using a specific head
    preds = engine.predict_task("churn", X_new)
    print(engine.list_tasks())   # ["churn", "revenue", "segment"]

Multi-task with PhysicsAgent
-----------------------------
::

    from physml.agent import PhysicsAgent

    agent = PhysicsAgent(engine, task_id="churn", query_strategy="entropy")
    action = agent.observe(X_new)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MultiTaskPhysicsEngine:
    """Shared-trunk, per-task-head neural engine.

    The trunk (attention block + first MLP layers) is trained once on seed
    data and then frozen.  Per-task output heads are lightweight MLPs trained
    on the attended representation.  New tasks can be added at any time
    without retraining the trunk.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default (256, 128)
        Trunk MLP architecture.
    head_hidden_sizes : tuple of int, default (64,)
        Architecture of each task-specific head MLP.
    max_attend_features : int, default 60
        Maximum number of features passed to the attention block.
    alpha : float, default 1e-4
        L2 regularisation for all MLP layers.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        head_hidden_sizes: tuple[int, ...] = (64,),
        max_attend_features: int = 60,
        alpha: float = 1e-4,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.head_hidden_sizes = head_hidden_sizes
        self.max_attend_features = max_attend_features
        self.alpha = alpha

        self._trunk: Any = None  # NeuralPhysicsEngine fitted trunk
        self._heads: dict[str, Any] = {}  # task_id → sklearn MLP
        self._task_meta: dict[str, dict[str, Any]] = {}  # task_id → metadata

    # ------------------------------------------------------------------
    # Trunk training
    # ------------------------------------------------------------------

    def fit_trunk(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        is_classifier: bool = False,
        n_epochs: int = 300,
        lr: float = 0.001,
        random_state: int = 42,
    ) -> "MultiTaskPhysicsEngine":
        """Train the shared trunk on representative data.

        After this call, the attention block is frozen.  All subsequent
        :meth:`fit_task` calls transform their data through this block before
        training a task-specific head.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        is_classifier : bool, default False
        n_epochs : int, default 300
        lr : float, default 0.001
        random_state : int, default 42

        Returns
        -------
        self
        """
        from physml.neural_engine import NeuralPhysicsEngine

        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y)

        trunk = NeuralPhysicsEngine(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_attend_features=self.max_attend_features,
            alpha=self.alpha,
        )
        trunk.fit_model(
            X_arr,
            y_arr,
            is_classifier=is_classifier,
            n_epochs=n_epochs,
            lr=lr,
            random_state=random_state,
        )
        self._trunk = trunk
        return self

    # ------------------------------------------------------------------
    # Per-task head training
    # ------------------------------------------------------------------

    def fit_task(
        self,
        task_id: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        is_classifier: bool | None = None,
        n_epochs: int = 200,
        lr: float = 0.001,
        random_state: int = 42,
    ) -> "MultiTaskPhysicsEngine":
        """Train a task-specific head using the shared trunk representation.

        If the trunk has not been trained yet, it is auto-initialised on this
        task's data before the head is fitted.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task.  Used to route predict calls.
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        is_classifier : bool or None
            If None, inferred from ``y`` (integer labels with ≤ 20 unique
            values → classifier, otherwise regressor).
        n_epochs : int, default 200
        lr : float, default 0.001
        random_state : int, default 42

        Returns
        -------
        self
        """
        try:
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise ImportError("scikit-learn is required for MultiTaskPhysicsEngine") from exc

        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y)

        # Infer task kind if not specified
        if is_classifier is None:
            unique = np.unique(y_arr)
            is_clf = (len(unique) <= 20) and np.issubdtype(y_arr.dtype, np.integer)
        else:
            is_clf = bool(is_classifier)

        # Auto-init trunk on first task
        if self._trunk is None:
            self.fit_trunk(
                X_arr,
                y_arr,
                is_classifier=is_clf,
                n_epochs=n_epochs,
                lr=lr,
                random_state=random_state,
            )

        # Transform through trunk attention (frozen)
        X_aug = self._transform_trunk(X_arr)

        # Build and fit the task-specific head
        n = X_arr.shape[0]
        batch = min(max(32, n // 10), 256)
        head_kwargs = dict(
            hidden_layer_sizes=self.head_hidden_sizes,
            activation="relu",
            solver="adam",
            alpha=float(self.alpha),
            batch_size=batch,
            learning_rate="adaptive",
            learning_rate_init=float(lr),
            max_iter=int(n_epochs),
            random_state=int(random_state),
            early_stopping=False,
        )

        label_enc: Any = None
        if is_clf:
            label_enc = LabelEncoder()
            y_enc = label_enc.fit_transform(y_arr.astype(str))
            head = MLPClassifier(**head_kwargs)
            head.fit(X_aug, y_enc)
        else:
            head = MLPRegressor(**head_kwargs)
            head.fit(X_aug, y_arr.astype(float))

        self._heads[task_id] = head
        self._task_meta[task_id] = {
            "is_classifier": is_clf,
            "label_enc": label_enc,
            "target_dtype": y_arr.dtype,
            "n_input_features": X_arr.shape[1],
        }
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_task(self, task_id: str, X: np.ndarray) -> np.ndarray:
        """Predict using the task-specific head.

        Parameters
        ----------
        task_id : str
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)

        Raises
        ------
        KeyError
            If ``task_id`` has not been fitted via :meth:`fit_task`.
        RuntimeError
            If the trunk has not been initialised.
        """
        if self._trunk is None:
            raise RuntimeError("Trunk not trained.  Call fit_trunk() or fit_task() first.")
        if task_id not in self._heads:
            raise KeyError(
                f"Task {task_id!r} not found.  "
                f"Known tasks: {list(self._heads)}.  "
                "Call fit_task(task_id, X, y) first."
            )

        X_arr = np.atleast_2d(X)
        X_aug = self._transform_trunk(X_arr)
        head = self._heads[task_id]
        meta = self._task_meta[task_id]

        preds = head.predict(X_aug)

        if meta["is_classifier"] and meta["label_enc"] is not None:
            preds = meta["label_enc"].inverse_transform(preds)
            try:
                preds = preds.astype(meta["target_dtype"])
            except (ValueError, TypeError):
                pass

        return preds

    def predict_proba_task(self, task_id: str, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for a classification task.

        Parameters
        ----------
        task_id : str
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)

        Raises
        ------
        KeyError
            If ``task_id`` is unknown.
        ValueError
            If the task is a regression task.
        """
        if task_id not in self._heads:
            raise KeyError(f"Task {task_id!r} not found.  Known: {list(self._heads)}")
        meta = self._task_meta[task_id]
        if not meta["is_classifier"]:
            raise ValueError(
                f"Task {task_id!r} is a regression task; predict_proba is not available."
            )
        X_arr = np.atleast_2d(X)
        X_aug = self._transform_trunk(X_arr)
        return self._heads[task_id].predict_proba(X_aug)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tasks(self) -> list[str]:
        """Return a list of all fitted task IDs."""
        return list(self._heads)

    def task_info(self, task_id: str) -> dict[str, Any]:
        """Return metadata for a fitted task.

        Parameters
        ----------
        task_id : str

        Returns
        -------
        dict with keys: is_classifier, target_dtype, n_input_features.
        """
        if task_id not in self._task_meta:
            raise KeyError(f"Task {task_id!r} not found.  Known: {list(self._heads)}")
        meta = self._task_meta[task_id]
        return {
            "is_classifier": meta["is_classifier"],
            "target_dtype": str(meta["target_dtype"]),
            "n_input_features": meta["n_input_features"],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the engine to disk using joblib.

        Parameters
        ----------
        path : str or Path
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for persistence") from exc
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "MultiTaskPhysicsEngine":
        """Load a previously saved engine.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        MultiTaskPhysicsEngine

        Raises
        ------
        TypeError
            If the file does not contain a :class:`MultiTaskPhysicsEngine`.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for persistence") from exc
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected MultiTaskPhysicsEngine, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transform_trunk(self, X_arr: np.ndarray) -> np.ndarray:
        """Apply the shared trunk attention transform to X_arr."""
        X_att = self._trunk.attn_.transform(X_arr)
        return np.hstack([X_arr, X_att])
