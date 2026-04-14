"""Neural physics engine — Stage 1 + 2 + 3 (continual learning) implementation.

Architecture
------------
Stage 1: 3-layer MLP backbone (sklearn MLPRegressor / MLPClassifier) with
pseudo-residual connections via feature concatenation.  Residual skip
connections mirror the "homeostasis" concept already present in
``PredictorRuntimeState``.

  Input (d features)
    → Feature Attention block        [Stage 2]
    → Concat [X_original, X_attended]
    → Dense(256) + ReLU              [hidden layer 1]
    → Dense(128) + ReLU              [hidden layer 2]
    → Output (regression: 1; classification: n_classes)

Stage 2: single-head feature-attention block before the MLP.  Attention
weights are derived from the training-data feature correlation matrix,
mapping the electrophoresis metaphor:
  - "charge"    = query-key similarity score (correlation between features)
  - "migration" = value-weighted aggregation across features

The attended representation is concatenated with the original features so
the MLP always has direct access to raw inputs (residual skip).

``WeightInfo`` objects in ``PredictionResult`` report attention weights as
feature importance — the same field that the estimator uses for
polynomial-feature selection.

Usage
-----
This module is used internally by ``PhysicsPredictor`` when
``backend="neural"``.  The public entry-point is
``run_neural_prediction``, which has the same call signature as
``run_physics_prediction``.

PyTorch is *not* required; the implementation uses NumPy + scikit-learn
(already in requirements.txt).  If ``torch`` is importable it is preferred
for GPU/autograd support in future extensions.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from physml.predictor import (
    BondInfo,
    EquilibriumZone,
    IterationInfo,
    MigrationInfo,
    PhysicsPlane,
    PredictionMetrics,
    PredictionResult,
    PredictorRuntimeState,
    WeightInfo,
    infer_feature_kind,
    infer_target_kind,
    update_predictor_state_from_result,
)

try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    MLPRegressor = None  # type: ignore[assignment]
    MLPClassifier = None  # type: ignore[assignment]
    LabelEncoder = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Stage 2 — Feature Attention Block
# ---------------------------------------------------------------------------

class _FeatureAttentionBlock:
    """Single-head feature attention using correlation-based attention scores.

    Treats each feature as a "query" that attends to all other features.
    The attention matrix is computed from the Pearson correlation matrix of
    the training data (no learned parameters — purely data-driven).

    This maps cleanly onto the electrophoresis metaphor:
    - charge         = query-key similarity (correlation between features)
    - migration      = value-weighted aggregation across features
    - terminal_speed = attention weight received by a feature
    """

    def __init__(self, max_attend_features: int = 60) -> None:
        self.max_attend_features = max_attend_features
        # (d_eff, d_eff) soft-attention matrix; set by fit()
        self.attn_matrix_: np.ndarray | None = None
        # (d,) mean attention received per feature; set by fit()
        self.feature_importance_: np.ndarray | None = None
        self.d_eff_: int = 0
        self.mu_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_FeatureAttentionBlock":
        """Compute softmax attention matrix from feature correlations."""
        n, d = X.shape
        d_eff = min(d, self.max_attend_features)
        self.d_eff_ = d_eff
        X_sub = X[:, :d_eff].astype(float)

        # Standardise (needed for stable correlation computation)
        mu = X_sub.mean(axis=0)
        std = X_sub.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        self.mu_ = mu
        self.std_ = std
        X_std = (X_sub - mu) / std
        X_std = np.nan_to_num(X_std, nan=0.0)

        # Pearson correlation matrix (d_eff, d_eff)
        corr = X_std.T @ X_std / max(n - 1, 1)

        # Scale by 1/sqrt(d_eff) (mirroring scaled dot-product attention)
        scores = corr / math.sqrt(max(d_eff, 1))

        # Softmax along axis=1 → row i: how much feature i attends to others
        scores_max = scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(scores - scores_max, -30.0, 0.0))
        self.attn_matrix_ = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-8)

        # Per-feature importance = mean attention *received* across all queries
        importance = self.attn_matrix_.mean(axis=0)  # (d_eff,)
        if d > d_eff:
            full_imp = np.zeros(d)
            full_imp[:d_eff] = importance
            self.feature_importance_ = full_imp
        else:
            self.feature_importance_ = importance
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply attention; return attended representation shape (n, d_eff)."""
        if self.attn_matrix_ is None:
            return X[:, : self.d_eff_]
        X_sub = X[:, : self.d_eff_].astype(float)
        X_sub = np.nan_to_num(X_sub, nan=0.0)
        # attended[b, i] = sum_j(attn[i, j] * x[b, j])
        # = X_sub @ attn.T
        return X_sub @ self.attn_matrix_.T  # (n, d_eff)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_dataframe(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str], LabelEncoder | None]:
    """Convert a combined train+test DataFrame to numeric matrices.

    Returns
    -------
    X : ndarray, shape (n, d)
        Feature matrix (numeric; categoricals one-hot-encoded; NaNs filled).
    y : ndarray, shape (n,)
        Target array.  For categorical targets encoded to integer labels.
    feature_names : list[str]
        Column names corresponding to X columns.
    feature_kinds : dict[str, str]
        Mapping from *original* column name to its inferred kind.
    label_enc : LabelEncoder or None
        Fitted LabelEncoder when target is categorical; None for numeric.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    feature_kinds: dict[str, str] = {
        c: infer_feature_kind(df[c]) for c in feature_cols
    }

    # Build numeric X
    x_parts: list[np.ndarray] = []
    out_names: list[str] = []
    out_kinds: dict[str, str] = {}

    for col in feature_cols:
        kind = feature_kinds[col]
        series = df[col]
        if kind in ("numeric", "datetime", "bool"):
            if kind == "datetime":
                arr = pd.to_numeric(
                    pd.to_datetime(series, errors="coerce").view("int64"),
                    errors="coerce",
                ).to_numpy(dtype=float)
            elif kind == "bool":
                arr = series.astype(float).to_numpy(dtype=float)
            else:
                arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            col_mean = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
            arr = np.where(np.isfinite(arr), arr, col_mean)
            x_parts.append(arr.reshape(-1, 1))
            out_names.append(col)
            out_kinds[col] = kind
        else:
            # One-hot encode categorical columns (top-20 levels + __OTHER__)
            dummies = pd.get_dummies(
                series.astype("string").fillna("__MISSING__"),
                prefix=col,
                dummy_na=False,
            )
            # Limit to top-20 + other to avoid dimension explosion
            if dummies.shape[1] > 20:
                top_cols = list(dummies.sum().nlargest(20).index)
                other = dummies[[c for c in dummies.columns if c not in top_cols]].any(axis=1)
                dummies = dummies[top_cols].copy()
                dummies[f"{col}___OTHER__"] = other.astype(int)
            arr_oh = dummies.to_numpy(dtype=float)
            for i, oh_col in enumerate(dummies.columns):
                x_parts.append(arr_oh[:, i].reshape(-1, 1))
                out_names.append(str(oh_col))
                out_kinds[str(oh_col)] = "bool"
            out_kinds[col] = kind  # keep original kind mapping

    if x_parts:
        X = np.hstack(x_parts)
    else:
        X = np.zeros((len(df), 1))
        out_names = ["__dummy__"]
        out_kinds = {"__dummy__": "numeric"}

    # Target
    y_raw = df[target_col].to_numpy()
    target_kind = infer_target_kind(df[target_col])
    label_enc: LabelEncoder | None = None
    if target_kind == "categorical":
        if _SKLEARN_AVAILABLE:
            label_enc = LabelEncoder()
            y = label_enc.fit_transform(y_raw.astype(str)).astype(int)
        else:
            # Fallback: manual integer encoding
            classes, y = np.unique(y_raw.astype(str), return_inverse=True)
            label_enc = None
    else:
        y = pd.to_numeric(pd.Series(y_raw), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    return X, y, out_names, out_kinds, label_enc


def _mlp_feature_importance(mlp: Any, n_features: int) -> np.ndarray:
    """L1 norm of first hidden-layer weights per input feature."""
    try:
        W = mlp.coefs_[0]  # (n_input_features, hidden_1)
        if W.shape[0] < n_features:
            # Input may include attended features
            imp = np.abs(W[:n_features]).sum(axis=1)
        else:
            imp = np.abs(W[:n_features]).sum(axis=1)
        imp_max = imp.max()
        return imp / (imp_max + 1e-8)
    except Exception:
        return np.ones(n_features) / max(n_features, 1)


def _build_weight_infos(
    feature_names: list[str],
    feature_kinds: dict[str, str],
    attn_importance: np.ndarray,
    mlp_importance: np.ndarray,
) -> list[WeightInfo]:
    """Combine attention + MLP importance into sorted WeightInfo list."""
    n = len(feature_names)
    attn_n = attn_importance[:n] if len(attn_importance) >= n else np.pad(attn_importance, (0, n - len(attn_importance)))
    mlp_n = mlp_importance[:n] if len(mlp_importance) >= n else np.pad(mlp_importance, (0, n - len(mlp_importance)))

    # Normalise each source to [0, 1]
    attn_max = attn_n.max() + 1e-8
    mlp_max = mlp_n.max() + 1e-8
    combined = 0.5 * attn_n / attn_max + 0.5 * mlp_n / mlp_max

    infos: list[WeightInfo] = []
    for i, name in enumerate(feature_names):
        kind = feature_kinds.get(name, "numeric")
        # Remap one-hot-encoded columns back to their category kind
        if kind not in ("numeric", "categorical", "datetime", "bool"):
            kind = "categorical"
        w = float(combined[i])
        infos.append(
            WeightInfo(
                feature=name,
                weight=w,
                method="neural_attention",
                feature_kind=kind,  # type: ignore[arg-type]
                signed=False,
            )
        )
    return sorted(infos, key=lambda x: abs(x.weight), reverse=True)


def _build_migration_infos(
    feature_names: list[str],
    feature_kinds: dict[str, str],
    X_train: np.ndarray,
    attn_importance: np.ndarray,
) -> list[MigrationInfo]:
    """Build minimal MigrationInfo for each feature from neural engine outputs."""
    n, d = X_train.shape
    d_eff = min(d, len(feature_names))
    infos: list[MigrationInfo] = []

    median_attn = float(np.median(attn_importance[:d_eff]))

    for i, name in enumerate(feature_names):
        if i >= d_eff:
            break
        col = X_train[:, i]
        finite = col[np.isfinite(col)]

        variance = float(np.var(finite)) if finite.size > 1 else 0.0
        std_val = math.sqrt(max(variance, 0.0))
        se = std_val / math.sqrt(max(len(finite), 1))
        density = float(len(finite)) / max(n, 1)
        mass = 1.0 / (std_val + 1.0)  # higher mass = more stable feature

        attn_w = float(attn_importance[i]) if i < len(attn_importance) else 0.0
        stable = attn_w >= median_attn

        # Map attention weight to arrival_speed / terminal_velocity
        arrival_speed = attn_w
        terminal_velocity = attn_w * 0.5
        viscosity = max(0.0, 1.0 - attn_w)
        direction: Literal["pulled", "repelled", "neutral"] = (
            "pulled" if attn_w > 0.01 else "neutral"
        )
        state: Literal["free", "dampened", "trapped"] = (
            "free" if stable else "dampened"
        )

        # Entropy proxy from value distribution
        if finite.size > 1:
            mn, mx = finite.min(), finite.max()
            if mx > mn:
                hist, _ = np.histogram(finite, bins=min(20, finite.size))
                prob = hist / hist.sum()
                prob = prob[prob > 0]
                entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
            else:
                entropy = 0.0
        else:
            entropy = 0.0

        kind = feature_kinds.get(name, "numeric")
        if kind not in ("numeric", "categorical", "datetime", "bool"):
            kind = "categorical"

        infos.append(
            MigrationInfo(
                feature=name,
                feature_kind=kind,  # type: ignore[arg-type]
                method="neural_attention",
                charge=attn_w,
                ionization="parametric",
                normality_p=None,
                p_value=None,
                mass=mass,
                stable=stable,
                complex_id=None,
                complex_size=None,
                entropy=entropy,
                variance=variance,
                standard_error=se,
                kl_divergence=0.0,
                density=density,
                viscosity=viscosity,
                terminal_velocity=terminal_velocity,
                arrival_speed=arrival_speed,
                direction=direction,
                state=state,
            )
        )
    return sorted(infos, key=lambda m: m.arrival_speed, reverse=True)


# ---------------------------------------------------------------------------
# Main Neural Engine
# ---------------------------------------------------------------------------

class NeuralPhysicsEngine:
    """3-layer MLP + single-head feature attention for tabular prediction.

    This class is a drop-in replacement for ``run_physics_prediction``.
    Call ``run()`` with the same signature to get a ``PredictionResult``.

    Architecture (Stage 1 + 2 combined)
    ------------------------------------
    Input (d features)
      → _FeatureAttentionBlock  (correlation-based soft attention)
      → concat [X_original, X_attended]        <- pseudo-residual skip
      → Dense(256) + ReLU
      → Dense(128) + ReLU
      → Output head

    The attention weights are propagated as ``WeightInfo.weight`` so the
    physics metaphor (charge = attention score) is preserved throughout.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...], default (256, 128)
    max_attend_features : int, default 60
        Maximum number of features to include in the attention block.
        Features beyond this cap pass through directly.
    alpha : float, default 1e-4
        L2 regularisation for the MLP.
    embed_dim : int
        Unused; kept for API forwards-compatibility with future torch backend.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        max_attend_features: int = 60,
        alpha: float = 1e-4,
        embed_dim: int = 16,  # reserved for future torch backend
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_attend_features = max_attend_features
        self.alpha = alpha
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_mlp(
        self,
        is_classifier: bool,
        n_epochs: int,
        lr: float,
        random_state: int,
        n_train: int,
        early_stopping: bool | None = None,
    ) -> Any:
        """Instantiate an sklearn MLP estimator."""
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for NeuralPhysicsEngine")

        batch = min(max(32, n_train // 10), 256)
        if early_stopping is None:
            early_stopping = n_train >= 40

        common = dict(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=float(self.alpha),
            batch_size=batch,
            learning_rate="adaptive",
            learning_rate_init=float(lr),
            max_iter=int(n_epochs),
            random_state=int(random_state),
            early_stopping=early_stopping,
            validation_fraction=0.15 if early_stopping else 0.0,
            n_iter_no_change=15,
            tol=1e-4,
        )
        if is_classifier:
            return MLPClassifier(**common)
        return MLPRegressor(**common)

    # ------------------------------------------------------------------
    # Stage 3 — Inductive / stateful interface (fit_model / partial_fit_model)
    # ------------------------------------------------------------------

    def fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        is_classifier: bool,
        n_epochs: int = 300,
        lr: float = 0.001,
        random_state: int = 42,
        encoded_feature_names: list[str] | None = None,
        encoded_feature_kinds: dict[str, str] | None = None,
        label_enc: Any = None,
    ) -> "NeuralPhysicsEngine":
        """Fit and store the attention block + MLP for inductive (stateful) use.

        After calling this, ``predict_model`` and ``partial_fit_model`` are
        available without passing training data again.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, n_features)
        y_train : ndarray, shape (n_train,)
        is_classifier : bool
        n_epochs : int
        lr : float
            Adam learning_rate_init.
        random_state : int
        encoded_feature_names : list[str] or None
            Column names corresponding to X_train features (for encoding
            alignment in partial_fit).
        encoded_feature_kinds : dict or None
        label_enc : LabelEncoder or None
            Fitted LabelEncoder used to encode y_train.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for NeuralPhysicsEngine")

        n_train = X_train.shape[0]
        # Stage 2 — fit attention block
        attn = _FeatureAttentionBlock(max_attend_features=self.max_attend_features)
        X_att = attn.fit_transform(X_train)
        X_aug = np.hstack([X_train, X_att])

        # Stage 1 — fit MLP
        mlp = self._make_mlp(
            is_classifier=is_classifier,
            n_epochs=n_epochs,
            lr=lr,
            random_state=random_state,
            n_train=n_train,
            early_stopping=False,  # must be False to allow partial_fit later
        )
        mlp.fit(X_aug, y_train)

        # Store fitted components
        self.attn_: _FeatureAttentionBlock = attn
        self.mlp_: Any = mlp
        self.is_classifier_: bool = is_classifier
        self.label_enc_: Any = label_enc
        self.n_input_features_: int = X_train.shape[1]
        self.encoded_feature_names_: list[str] = list(encoded_feature_names or [])
        self.encoded_feature_kinds_: dict[str, str] = dict(encoded_feature_kinds or {})

        # EWC anchor — store weights + Fisher approximation
        self._theta_old_: np.ndarray | None = self._get_flat_weights()
        self._fisher_: np.ndarray | None = self._compute_fisher()
        return self

    def predict_model(self, X_new: np.ndarray) -> np.ndarray:
        """Predict using the stored MLP (inductive mode, no re-training).

        Parameters
        ----------
        X_new : ndarray, shape (n_samples, n_input_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Raw integer labels (classification) or floats (regression).
        """
        if not hasattr(self, "mlp_"):
            raise RuntimeError("fit_model() must be called before predict_model()")
        X_att = self.attn_.transform(X_new)
        X_aug = np.hstack([X_new, X_att])
        return self.mlp_.predict(X_aug)

    def predict_proba_model(self, X_new: np.ndarray) -> np.ndarray:
        """Return class probability estimates (classification only).

        Parameters
        ----------
        X_new : ndarray, shape (n_samples, n_input_features)

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Probability of each class, columns ordered by
            ``self.mlp_.classes_``.

        Raises
        ------
        RuntimeError
            If ``fit_model`` has not been called.
        ValueError
            If the engine was fitted on a regression task.
        """
        if not hasattr(self, "mlp_"):
            raise RuntimeError("fit_model() must be called before predict_proba_model()")
        if not getattr(self, "is_classifier_", False):
            raise ValueError("predict_proba_model is only available for classifiers")
        X_att = self.attn_.transform(X_new)
        X_aug = np.hstack([X_new, X_att])
        return self.mlp_.predict_proba(X_aug)

    def partial_fit_model(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        X_replay: np.ndarray | None = None,
        y_replay: np.ndarray | None = None,
        *,
        ewc_lambda: float = 0.4,
    ) -> "NeuralPhysicsEngine":
        """Incrementally update the stored MLP with new data and replay buffer.

        Uses sklearn ``partial_fit`` (SGD step) on the combined
        new + replay batch, then applies a post-hoc EWC weight-consolidation
        step to prevent catastrophic forgetting.

        Parameters
        ----------
        X_new : ndarray, shape (n_new, n_input_features)
        y_new : ndarray, shape (n_new,)
        X_replay : ndarray or None
            Historic rows drawn from the caller's replay buffer.
        y_replay : ndarray or None
        ewc_lambda : float, default 0.4
            Consolidation strength.  0 = no EWC, 1 = weights frozen.
        """
        if not hasattr(self, "mlp_"):
            raise RuntimeError("fit_model() must be called before partial_fit_model()")

        # Mix new data with replay
        if X_replay is not None and len(X_replay) > 0:
            X_all = np.vstack([X_new, X_replay])
            y_all = np.concatenate([y_new, y_replay])
        else:
            X_all, y_all = X_new, y_new

        X_att = self.attn_.transform(X_all)
        X_aug = np.hstack([X_all, X_att])

        if self.is_classifier_:
            all_classes = np.unique(
                np.concatenate([y_all, self.mlp_.classes_])
            )
            self.mlp_.partial_fit(X_aug, y_all, classes=all_classes)
        else:
            self.mlp_.partial_fit(X_aug, y_all)

        # EWC consolidation
        if ewc_lambda > 0.0:
            self._apply_ewc(ewc_lambda)
        return self

    # ── EWC helpers ────────────────────────────────────────────────────

    def _get_flat_weights(self) -> np.ndarray | None:
        """Flatten all MLP weight matrices and biases into a 1-D vector."""
        if not hasattr(self, "mlp_"):
            return None
        parts: list[np.ndarray] = []
        for W in self.mlp_.coefs_:
            parts.append(W.ravel())
        for b in self.mlp_.intercepts_:
            parts.append(b.ravel())
        return np.concatenate(parts)

    def _set_flat_weights(self, flat: np.ndarray) -> None:
        """Write a flat weight vector back into the MLP coefs + intercepts."""
        idx = 0
        for W in self.mlp_.coefs_:
            n = W.size
            W[:] = flat[idx : idx + n].reshape(W.shape)
            idx += n
        for b in self.mlp_.intercepts_:
            n = b.size
            b[:] = flat[idx : idx + n].reshape(b.shape)
            idx += n

    def _compute_fisher(self) -> np.ndarray | None:
        """Approximate diagonal Fisher information as normalised |θ|.

        A full empirical Fisher requires per-sample gradients which sklearn
        does not expose.  We use |θ| / max(|θ|) as a robust proxy: large
        weights are assumed to encode more knowledge and should be protected
        more strongly during consolidation.
        """
        theta = self._get_flat_weights()
        if theta is None:
            return None
        fisher = np.abs(theta) / (np.abs(theta).max() + 1e-8)
        return fisher

    def _apply_ewc(self, ewc_lambda: float) -> None:
        """Pull current weights toward the EWC anchor θ_old.

        The update rule is:
            θ_new = θ_current − λ · F · (θ_current − θ_old)

        where F is the diagonal Fisher approximation stored at ``fit_model``
        time.  This is equivalent to a per-parameter ridge penalty toward the
        anchor, applied once per ``partial_fit`` call.
        """
        if self._theta_old_ is None or self._fisher_ is None:
            return
        theta_cur = self._get_flat_weights()
        if theta_cur is None:
            return
        delta = theta_cur - self._theta_old_
        theta_final = theta_cur - float(ewc_lambda) * self._fisher_ * delta
        self._set_flat_weights(theta_final)

    def encode_aligned(
        self,
        df: pd.DataFrame,
        target_col: str = "__target__",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a new DataFrame using the stored feature schema.

        Aligns one-hot columns to those seen during ``fit_model``, filling
        missing columns with 0 and dropping unseen columns.

        Returns
        -------
        X : ndarray, shape (n, n_input_features_)
        y : ndarray, shape (n,)
        """
        X_raw, y_raw, feat_names, feat_kinds, _ = _encode_dataframe(df, target_col)

        if not self.encoded_feature_names_:
            return X_raw, y_raw

        name_to_col: dict[str, int] = {n: i for i, n in enumerate(feat_names)}
        n = X_raw.shape[0]
        d = len(self.encoded_feature_names_)
        X_aligned = np.zeros((n, d), dtype=float)
        for j, name in enumerate(self.encoded_feature_names_):
            if name in name_to_col:
                X_aligned[:, j] = X_raw[:, name_to_col[name]]
        return X_aligned, y_raw

    # ── Stage 6 — Save / load ──────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the fitted engine to disk using joblib.

        Parameters
        ----------
        path : str or Path
            File path (e.g. ``"my_engine.pkl"``).
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for save/load") from exc
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPhysicsEngine":
        """Load a previously saved engine.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        NeuralPhysicsEngine
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for save/load") from exc
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected NeuralPhysicsEngine, got {type(obj)}")
        return obj

    # ── Stage 6 — Pretraining / transfer learning ──────────────────────

    @classmethod
    def pretrain(
        cls,
        datasets: list[pd.DataFrame],
        target_col: str,
        *,
        n_cycles: int = 20,
        cycle_learning_rate: float = 0.18,
        random_seed: int = 42,
        ewc_lambda: float = 0.4,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        max_attend_features: int = 60,
        alpha: float = 1e-4,
    ) -> "NeuralPhysicsEngine":
        """Train sequentially across multiple datasets with EWC regularisation.

        Implements curriculum learning: each dataset in ``datasets`` is used
        in order.  After the first dataset the engine uses ``partial_fit_model``
        with EWC consolidation so that knowledge from earlier tasks is
        retained.

        Parameters
        ----------
        datasets : list[DataFrame]
            Each DataFrame must contain ``target_col``.
        target_col : str
        n_cycles : int
            Epochs per dataset (first fit uses ``n_cycles * 10``).
        cycle_learning_rate : float
        random_seed : int
        ewc_lambda : float
            EWC consolidation strength between tasks (0 = disabled).
        hidden_layer_sizes, max_attend_features, alpha
            Architecture hyperparameters forwarded to the engine.

        Returns
        -------
        NeuralPhysicsEngine
            Fitted engine ready for ``predict_model`` or further fine-tuning.
        """
        engine = cls(
            hidden_layer_sizes=hidden_layer_sizes,
            max_attend_features=max_attend_features,
            alpha=alpha,
        )
        n_epochs_first = int(np.clip(n_cycles * 10, 100, 2000))
        n_epochs_cont = max(50, n_epochs_first // 4)
        lr = float(np.clip(cycle_learning_rate * 0.01, 1e-4, 0.01))

        for task_idx, df in enumerate(datasets):
            if target_col not in df.columns:
                continue
            df_reset = df.reset_index(drop=True)
            X, y, feat_names, feat_kinds, label_enc = _encode_dataframe(df_reset, target_col)
            if len(X) < 4:
                continue
            target_kind = infer_target_kind(df_reset[target_col])
            is_clf = target_kind == "categorical"

            if task_idx == 0:
                engine.fit_model(
                    X, y,
                    is_classifier=is_clf,
                    n_epochs=n_epochs_first,
                    lr=lr,
                    random_state=random_seed,
                    encoded_feature_names=feat_names,
                    encoded_feature_kinds=feat_kinds,
                    label_enc=label_enc,
                )
            else:
                X_aligned, y_aligned = engine.encode_aligned(df_reset, target_col)
                engine.partial_fit_model(X_aligned, y_aligned, ewc_lambda=ewc_lambda)
                # Refresh EWC anchor after each task
                engine._theta_old_ = engine._get_flat_weights()
                engine._fisher_ = engine._compute_fisher()
        return engine

    # ------------------------------------------------------------------
    # Public interface — matches run_physics_prediction call signature
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        plane: PhysicsPlane = PhysicsPlane.solid,
        runtime_state: PredictorRuntimeState | None = None,
        train_fraction: float = 0.8,
        random_seed: int = 42,
        n_cycles: int = 30,
        cycle_learning_rate: float = 0.18,
        return_predictions: bool = True,
        explicit_train_mask: np.ndarray | None = None,
        max_preview_rows: int = 25,
        # Accept and ignore physics-specific kwargs for API compatibility
        **_kwargs: Any,
    ) -> PredictionResult | None:
        """Run neural prediction and return a ``PredictionResult``.

        Parameters
        ----------
        df : DataFrame
            Combined train + test rows (must include ``target_col``).
        target_col : str
            Name of the target column.
        plane : PhysicsPlane
            Medium preset (passed through to PredictionResult for API parity).
        runtime_state : PredictorRuntimeState or None
            If provided, updated with result metrics (homeostasis wiring).
        train_fraction : float
            Fraction used as train when ``explicit_train_mask`` is None.
        random_seed : int
            Random seed for reproducibility.
        n_cycles : int
            Maps to ``n_epochs = n_cycles * 10`` (clamped to [100, 2000]).
        cycle_learning_rate : float
            Maps to Adam ``learning_rate_init = cycle_learning_rate * 0.01``.
        return_predictions : bool
            When True, populate ``test_predicted`` in the result.
        explicit_train_mask : ndarray[bool] or None
            Boolean mask of length ``len(df)``; True = training row.

        Returns
        -------
        PredictionResult or None
            Full result object compatible with the physics engine output.
        """
        if not _SKLEARN_AVAILABLE:
            return None

        df = df.reset_index(drop=True)
        n_rows = len(df)
        if n_rows < 4:
            return None

        # ── Train / test split ─────────────────────────────────────────
        if explicit_train_mask is not None:
            train_mask = np.asarray(explicit_train_mask, dtype=bool)
            if len(train_mask) != n_rows:
                train_mask = np.ones(n_rows, dtype=bool)
                train_mask[int(n_rows * train_fraction):] = False
        else:
            rng = np.random.default_rng(int(random_seed))
            idx = rng.permutation(n_rows)
            n_train_rows = max(2, int(n_rows * train_fraction))
            train_mask = np.zeros(n_rows, dtype=bool)
            train_mask[idx[:n_train_rows]] = True

        test_mask = ~train_mask
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        if n_train < 2:
            return None

        # ── Encode to numeric matrices ─────────────────────────────────
        X, y, feature_names, feature_kinds, label_enc = _encode_dataframe(df, target_col)
        n_features = len(feature_names)
        target_kind = infer_target_kind(df[target_col])
        is_classifier = target_kind == "categorical"

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # ── Stage 2: Feature Attention Block ───────────────────────────
        attn = _FeatureAttentionBlock(max_attend_features=self.max_attend_features)
        X_train_att = attn.fit_transform(X_train)   # (n_train, d_eff)
        X_test_att = attn.transform(X_test)         # (n_test,  d_eff)

        attn_importance = (
            attn.feature_importance_
            if attn.feature_importance_ is not None
            else np.ones(n_features) / max(n_features, 1)
        )

        # Pseudo-residual: concatenate original features with attended
        # The MLP learns to blend raw and attended representations.
        X_train_aug = np.hstack([X_train, X_train_att])  # (n_train, d + d_eff)
        X_test_aug = np.hstack([X_test, X_test_att])      # (n_test,  d + d_eff)

        # ── Stage 1: MLP Backbone ──────────────────────────────────────
        n_epochs = int(np.clip(int(n_cycles) * 10, 100, 2000))
        lr = float(cycle_learning_rate) * 0.01
        lr = float(np.clip(lr, 1e-4, 0.01))

        mlp = self._make_mlp(
            is_classifier=is_classifier,
            n_epochs=n_epochs,
            lr=lr,
            random_state=int(random_seed),
            n_train=n_train,
        )

        try:
            mlp.fit(X_train_aug, y_train)
        except Exception:
            return None

        # ── Predictions & metrics ──────────────────────────────────────
        try:
            y_pred_train = mlp.predict(X_train_aug)
            y_pred_test = mlp.predict(X_test_aug) if n_test > 0 else np.array([])
        except Exception:
            return None

        # Decode labels back to original space
        test_predicted_raw: list[Any]
        test_actual_raw: list[Any]
        if is_classifier and label_enc is not None:
            y_pred_test_dec = (
                label_enc.inverse_transform(y_pred_test.astype(int))
                if n_test > 0 else np.array([])
            )
            y_test_dec = label_enc.inverse_transform(y_test.astype(int))
            y_pred_train_dec = label_enc.inverse_transform(y_pred_train.astype(int))
            test_predicted_raw = list(y_pred_test_dec)
            test_actual_raw = list(y_test_dec)
            accuracy = float(np.mean(y_pred_test_dec == y_test_dec)) if n_test > 0 else None
            classes, counts = np.unique(y_train, return_counts=True)
            baseline_class = classes[np.argmax(counts)]
            baseline_accuracy = float(
                np.mean(
                    label_enc.inverse_transform(
                        np.full(n_test, baseline_class, dtype=int)
                    ) == y_test_dec
                )
            ) if n_test > 0 else None
            mae = rmse = baseline_mae = baseline_rmse = None
        else:
            test_predicted_raw = [float(v) for v in y_pred_test] if n_test > 0 else []
            test_actual_raw = [float(v) for v in y_test] if n_test > 0 else []
            if n_test > 0:
                errs = y_test.astype(float) - y_pred_test.astype(float)
                mae = float(np.mean(np.abs(errs)))
                rmse = float(np.sqrt(np.mean(errs ** 2)))
                base_pred = float(np.mean(y_train.astype(float)))
                base_errs = y_test.astype(float) - base_pred
                baseline_mae = float(np.mean(np.abs(base_errs)))
                baseline_rmse = float(np.sqrt(np.mean(base_errs ** 2)))
            else:
                mae = rmse = baseline_mae = baseline_rmse = None
            accuracy = baseline_accuracy = None

        # Best-cycle proxy: use n_iter_ from sklearn MLP
        best_cycle = int(getattr(mlp, "n_iter_", n_epochs))
        if mae is not None and baseline_mae is not None and baseline_mae > 0:
            best_lift = float(1.0 - mae / baseline_mae)
        elif accuracy is not None and baseline_accuracy is not None and baseline_accuracy > 0:
            best_lift = float(accuracy - baseline_accuracy)
        else:
            best_lift = 0.0

        metrics = PredictionMetrics(
            target_kind=target_kind,
            n_rows=n_rows,
            n_train=n_train,
            n_test=n_test,
            train_fraction=float(n_train) / max(n_rows, 1),
            random_seed=int(random_seed),
            n_features_used=n_features,
            mae=mae,
            rmse=rmse,
            accuracy=accuracy,
            baseline_accuracy=baseline_accuracy,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            best_cycle=best_cycle,
            best_lift=best_lift,
            # Gel-specific fields set to proxy values for API parity
            buffer_ionization="parametric",
            buffer_normality_p=None,
            gel_band_sharpness=float(np.clip(best_lift, 0.0, 1.0)) if best_lift else None,
            gel_smearing=float(np.clip(1.0 - abs(best_lift), 0.0, 1.0)) if best_lift is not None else None,
            gel_ghost_band_rate=0.0,
            gel_confidence_mean=None,
            gel_confidence_std=None,
        )

        # ── Feature importance (combining attention + MLP weights) ─────
        mlp_imp = _mlp_feature_importance(mlp, n_features)
        weights = _build_weight_infos(feature_names, feature_kinds, attn_importance, mlp_imp)

        # ── Migration map ──────────────────────────────────────────────
        migration_map = _build_migration_infos(
            feature_names, feature_kinds, X_train, attn_importance
        )

        # ── Iteration gains (from MLP loss curve proxy) ────────────────
        iteration_gains: list[IterationInfo] = []
        loss_curve = getattr(mlp, "loss_curve_", None)
        if loss_curve:
            n_reported = min(len(loss_curve), 20)
            step = max(1, len(loss_curve) // n_reported)
            for k in range(0, len(loss_curve), step):
                iteration_gains.append(
                    IterationInfo(
                        cycle=k + 1,
                        test_mae=float(loss_curve[k]) if not is_classifier else None,
                        test_rmse=None,
                        test_accuracy=None,
                        lift_over_baseline=None,
                    )
                )
        else:
            iteration_gains = [IterationInfo(cycle=best_cycle, test_mae=mae)]

        # ── Preview rows ───────────────────────────────────────────────
        preview: list[dict[str, Any]] = []
        if n_test > 0:
            test_indices = np.where(test_mask)[0]
            for j in range(min(int(max_preview_rows), n_test)):
                row: dict[str, Any] = {
                    "row": int(test_indices[j]),
                    "predicted": test_predicted_raw[j] if j < len(test_predicted_raw) else None,
                    "actual": test_actual_raw[j] if j < len(test_actual_raw) else None,
                }
                preview.append(row)

        # ── Equilibrium zones (attention-based feature clusters) ───────
        equilibrium_zones = _build_equilibrium_zones(
            feature_names, attn_importance, attn.attn_matrix_
        )

        # ── Assemble result ────────────────────────────────────────────
        test_row_indices = list(np.where(test_mask)[0]) if n_test > 0 else None
        result = PredictionResult(
            target=target_col,
            target_kind=target_kind,
            plane=plane,
            weights=weights,
            migration_map=migration_map,
            bonding_map=[],
            iteration_gains=iteration_gains,
            equilibrium_zones=equilibrium_zones,
            metrics=metrics,
            preview_rows=preview,
            diagnostics={"backend": "neural", "n_epochs_run": best_cycle},
            test_row_indices=test_row_indices,
            test_actual=test_actual_raw if return_predictions else None,
            test_predicted=test_predicted_raw if return_predictions else None,
        )

        # ── Homeostasis wiring (Stage 1 step 4) ───────────────────────
        # PredictorRuntimeState.homeostasis_score maps to optimizer epoch ratio
        if runtime_state is not None:
            try:
                update_predictor_state_from_result(runtime_state, result)
                # Wire Adam epoch to cycle_index for learning-rate warmup semantics
                runtime_state.metadata["neural_epoch"] = best_cycle
                runtime_state.metadata["neural_lr_init"] = lr
            except Exception:
                pass

        return result


def _build_equilibrium_zones(
    feature_names: list[str],
    attn_importance: np.ndarray,
    attn_matrix: np.ndarray | None,
    n_zones: int = 5,
) -> list[EquilibriumZone]:
    """Cluster features into equilibrium zones by attention importance quantile."""
    d = len(feature_names)
    if d == 0:
        return []

    imp = attn_importance[:d]
    if len(imp) < d:
        imp = np.pad(imp, (0, d - len(imp)))

    # Divide into n_zones quantile buckets
    zone_assignments = np.minimum(
        (imp * n_zones).astype(int), n_zones - 1
    )

    zones: list[EquilibriumZone] = []
    for zone_id in range(n_zones):
        mask = zone_assignments == zone_id
        feats = [feature_names[i] for i in range(d) if mask[i]]
        if not feats:
            continue
        avg_pi = float(np.mean(imp[mask]))
        # avg_momentum: proxy from attention matrix column sums
        avg_mom = 0.0
        if attn_matrix is not None and attn_matrix.shape[1] >= d:
            indices = [i for i in range(d) if mask[i] and i < attn_matrix.shape[1]]
            if indices:
                avg_mom = float(attn_matrix[:, indices].mean())
        zones.append(
            EquilibriumZone(
                zone_id=zone_id,
                features=feats,
                avg_pI=avg_pi,
                avg_momentum=avg_mom,
                strength=float(len(feats)) / max(d, 1),
            )
        )
    return zones


# ---------------------------------------------------------------------------
# Functional interface — mirrors run_physics_prediction
# ---------------------------------------------------------------------------

def run_neural_prediction(
    df: pd.DataFrame,
    *,
    target_col: str,
    plane: PhysicsPlane = PhysicsPlane.solid,
    runtime_state: PredictorRuntimeState | None = None,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    n_cycles: int = 30,
    cycle_learning_rate: float = 0.18,
    return_predictions: bool = True,
    explicit_train_mask: np.ndarray | None = None,
    max_preview_rows: int = 25,
    hidden_layer_sizes: tuple[int, ...] = (256, 128),
    max_attend_features: int = 60,
    alpha: float = 1e-4,
    **kwargs: Any,
) -> PredictionResult | None:
    """Functional interface for the neural engine — mirrors ``run_physics_prediction``.

    Parameters
    ----------
    df : DataFrame
        Combined train + test rows including ``target_col``.
    target_col : str
    plane, runtime_state, train_fraction, random_seed, n_cycles,
    cycle_learning_rate, return_predictions, explicit_train_mask,
    max_preview_rows : see ``NeuralPhysicsEngine.run`` for documentation.
    hidden_layer_sizes : tuple[int, ...], default (256, 128)
    max_attend_features : int, default 60
    alpha : float, default 1e-4

    Returns
    -------
    PredictionResult or None
    """
    engine = NeuralPhysicsEngine(
        hidden_layer_sizes=hidden_layer_sizes,
        max_attend_features=max_attend_features,
        alpha=alpha,
    )
    return engine.run(
        df,
        target_col=target_col,
        plane=plane,
        runtime_state=runtime_state,
        train_fraction=train_fraction,
        random_seed=random_seed,
        n_cycles=n_cycles,
        cycle_learning_rate=cycle_learning_rate,
        return_predictions=return_predictions,
        explicit_train_mask=explicit_train_mask,
        max_preview_rows=max_preview_rows,
        **kwargs,
    )
