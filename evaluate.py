"""PhysML — Comprehensive benchmark evaluation.

Compares the PhysML physics predictor against standard ML baselines on
multiple scikit-learn datasets for both classification and regression tasks.

Baseline models included
------------------------
* Random Forest  (RF)
* Extra Trees    (ET)
* Gradient Boosting (GB)
* Histogram Gradient Boosting (HGB)  — often the strongest sklearn tree
* Multi-layer Perceptron (MLP / neural net)
* K-Nearest Neighbours (KNN)
* Support Vector Machine (SVM)
* Logistic Regression / Ridge Regression (linear baseline)
* AdaBoost
* PhysML (physics electrophoresis engine)

Datasets
--------
Classification : iris, breast_cancer, wine, digits (subset)
Regression     : diabetes, california_housing (subset), linnerud

Usage
-----
    python evaluate.py
    python evaluate.py --output results.json
    python evaluate.py --tasks classification
    python evaluate.py --tasks regression
    python evaluate.py --quick          # fewer cycles / smaller datasets
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# scikit-learn imports ──────────────────────────────────────────────────
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    fetch_california_housing,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# PhysML imports ─────────────────────────────────────────────────────────
from physml import PhysicsPlane, run_physics_prediction
from physml.estimator import PhysicsPredictor
from physml.agent import DataStream, PhysicsAgent


# ── Constants ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_SPLITS = 5          # cross-validation folds
N_ESTIMATORS = 200    # tree ensemble size


# ── Data helpers ─────────────────────────────────────────────────────────

@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    task: str        # "classification" | "regression"
    target_name: str


def _load_classification_datasets(quick: bool = False) -> list[Dataset]:
    """Load standard classification benchmark datasets."""
    datasets: list[Dataset] = []

    iris = load_iris(as_frame=False)
    datasets.append(Dataset("iris", iris.data, iris.target, "classification", "species"))

    bc = load_breast_cancer(as_frame=False)
    datasets.append(Dataset("breast_cancer", bc.data, bc.target, "classification", "malignant"))

    wine = load_wine(as_frame=False)
    datasets.append(Dataset("wine", wine.data, wine.target, "classification", "class"))

    return datasets


def _load_regression_datasets(quick: bool = False) -> list[Dataset]:
    """Load standard regression benchmark datasets."""
    datasets: list[Dataset] = []

    dia = load_diabetes(as_frame=False)
    datasets.append(Dataset("diabetes", dia.data, dia.target, "regression", "progression"))

    try:
        cal = fetch_california_housing(as_frame=False)
        # Subsample to keep evaluation fast
        n = 2000 if quick else 8000
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(cal.target), size=min(n, len(cal.target)), replace=False)
        datasets.append(Dataset("california_housing", cal.data[idx], cal.target[idx], "regression", "median_house_value"))
    except Exception:
        pass

    return datasets


# ── Model factories ───────────────────────────────────────────────────────

def _classification_baselines() -> list[tuple[str, Any]]:
    rs = RANDOM_SEED
    est = N_ESTIMATORS
    return [
        ("RandomForest", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", RandomForestClassifier(n_estimators=est, random_state=rs, n_jobs=-1)),
        ])),
        ("ExtraTrees", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", ExtraTreesClassifier(n_estimators=est, random_state=rs, n_jobs=-1)),
        ])),
        ("GradientBoosting", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", GradientBoostingClassifier(n_estimators=est, random_state=rs)),
        ])),
        ("HistGradientBoosting", Pipeline([
            ("mdl", HistGradientBoostingClassifier(max_iter=est, random_state=rs)),
        ])),
        ("MLP_NeuralNet", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=400,
                random_state=rs,
                early_stopping=True,
                n_iter_no_change=20,
            )),
        ])),
        ("KNN", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ])),
        ("SVM", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=rs)),
        ])),
        ("LogisticRegression", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", LogisticRegression(max_iter=2000, random_state=rs, n_jobs=-1)),
        ])),
        ("AdaBoost", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", AdaBoostClassifier(n_estimators=100, random_state=rs)),
        ])),
    ]


def _regression_baselines() -> list[tuple[str, Any]]:
    rs = RANDOM_SEED
    est = N_ESTIMATORS
    return [
        ("RandomForest", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", RandomForestRegressor(n_estimators=est, random_state=rs, n_jobs=-1)),
        ])),
        ("ExtraTrees", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", ExtraTreesRegressor(n_estimators=est, random_state=rs, n_jobs=-1)),
        ])),
        ("GradientBoosting", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", GradientBoostingRegressor(n_estimators=est, random_state=rs)),
        ])),
        ("HistGradientBoosting", Pipeline([
            ("mdl", HistGradientBoostingRegressor(max_iter=est, random_state=rs)),
        ])),
        ("MLP_NeuralNet", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                random_state=rs,
                early_stopping=True,
                n_iter_no_change=20,
            )),
        ])),
        ("KNN", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", KNeighborsRegressor(n_neighbors=7, n_jobs=-1)),
        ])),
        ("SVR", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", SVR(kernel="rbf", C=10.0, gamma="scale")),
        ])),
        ("Ridge", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", Ridge(alpha=1.0)),
        ])),
        ("AdaBoost", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", AdaBoostRegressor(n_estimators=100, random_state=rs)),
        ])),
    ]


# ── PhysML evaluation helpers ─────────────────────────────────────────────

def _physml_cv_classification(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    n_cycles: int = 20,
    plane: PhysicsPlane = PhysicsPlane.liquid,
    random_seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Run PhysML on a classification dataset using k-fold CV."""
    fold_acc: list[float] = []
    fold_f1: list[float] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build combined df: train rows first, then test rows
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        df = pd.DataFrame(X_all, columns=feature_names)
        df["__target__"] = y_all

        n_train = len(train_idx)
        explicit_mask = np.zeros(len(train_idx) + len(test_idx), dtype=bool)
        explicit_mask[:n_train] = True

        try:
            result = run_physics_prediction(
                df,
                target_col="__target__",
                plane=plane,
                n_cycles=n_cycles,
                random_seed=random_seed,
                return_predictions=True,
                enable_isotopes=True,
                explicit_train_mask=explicit_mask,
            )
        except Exception:
            result = None

        if result is not None and result.test_predicted and result.test_actual:
            y_pred = np.array(result.test_predicted, dtype=str)
            y_true = np.array(result.test_actual, dtype=str)
            if len(y_pred) == len(y_true):
                fold_acc.append(float(accuracy_score(y_true, y_pred)))
                fold_f1.append(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)))
            else:
                fold_acc.append(float(result.metrics.accuracy or 0.0))
                fold_f1.append(0.0)
        elif result is not None and result.metrics.accuracy is not None:
            fold_acc.append(float(result.metrics.accuracy))
            fold_f1.append(0.0)
        else:
            fold_acc.append(0.0)
            fold_f1.append(0.0)

    return {
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "f1_weighted_mean": float(np.mean(fold_f1)),
        "f1_weighted_std": float(np.std(fold_f1)),
    }


def _physml_cv_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    n_cycles: int = 20,
    plane: PhysicsPlane = PhysicsPlane.solid,
    random_seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Run PhysML on a regression dataset using k-fold CV."""
    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_r2: list[float] = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        df = pd.DataFrame(X_all, columns=feature_names)
        df["__target__"] = y_all

        n_train = len(train_idx)
        explicit_mask = np.zeros(n_train + len(test_idx), dtype=bool)
        explicit_mask[:n_train] = True

        try:
            result = run_physics_prediction(
                df,
                target_col="__target__",
                plane=plane,
                n_cycles=n_cycles,
                random_seed=random_seed,
                return_predictions=True,
                enable_isotopes=True,
                explicit_train_mask=explicit_mask,
            )
        except Exception:
            result = None

        if result is not None and result.test_predicted and result.test_actual:
            try:
                y_pred = np.array(result.test_predicted, dtype=float)
                y_true = np.array(result.test_actual, dtype=float)
                if len(y_pred) == len(y_true) and len(y_pred) > 0:
                    fold_mae.append(float(mean_absolute_error(y_true, y_pred)))
                    fold_rmse.append(float(math.sqrt(mean_squared_error(y_true, y_pred))))
                    fold_r2.append(float(r2_score(y_true, y_pred)))
                else:
                    raise ValueError("length mismatch")
            except Exception:
                m = result.metrics
                fold_mae.append(float(m.mae or 0.0))
                fold_rmse.append(float(m.rmse or 0.0))
                fold_r2.append(0.0)
        elif result is not None and result.metrics.mae is not None:
            m = result.metrics
            fold_mae.append(float(m.mae or 0.0))
            fold_rmse.append(float(m.rmse or 0.0))
            fold_r2.append(0.0)
        else:
            fold_mae.append(float("inf"))
            fold_rmse.append(float("inf"))
            fold_r2.append(0.0)

    return {
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_std": float(np.std(fold_rmse)),
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2)),
    }


# ── Sklearn cross-validation ──────────────────────────────────────────────

def _sklearn_cv_classification(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
) -> dict[str, float]:
    fold_acc: list[float] = []
    fold_f1: list[float] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            fold_acc.append(float(accuracy_score(y_te, y_pred)))
            fold_f1.append(float(f1_score(y_te, y_pred, average="weighted", zero_division=0)))
        except Exception:
            fold_acc.append(0.0)
            fold_f1.append(0.0)
    return {
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "f1_weighted_mean": float(np.mean(fold_f1)),
        "f1_weighted_std": float(np.std(fold_f1)),
    }


def _sklearn_cv_regression(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
) -> dict[str, float]:
    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_r2: list[float] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            fold_mae.append(float(mean_absolute_error(y_te, y_pred)))
            fold_rmse.append(float(math.sqrt(mean_squared_error(y_te, y_pred))))
            fold_r2.append(float(r2_score(y_te, y_pred)))
        except Exception:
            fold_mae.append(float("inf"))
            fold_rmse.append(float("inf"))
            fold_r2.append(0.0)
    return {
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_std": float(np.std(fold_rmse)),
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2)),
    }


# ── Display ───────────────────────────────────────────────────────────────

def _print_classification_table(dataset_name: str, rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: float(r.get("accuracy_mean", 0.0)), reverse=True)
    header = f"\n{'='*70}\nDataset: {dataset_name}  (Classification, {N_SPLITS}-fold CV)\n{'='*70}"
    print(header)
    print(f"{'Model':<28} {'Accuracy':>10} {'±':>4} {'F1 (weighted)':>14} {'±':>4} {'Time(s)':>8}")
    print("-" * 70)
    for r in rows_sorted:
        marker = " ◄" if r["model"] == "PhysML" else ""
        print(
            f"{r['model']:<28} {r['accuracy_mean']:>10.4f} {r['accuracy_std']:>4.3f} "
            f"{r['f1_weighted_mean']:>14.4f} {r['f1_weighted_std']:>4.3f} "
            f"{r.get('elapsed_s', 0.0):>8.2f}{marker}"
        )


def _print_regression_table(dataset_name: str, rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: float(r.get("r2_mean", -1e9)), reverse=True)
    header = f"\n{'='*70}\nDataset: {dataset_name}  (Regression, {N_SPLITS}-fold CV)\n{'='*70}"
    print(header)
    print(f"{'Model':<28} {'R²':>8} {'±':>4} {'RMSE':>10} {'±':>6} {'MAE':>10} {'Time(s)':>8}")
    print("-" * 70)
    for r in rows_sorted:
        marker = " ◄" if r["model"] == "PhysML" else ""
        print(
            f"{r['model']:<28} {r['r2_mean']:>8.4f} {r['r2_std']:>4.3f} "
            f"{r['rmse_mean']:>10.4f} {r['rmse_std']:>6.3f} "
            f"{r['mae_mean']:>10.4f} "
            f"{r.get('elapsed_s', 0.0):>8.2f}{marker}"
        )


# ── Main evaluation logic ─────────────────────────────────────────────────

def run_classification_benchmark(quick: bool = False) -> list[dict[str, Any]]:
    datasets = _load_classification_datasets(quick=quick)
    baselines = _classification_baselines()
    n_cycles_phys = 15 if quick else 25
    all_results: list[dict[str, Any]] = []

    for ds in datasets:
        print(f"\n[Classification] {ds.name}  shape={ds.X.shape}  classes={len(np.unique(ds.y))}")
        rows: list[dict[str, Any]] = []

        # PhysML
        t0 = time.perf_counter()
        phys_metrics = _physml_cv_classification(
            ds.X, ds.y,
            n_splits=N_SPLITS,
            n_cycles=n_cycles_phys,
            plane=PhysicsPlane.liquid,
            random_seed=RANDOM_SEED,
        )
        elapsed = time.perf_counter() - t0
        row: dict[str, Any] = {"model": "PhysML", "elapsed_s": round(elapsed, 3), **phys_metrics}
        rows.append(row)
        print(f"  PhysML done  ({elapsed:.1f}s)  acc={phys_metrics['accuracy_mean']:.4f}")

        # Sklearn baselines
        for name, model in baselines:
            t0 = time.perf_counter()
            metrics = _sklearn_cv_classification(model, ds.X, ds.y, n_splits=N_SPLITS)
            elapsed = time.perf_counter() - t0
            row = {"model": name, "elapsed_s": round(elapsed, 3), **metrics}
            rows.append(row)
            print(f"  {name:<28} ({elapsed:.1f}s)  acc={metrics['accuracy_mean']:.4f}")

        _print_classification_table(ds.name, rows)
        all_results.append({"dataset": ds.name, "task": "classification", "results": rows})

    return all_results


def run_regression_benchmark(quick: bool = False) -> list[dict[str, Any]]:
    datasets = _load_regression_datasets(quick=quick)
    baselines = _regression_baselines()
    n_cycles_phys = 15 if quick else 25
    all_results: list[dict[str, Any]] = []

    for ds in datasets:
        print(f"\n[Regression] {ds.name}  shape={ds.X.shape}")
        rows: list[dict[str, Any]] = []

        # PhysML
        t0 = time.perf_counter()
        phys_metrics = _physml_cv_regression(
            ds.X, ds.y,
            n_splits=N_SPLITS,
            n_cycles=n_cycles_phys,
            plane=PhysicsPlane.solid,
            random_seed=RANDOM_SEED,
        )
        elapsed = time.perf_counter() - t0
        row = {"model": "PhysML", "elapsed_s": round(elapsed, 3), **phys_metrics}
        rows.append(row)
        print(f"  PhysML done  ({elapsed:.1f}s)  R²={phys_metrics['r2_mean']:.4f}")

        # Sklearn baselines
        for name, model in baselines:
            t0 = time.perf_counter()
            metrics = _sklearn_cv_regression(model, ds.X, ds.y, n_splits=N_SPLITS)
            elapsed = time.perf_counter() - t0
            row = {"model": name, "elapsed_s": round(elapsed, 3), **metrics}
            rows.append(row)
            print(f"  {name:<28} ({elapsed:.1f}s)  R²={metrics['r2_mean']:.4f}")

        _print_regression_table(ds.name, rows)
        all_results.append({"dataset": ds.name, "task": "regression", "results": rows})

    return all_results


def run_agent_streaming_benchmark(quick: bool = False) -> list[dict[str, Any]]:
    """Simulate the autonomous agent loop on a regression dataset.

    The dataset is split into a seed portion (first 20 %) used to ``fit``
    the predictor, and a stream portion (remaining 80 %) that arrives in
    mini-batches.  For each batch the agent:

    1. Predicts on the incoming X.
    2. Computes the prediction error *before* updating.
    3. Calls ``partial_fit`` with the true labels (oracle feedback).
    4. Re-computes the error *after* updating.

    Reported metrics show how R² and RMSE change as the agent receives more
    labelled data — the "learning curve under the agent loop".

    Args:
        quick: When True, uses fewer batches and a shorter seed.
    """
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_squared_error, r2_score as _r2

    data = load_diabetes()
    X_all, y_all = data.data, data.target
    n = len(y_all)
    seed_n = max(30, int(n * (0.1 if quick else 0.2)))
    batch_size = 20 if quick else 30
    n_batches = 3 if quick else 6

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(n)
    X_seed, y_seed = X_all[idx[:seed_n]], y_all[idx[:seed_n]]
    X_stream, y_stream = X_all[idx[seed_n:]], y_all[idx[seed_n:]]

    # Build a neural PhysicsPredictor for online learning
    predictor = PhysicsPredictor(backend="neural", plane="solid", n_cycles=15)
    agent = PhysicsAgent(predictor, uncertainty_threshold=0.35)

    predictor.fit(X_seed, y_seed)

    print(f"\n{'='*70}")
    print("AGENT STREAMING BENCHMARK — online learning (neural backend, diabetes)")
    print(f"{'='*70}")
    print(f"  Seed rows: {seed_n}   Batch size: {batch_size}   Batches: {n_batches}")
    print(f"  {'Batch':>6} {'Rows seen':>10} {'R² before':>10} {'R² after':>10}  {'ΔRMSE':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}  {'-'*8}")

    all_batches: list[dict[str, Any]] = []
    rows_seen = seed_n

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(y_stream))
        if start >= len(y_stream):
            break
        Xb, yb = X_stream[start:end], y_stream[start:end]

        # Evaluate BEFORE update
        try:
            y_pre = predictor.predict(Xb)
            r2_pre = float(_r2(yb, y_pre))
            rmse_pre = float(math.sqrt(mean_squared_error(yb, y_pre)))
        except Exception:
            r2_pre, rmse_pre = float("nan"), float("nan")

        # Agent feedback — teach with true labels
        try:
            agent.reward(Xb, yb)
        except Exception:
            predictor.partial_fit(Xb, yb)

        # Evaluate AFTER update
        try:
            y_post = predictor.predict(Xb)
            r2_post = float(_r2(yb, y_post))
            rmse_post = float(math.sqrt(mean_squared_error(yb, y_post)))
        except Exception:
            r2_post, rmse_post = float("nan"), float("nan")

        rows_seen += len(yb)
        delta_rmse = rmse_post - rmse_pre
        sign = "↑" if delta_rmse > 0 else "↓"
        print(
            f"  {b+1:>6} {rows_seen:>10} {r2_pre:>10.4f} {r2_post:>10.4f}  "
            f"{abs(delta_rmse):>7.2f}{sign}"
        )
        all_batches.append({
            "batch": b + 1,
            "rows_seen": rows_seen,
            "r2_before": r2_pre,
            "r2_after": r2_post,
            "rmse_before": rmse_pre,
            "rmse_after": rmse_post,
        })

    return [{"dataset": "diabetes_streaming", "task": "regression_online", "batches": all_batches}]


def _print_summary(all_results: list[dict[str, Any]]) -> None:
    """Print a compact summary of PhysML's rank among baselines."""
    print(f"\n{'='*70}")
    print("SUMMARY — PhysML rank across all benchmarks")
    print(f"{'='*70}")
    print(f"{'Dataset':<25} {'Task':<15} {'Metric':<12} {'PhysML':>10} {'Best Baseline':>14} {'Rank':>6}")
    print("-" * 70)

    for block in all_results:
        ds_name = block["dataset"]
        task = block["task"]
        rows = block.get("results")
        # Agent / streaming blocks don't have a "results" list; skip them
        if not rows:
            continue
        if task == "classification":
            key = "accuracy_mean"
            label = "Accuracy"
        else:
            key = "r2_mean"
            label = "R²"

        phys_val = next((r[key] for r in rows if r["model"] == "PhysML"), None)
        if phys_val is None:
            continue
        others = [r for r in rows if r["model"] != "PhysML"]
        best_other = max(others, key=lambda r: float(r.get(key, -1e9))) if others else None
        best_val = float(best_other[key]) if best_other else float("nan")
        all_vals = sorted([float(r.get(key, -1e9)) for r in rows], reverse=True)
        rank = all_vals.index(phys_val) + 1 if phys_val in all_vals else "?"

        print(
            f"{ds_name:<25} {task:<15} {label:<12} "
            f"{phys_val:>10.4f} {best_val:>14.4f} {str(rank):>6}"
        )


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhysML benchmark: physics predictor vs standard ML models"
    )
    parser.add_argument(
        "--tasks",
        choices=["all", "classification", "regression", "agent"],
        default="all",
        help="Which task types to benchmark (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to this file path",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster benchmark with fewer cycles and smaller datasets",
    )
    args = parser.parse_args()

    print("PhysML Benchmark")
    print(f"  n_cv_folds={N_SPLITS}  random_seed={RANDOM_SEED}  quick={args.quick}")

    all_results: list[dict[str, Any]] = []

    if args.tasks in ("all", "classification"):
        clf_results = run_classification_benchmark(quick=args.quick)
        all_results.extend(clf_results)

    if args.tasks in ("all", "regression"):
        reg_results = run_regression_benchmark(quick=args.quick)
        all_results.extend(reg_results)

    if args.tasks in ("all", "agent"):
        agent_results = run_agent_streaming_benchmark(quick=args.quick)
        all_results.extend(agent_results)

    _print_summary(all_results)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps({"physml_benchmark": all_results}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
