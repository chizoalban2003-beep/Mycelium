#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction


def _split_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    tf = float(train_fraction)
    if tf >= 0.999:
        tf = 0.999
    if not (0.05 <= tf <= 0.95):
        raise ValueError(f"train_fraction must be in [0.05, 0.95], got {train_fraction}")
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(int(n))
    n_train = int(round(int(n) * tf))
    n_train = max(1, min(int(n) - 1, n_train))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return train_idx, test_idx


def _fmt(v: float | None, *, digits: int = 4) -> str:
    if v is None or not math.isfinite(float(v)):
        return "-"
    return f"{float(v):.{digits}f}"


@dataclass(frozen=True)
class ClsRow:
    model: str
    accuracy: float
    f1_macro: float
    seconds: float


@dataclass(frozen=True)
class RegRow:
    model: str
    mae: float
    rmse: float
    r2: float
    seconds: float


def _bench_classification(df: pd.DataFrame, *, target_col: str, seed: int, train_fraction: float) -> list[ClsRow]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    train_idx, test_idx = _split_indices(len(df), train_fraction, seed)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype("string").fillna("__MISSING__")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Two preprocessors:
    # - ordinal: compact, dense (good for tree/GB/KNN; avoids massive one-hot blowups)
    # - sparse one-hot: good for linear models
    pre_ordinal = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "ord",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pre_sparse = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "oh",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True,
                            min_frequency=10,
                        ),
                    ),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=1.0,
    )

    models: list[tuple[str, object, str]] = [
        ("Dummy (most_frequent)", DummyClassifier(strategy="most_frequent"), "ordinal"),
        ("HistGB", HistGradientBoostingClassifier(random_state=seed), "ordinal"),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1), "ordinal"),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1), "ordinal"),
        ("KNN", KNeighborsClassifier(n_neighbors=25), "ordinal"),
        (
            "LogReg",
            LogisticRegression(
                max_iter=2000,
                solver="saga",
                n_jobs=-1,
            ),
            "sparse",
        ),
        ("LinearSVC", LinearSVC(max_iter=8000, random_state=seed), "sparse"),
        ("DecisionTree", DecisionTreeClassifier(random_state=seed), "ordinal"),
    ]

    rows: list[ClsRow] = []

    # Mycelium
    t0 = time.perf_counter()
    pred = run_physics_prediction(
        df,
        target_col=target_col,
        plane=PhysicsPlane.gas,
        train_fraction=train_fraction,
        random_seed=seed,
        top_k_weights=30,
        n_cycles=50,
        cycle_learning_rate=0.18,
        cascade_enabled=True,
        competitive_inhibition=True,
        thermal_noise=False,
        stage2_cycles=2,
        stage2_trigger_cycle=50,
        stage2_shatter_complexes=True,
        inhibition_strength=0.7,
        scavenger_cycles=1,
        low_confidence_mode="none",
        return_predictions=True,
    )
    dt = time.perf_counter() - t0
    y_true_m = np.array(pred.test_actual or [], dtype=str)
    y_pred_m = np.array(pred.test_predicted or [], dtype=str)
    acc_m = float(accuracy_score(y_true_m, y_pred_m)) if y_true_m.size else float("nan")
    f1_m = float(f1_score(y_true_m, y_pred_m, average="macro")) if y_true_m.size else float("nan")
    rows.append(ClsRow("Mycelium (tuned gas, n=50)", acc_m, f1_m, dt))

    # Mycelium default-ish
    t1 = time.perf_counter()
    pred_def = run_physics_prediction(
        df,
        target_col=target_col,
        plane=PhysicsPlane.solid,
        train_fraction=train_fraction,
        random_seed=seed,
        top_k_weights=30,
        n_cycles=30,
        cycle_learning_rate=0.18,
        cascade_enabled=True,
        competitive_inhibition=True,
        thermal_noise=False,
        stage2_cycles=2,
        stage2_trigger_cycle=50,
        stage2_shatter_complexes=True,
        inhibition_strength=0.7,
        scavenger_cycles=1,
        low_confidence_mode="none",
        return_predictions=True,
    )
    dt_def = time.perf_counter() - t1
    y_true_d = np.array(pred_def.test_actual or [], dtype=str)
    y_pred_d = np.array(pred_def.test_predicted or [], dtype=str)
    acc_d = float(accuracy_score(y_true_d, y_pred_d)) if y_true_d.size else float("nan")
    f1_d = float(f1_score(y_true_d, y_pred_d, average="macro")) if y_true_d.size else float("nan")
    rows.append(ClsRow("Mycelium (default)", acc_d, f1_d, dt_def))

    # Sklearn baselines (same split indices as Mycelium)
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    for name, model, prep_kind in models:
        prep = pre_sparse if prep_kind == "sparse" else pre_ordinal
        pipe = Pipeline([("pre", prep), ("model", model)])
        t = time.perf_counter()
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)
        seconds = time.perf_counter() - t
        acc = float(accuracy_score(y_test, y_hat))
        f1 = float(f1_score(y_test, y_hat, average="macro"))
        rows.append(ClsRow(name, acc, f1, seconds))

    rows_sorted = sorted(rows, key=lambda r: (r.accuracy, r.f1_macro), reverse=True)
    return rows_sorted


def _bench_regression(df: pd.DataFrame, *, target_col: str, seed: int, train_fraction: float) -> list[RegRow]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    train_idx, test_idx = _split_indices(len(df), train_fraction, seed)

    X = df.drop(columns=[target_col])
    y = pd.to_numeric(df[target_col], errors="coerce")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre_ordinal = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "ord",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pre_sparse = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "oh",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True,
                            min_frequency=10,
                        ),
                    ),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=1.0,
    )

    models: list[tuple[str, object, str]] = [
        ("Dummy (mean)", DummyRegressor(strategy="mean"), "ordinal"),
        ("HistGB", HistGradientBoostingRegressor(random_state=seed), "ordinal"),
        ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1), "ordinal"),
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=400, random_state=seed, n_jobs=-1), "ordinal"),
        ("KNN", KNeighborsRegressor(n_neighbors=25), "ordinal"),
        ("Ridge", Ridge(alpha=1.0, random_state=seed), "sparse"),
        ("DecisionTree", DecisionTreeRegressor(random_state=seed), "ordinal"),
    ]

    rows: list[RegRow] = []

    # Mycelium
    t0 = time.perf_counter()
    pred = run_physics_prediction(
        df,
        target_col=target_col,
        plane=PhysicsPlane.gas,
        train_fraction=train_fraction,
        random_seed=seed,
        top_k_weights=30,
        n_cycles=50,
        cycle_learning_rate=0.18,
        cascade_enabled=True,
        competitive_inhibition=True,
        thermal_noise=False,
        stage2_cycles=2,
        stage2_trigger_cycle=50,
        stage2_shatter_complexes=True,
        inhibition_strength=0.7,
        scavenger_cycles=1,
        low_confidence_mode="none",
        return_predictions=True,
    )
    dt = time.perf_counter() - t0
    y_true_m = np.array(pred.test_actual or [], dtype="float64")
    y_pred_m = np.array(pred.test_predicted or [], dtype="float64")
    mae_m = float(mean_absolute_error(y_true_m, y_pred_m))
    rmse_m = float(math.sqrt(float(mean_squared_error(y_true_m, y_pred_m))))
    r2_m = float(r2_score(y_true_m, y_pred_m))
    rows.append(RegRow("Mycelium (tuned gas, n=50)", mae_m, rmse_m, r2_m, dt))

    # Sklearn baselines (same split indices as Mycelium)
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Drop any NaNs in y (should be none for this dataset)
    train_mask = y_train.notna()
    test_mask = y_test.notna()

    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask]

    for name, model, prep_kind in models:
        prep = pre_sparse if prep_kind == "sparse" else pre_ordinal
        pipe = Pipeline([("pre", prep), ("model", model)])
        t = time.perf_counter()
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)
        seconds = time.perf_counter() - t
        mae = float(mean_absolute_error(y_test, y_hat))
        rmse = float(math.sqrt(float(mean_squared_error(y_test, y_hat))))
        r2 = float(r2_score(y_test, y_hat))
        rows.append(RegRow(name, mae, rmse, r2, seconds))

    # Sort primarily by rmse (lower is better), then mae.
    rows_sorted = sorted(rows, key=lambda r: (r.rmse, r.mae))
    return rows_sorted


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Mycelium vs sklearn on job_salary_prediction_dataset")
    parser.add_argument("--path", default="tmp_eval/job_salary_prediction_dataset.csv")
    parser.add_argument("--nrows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--cls-target", default="remote_work")
    parser.add_argument("--reg-target", default="salary")
    args = parser.parse_args()

    path = Path(args.path)
    df = pd.read_csv(path, nrows=int(args.nrows) if int(args.nrows) > 0 else None)

    print("Dataset:", str(path), f"(nrows={df.shape[0]})")
    print(f"seed={int(args.seed)}  train_fraction={float(args.train_fraction)}")

    print("\nForced prediction (classification):")
    print(f"Target: {args.cls_target}")
    cls_rows = _bench_classification(df, target_col=str(args.cls_target), seed=int(args.seed), train_fraction=float(args.train_fraction))
    print("| Model | Accuracy | F1 (macro) | Time (s) |")
    print("|---|---|---|---|")
    for r in cls_rows:
        print(f"| {r.model} | {_fmt(r.accuracy)} | {_fmt(r.f1_macro)} | {_fmt(r.seconds, digits=2)} |")

    print("\nForced prediction (regression):")
    print(f"Target: {args.reg_target}")
    reg_rows = _bench_regression(df, target_col=str(args.reg_target), seed=int(args.seed), train_fraction=float(args.train_fraction))
    print("| Model | MAE | RMSE | R2 | Time (s) |")
    print("|---|---|---|---|---|")
    for r in reg_rows:
        print(f"| {r.model} | {_fmt(r.mae, digits=2)} | {_fmt(r.rmse, digits=2)} | {_fmt(r.r2)} | {_fmt(r.seconds, digits=2)} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
