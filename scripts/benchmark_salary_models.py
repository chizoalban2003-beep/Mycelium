from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass


# Allow running as `python scripts/benchmark_salary_models.py` without installing the package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float


def train_test_split_mask(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Match mycelium_app.physics_predictor._train_test_split_mask behavior.
    tf = float(train_fraction)
    if tf >= 0.999:
        train = np.ones(n, dtype=bool)
        test = np.ones(n, dtype=bool)
        return train, test
    if not (0.05 <= tf <= 0.95):
        raise ValueError("train_fraction must be between 0.05 and 0.95 (or 1.0)")
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    n_train = int(round(n * tf))
    n_train = max(1, min(n - 1, n_train))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    train[train_idx] = True
    test[test_idx] = True
    return train, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return Metrics(mae=mae, rmse=rmse)


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Mycelium v4 vs RF vs Gradient Boosting on the salary CSV")
    parser.add_argument("--csv", default="tmp_eval/job_salary_prediction_dataset.csv")
    parser.add_argument("--target", default="salary")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=50_000, help="Rows to load (default 50k for speed). Use 0 for all.")
    parser.add_argument("--rf-trees", type=int, default=150)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--gb-max-iter", type=int, default=250)
    parser.add_argument("--plane", default="liquid", choices=["solid", "liquid", "gas"])

    args = parser.parse_args()

    nrows = None if int(args.nrows) == 0 else int(args.nrows)
    df = pd.read_csv(args.csv, nrows=nrows)

    target = args.target
    if target not in df.columns and target == "s" and "salary" in df.columns:
        target = "salary"

    if target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found. Available columns: {df.columns.tolist()}")

    feature_cols = [c for c in df.columns if c != target]
    y = df[target].to_numpy()

    train_mask, test_mask = train_test_split_mask(len(df), args.train_fraction, args.seed)

    # Mycelium v4
    t0 = time.time()
    myc = run_physics_prediction(
        df,
        target_col=target,
        plane=PhysicsPlane(args.plane),
        train_fraction=float(args.train_fraction),
        random_seed=int(args.seed),
        cascade_enabled=True,
        competitive_inhibition=True,
        thermal_noise=True,
    )
    t_my = time.time() - t0

    # Sklearn prep
    X = df[feature_cols]
    pre = build_preprocessor(df, feature_cols)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=int(args.rf_trees),
        max_depth=int(args.rf_max_depth) if int(args.rf_max_depth) > 0 else None,
        random_state=int(args.seed),
        n_jobs=-1,
    )
    rf_pipe = Pipeline(steps=[("pre", pre), ("model", rf)])
    t0 = time.time()
    rf_pipe.fit(X[train_mask], y[train_mask])
    pred_rf = rf_pipe.predict(X[test_mask])
    t_rf = time.time() - t0

    # Gradient Boosting (fast histogram-based)
    gb = HistGradientBoostingRegressor(
        random_state=int(args.seed),
        max_iter=int(args.gb_max_iter),
    )
    gb_pipe = Pipeline(steps=[("pre", pre), ("model", gb)])
    t0 = time.time()
    gb_pipe.fit(X[train_mask], y[train_mask])
    pred_gb = gb_pipe.predict(X[test_mask])
    t_gb = time.time() - t0

    # Metrics
    m_rf = regression_metrics(y[test_mask], pred_rf)
    m_gb = regression_metrics(y[test_mask], pred_gb)

    # Mycelium metrics are already computed on test split.
    print("Dataset:", args.csv)
    print("Rows:", len(df), "Train/Test:", int(train_mask.sum()), "/", int(test_mask.sum()), "Seed:", args.seed)
    print("Target:", target)
    print()
    print("Model\t\tMAE\t\tRMSE\t\tTime(s)")
    print(f"Mycelium v4\t{myc.metrics.mae:.3f}\t{myc.metrics.rmse:.3f}\t{t_my:.2f}")
    print(f"RandomForest\t{m_rf.mae:.3f}\t{m_rf.rmse:.3f}\t{t_rf:.2f}")
    print(f"HistGB\t\t{m_gb.mae:.3f}\t{m_gb.rmse:.3f}\t{t_gb:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
