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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    metrics: Metrics
    seconds: float


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
    parser = argparse.ArgumentParser(
        description="Benchmark Mycelium v4 vs DecisionTree vs RandomForest vs Gradient Boosting vs Neural Net (MLP) on the salary CSV"
    )
    parser.add_argument("--csv", default="tmp_eval/job_salary_prediction_dataset.csv")
    parser.add_argument("--target", default="salary")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=50_000, help="Rows to load (default 50k for speed). Use 0 for all.")
    parser.add_argument("--no-tree", action="store_true", help="Disable DecisionTreeRegressor")
    parser.add_argument("--no-mlp", action="store_true", help="Disable MLPRegressor (neural net)")
    parser.add_argument("--tree-max-depth", type=int, default=18)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5)
    parser.add_argument("--rf-trees", type=int, default=150)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--gb-max-iter", type=int, default=250)
    parser.add_argument("--mlp-hidden", default="128,64", help="Comma-separated hidden layer sizes")
    parser.add_argument("--mlp-max-iter", type=int, default=80)
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

    rows: list[BenchmarkRow] = []

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

    # Decision Tree
    if not bool(args.no_tree):
        tree = DecisionTreeRegressor(
            random_state=int(args.seed),
            max_depth=int(args.tree_max_depth) if int(args.tree_max_depth) > 0 else None,
            min_samples_leaf=int(args.tree_min_samples_leaf),
        )
        tree_pipe = Pipeline(steps=[("pre", pre), ("model", tree)])
        t0 = time.time()
        tree_pipe.fit(X[train_mask], y[train_mask])
        pred_tree = tree_pipe.predict(X[test_mask])
        t_tree = time.time() - t0
        rows.append(BenchmarkRow("DecisionTree", regression_metrics(y[test_mask], pred_tree), float(t_tree)))

    # Neural Net (MLP)
    if not bool(args.no_mlp):
        hidden = tuple(int(x.strip()) for x in str(args.mlp_hidden).split(",") if x.strip())
        if len(hidden) == 0:
            raise SystemExit("--mlp-hidden must contain at least one layer size, e.g. '128,64'")
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            random_state=int(args.seed),
            early_stopping=True,
            max_iter=int(args.mlp_max_iter),
            learning_rate_init=0.001,
            batch_size=256,
        )
        mlp_pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("scale", StandardScaler()),
                ("model", mlp),
            ]
        )
        t0 = time.time()
        mlp_pipe.fit(X[train_mask], y[train_mask])
        pred_mlp = mlp_pipe.predict(X[test_mask])
        t_mlp = time.time() - t0
        rows.append(BenchmarkRow("MLP", regression_metrics(y[test_mask], pred_mlp), float(t_mlp)))

    # Metrics
    m_rf = regression_metrics(y[test_mask], pred_rf)
    m_gb = regression_metrics(y[test_mask], pred_gb)

    rows.append(BenchmarkRow("RandomForest", m_rf, float(t_rf)))
    rows.append(BenchmarkRow("HistGB", m_gb, float(t_gb)))

    # Mycelium metrics are already computed on test split.
    print("Dataset:", args.csv)
    print("Rows:", len(df), "Train/Test:", int(train_mask.sum()), "/", int(test_mask.sum()), "Seed:", args.seed)
    print("Target:", target)
    print()

    print("Model           MAE             RMSE            Time(s)")
    print(f"Mycelium v4     {myc.metrics.mae:>14.3f} {myc.metrics.rmse:>14.3f} {t_my:>10.2f}")
    for row in sorted(rows, key=lambda r: r.metrics.mae):
        print(f"{row.name:<14s} {row.metrics.mae:>14.3f} {row.metrics.rmse:>14.3f} {row.seconds:>10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
