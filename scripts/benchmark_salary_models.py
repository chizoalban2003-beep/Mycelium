from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
import csv


# Allow running as `python scripts/benchmark_salary_models.py` without installing the package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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


def _frange(start: float, stop: float, step: float) -> list[float]:
    vals: list[float] = []
    x = float(start)
    stopf = float(stop)
    stepf = float(step)
    if stepf <= 0:
        raise ValueError("step must be > 0")
    # inclusive stop
    while x <= stopf + 1e-12:
        vals.append(float(round(x, 6)))
        x += stepf
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Mycelium vs sklearn regressors on a tabular CSV"
    )
    parser.add_argument("--csv", required=True, help="Path to a CSV dataset")
    parser.add_argument(
        "--target",
        default="salary",
        help="Regression target column name (default: salary), or 'random' to pick a numeric column",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=50_000, help="Rows to load (default 50k for speed). Use 0 for all.")
    parser.add_argument(
        "--random-target-min-unique",
        type=int,
        default=10,
        help="When --target=random, only consider numeric columns with at least this many unique values.",
    )
    parser.add_argument("--no-tree", action="store_true", help="Disable DecisionTreeRegressor")
    parser.add_argument("--no-mlp", action="store_true", help="Disable MLPRegressor (neural net)")
    parser.add_argument("--no-linear", action="store_true", help="Disable Linear/Ridge/ElasticNet baselines")
    parser.add_argument("--no-knn", action="store_true", help="Disable KNeighborsRegressor")
    parser.add_argument("--no-extra-trees", action="store_true", help="Disable ExtraTreesRegressor")
    parser.add_argument("--tree-max-depth", type=int, default=18)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=5)
    parser.add_argument("--rf-trees", type=int, default=150)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--extra-trees", type=int, default=400)
    parser.add_argument("--extra-max-depth", type=int, default=18)
    parser.add_argument("--knn-k", type=int, default=25)
    parser.add_argument("--enet-alpha", type=float, default=0.001)
    parser.add_argument("--enet-l1", type=float, default=0.2)
    parser.add_argument("--gb-max-iter", type=int, default=250)
    parser.add_argument("--mlp-hidden", default="128,64", help="Comma-separated hidden layer sizes")
    parser.add_argument("--mlp-max-iter", type=int, default=80)
    parser.add_argument("--plane", default="liquid", choices=["solid", "liquid", "gas"])

    parser.add_argument(
        "--mycelium-cycles",
        type=int,
        default=None,
        help="Mycelium electrophoresis cycles (default: 30 without Field-Effect, 100 with Field-Effect; sweep auto-sizes)",
    )

    # Optional: sweep only Mycelium configs (skip sklearn baselines).
    parser.add_argument(
        "--mycelium-sweep",
        action="store_true",
        help="Grid-search Mycelium Field-Effect knobs (alpha/start/type/decay) and report best configs",
    )
    parser.add_argument(
        "--mycelium-sweep-out",
        default="",
        help="Optional CSV path to write sweep rows (default: don't write)",
    )
    parser.add_argument(
        "--mycelium-sweep-top",
        type=int,
        default=15,
        help="How many best configs to print for MAE and RMSE",
    )

    # Sweep ranges (mirrors scripts/deep_freeze_sweep.py defaults).
    parser.add_argument("--mycelium-field-alpha-start", type=float, default=0.01)
    parser.add_argument("--mycelium-field-alpha-stop", type=float, default=0.25)
    parser.add_argument("--mycelium-field-alpha-step", type=float, default=0.03)
    parser.add_argument("--mycelium-field-start-start", type=int, default=40)
    parser.add_argument("--mycelium-field-start-stop", type=int, default=90)
    parser.add_argument("--mycelium-field-start-step", type=int, default=10)
    parser.add_argument(
        "--mycelium-field-coupling-types",
        default="linear,r_squared",
        help="Comma-separated: linear,r_squared",
    )
    parser.add_argument(
        "--mycelium-field-decay-values",
        default="1.0",
        help="Comma-separated floats; 1.0=constant, <1.0 decays, >1.0 grows after activation",
    )
    parser.add_argument(
        "--mycelium-sweep-include-field-off",
        action="store_true",
        help="Also include a field_disabled row in the sweep",
    )

    # Mycelium Field-Effect (v4.5+): optional late-cycle coupling.
    parser.add_argument("--mycelium-field", action="store_true", help="Enable Field-Effect coupling for Mycelium")
    parser.add_argument("--mycelium-field-alpha", type=float, default=0.10)
    parser.add_argument("--mycelium-field-start", type=int, default=80)
    parser.add_argument("--mycelium-field-coupling", default="linear", choices=["linear", "r_squared"])
    parser.add_argument(
        "--mycelium-field-alpha-exp-decay",
        type=float,
        default=1.0,
        help="1.0=constant; <1.0 decays after start; >1.0 grows after start",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"CSV not found: {args.csv}. Provide a dataset path via --csv (the repo no longer ships the old job-salary dataset)."
        )

    nrows = None if int(args.nrows) == 0 else int(args.nrows)
    df = pd.read_csv(args.csv, nrows=nrows)

    target = str(args.target)
    if target.lower() == "random":
        candidates: list[str] = []
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if int(df[c].nunique(dropna=True)) < int(args.random_target_min_unique):
                continue
            candidates.append(c)
        if not candidates:
            raise SystemExit(
                "--target=random found no numeric target columns. "
                f"Try lowering --random-target-min-unique (current {int(args.random_target_min_unique)})."
            )
        rng = np.random.default_rng(int(args.seed))
        target = str(rng.choice(candidates))
    elif target not in df.columns and target == "s" and "salary" in df.columns:
        target = "salary"

    if target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found. Available columns: {df.columns.tolist()}")

    feature_cols = [c for c in df.columns if c != target]
    y = df[target].to_numpy()

    train_mask, test_mask = train_test_split_mask(len(df), args.train_fraction, args.seed)

    if bool(args.mycelium_sweep):
        # Mycelium-only sweep: vary Field-Effect parameters.
        alphas = _frange(args.mycelium_field_alpha_start, args.mycelium_field_alpha_stop, args.mycelium_field_alpha_step)
        starts = list(range(int(args.mycelium_field_start_start), int(args.mycelium_field_start_stop) + 1, int(args.mycelium_field_start_step)))
        raw_types = [t.strip().lower() for t in str(args.mycelium_field_coupling_types).split(",") if t.strip()]
        types = [t for t in raw_types if t in ("linear", "r_squared")] or ["linear"]
        raw_decays = [x.strip() for x in str(args.mycelium_field_decay_values).split(",") if x.strip()]
        decays: list[float] = []
        for x in raw_decays:
            try:
                v = float(x)
            except Exception:
                continue
            if math.isfinite(v) and v > 0:
                decays.append(v)
        if not decays:
            decays = [1.0]

        # Pick cycle budget.
        cycles = int(args.mycelium_cycles) if args.mycelium_cycles is not None else 0
        if cycles <= 0:
            cycles = max(100, int(max(starts) if starts else 100) + 5)

        sweep_cfgs: list[tuple[bool, float, int, str, float]] = []
        if bool(args.mycelium_sweep_include_field_off):
            sweep_cfgs.append((False, 0.0, 0, "linear", 1.0))
        for a in alphas:
            for s in starts:
                for t in types:
                    for d in decays:
                        sweep_cfgs.append((True, float(a), int(s), str(t), float(d)))

        out_csv = str(args.mycelium_sweep_out).strip()
        writer: csv.DictWriter | None = None
        out_fh = None
        if out_csv:
            out_fh = open(out_csv, "w", newline="")
            writer = csv.DictWriter(
                out_fh,
                fieldnames=[
                    "field_enabled",
                    "field_alpha",
                    "field_start_cycle",
                    "field_coupling",
                    "field_decay",
                    "mae",
                    "rmse",
                    "seconds",
                ],
            )
            writer.writeheader()

        results: list[dict[str, float | str]] = []
        try:
            print("Mycelium sweep configs:", len(sweep_cfgs))
            print("Dataset:", args.csv)
            print("Rows:", len(df), "Train/Test:", int(train_mask.sum()), "/", int(test_mask.sum()), "Seed:", args.seed)
            print("Target:", target)
            print()

            for i, (enabled, alpha, start, coupling, decay) in enumerate(sweep_cfgs, start=1):
                t0 = time.time()
                pred = run_physics_prediction(
                    df,
                    target_col=target,
                    plane=PhysicsPlane(args.plane),
                    train_fraction=float(args.train_fraction),
                    random_seed=int(args.seed),
                    n_cycles=int(cycles),
                    cascade_enabled=True,
                    competitive_inhibition=True,
                    thermal_noise=True,
                    field_effect_enabled=bool(enabled),
                    field_effect_alpha=float(alpha),
                    field_effect_start_cycle=int(start),
                    field_effect_use_abs_corr=True,
                    field_effect_coupling=str(coupling),
                    field_effect_alpha_exp_decay=float(decay),
                    return_predictions=True,
                )
                seconds = float(time.time() - t0)

                mae = float(pred.metrics.mae or float("nan"))
                rmse = float(pred.metrics.rmse or float("nan"))
                row: dict[str, float | str] = {
                    "field_enabled": "1" if enabled else "0",
                    "field_alpha": float(alpha),
                    "field_start_cycle": int(start),
                    "field_coupling": str(coupling),
                    "field_decay": float(decay),
                    "mae": mae,
                    "rmse": rmse,
                    "seconds": seconds,
                }
                results.append(row)
                if writer is not None:
                    writer.writerow(row)

                if i % 20 == 0 or i == len(sweep_cfgs):
                    print(f"  {i}/{len(sweep_cfgs)} done")
        finally:
            if out_fh is not None:
                out_fh.close()

        top_n = int(args.mycelium_sweep_top)
        if top_n <= 0:
            top_n = 10
        best_mae = sorted(results, key=lambda r: (float(r["mae"]), float(r["rmse"])))[:top_n]
        best_rmse = sorted(results, key=lambda r: (float(r["rmse"]), float(r["mae"])))[:top_n]

        def _print(tag: str, rows0: list[dict[str, float | str]]) -> None:
            print(tag)
            print("field  alpha   start type       decay     MAE      RMSE   time(s)")
            for r in rows0:
                print(
                    f"{r['field_enabled']:>5s} "
                    f"{float(r['field_alpha']):>6.3f} "
                    f"{int(r['field_start_cycle']):>5d} "
                    f"{str(r['field_coupling']):<10s} "
                    f"{float(r['field_decay']):>7.4f} "
                    f"{float(r['mae']):>8.2f} "
                    f"{float(r['rmse']):>8.2f} "
                    f"{float(r['seconds']):>7.2f}"
                )
            print()

        _print("BEST_BY_MAE", best_mae)
        _print("BEST_BY_RMSE", best_rmse)

        if out_csv:
            print("Wrote sweep CSV:", out_csv)
        return 0

    # Choose cycle budget for the single Mycelium run.
    myc_cycles = int(args.mycelium_cycles) if args.mycelium_cycles is not None else 0
    if myc_cycles <= 0:
        myc_cycles = 100 if bool(args.mycelium_field) else 30
    if bool(args.mycelium_field) and int(args.mycelium_field_start) > int(myc_cycles):
        print(
            f"WARNING: Field-Effect start cycle ({int(args.mycelium_field_start)}) > mycelium cycles ({int(myc_cycles)}); "
            "Field-Effect will not activate. Consider --mycelium-cycles 100 (or higher)."
        )

    # Mycelium v4
    t0 = time.time()
    myc = run_physics_prediction(
        df,
        target_col=target,
        plane=PhysicsPlane(args.plane),
        train_fraction=float(args.train_fraction),
        random_seed=int(args.seed),
        n_cycles=int(myc_cycles),
        cascade_enabled=True,
        competitive_inhibition=True,
        thermal_noise=True,
        field_effect_enabled=bool(args.mycelium_field),
        field_effect_alpha=float(args.mycelium_field_alpha),
        field_effect_start_cycle=int(args.mycelium_field_start),
        field_effect_use_abs_corr=True,
        field_effect_coupling=str(args.mycelium_field_coupling),
        field_effect_alpha_exp_decay=float(args.mycelium_field_alpha_exp_decay),
    )
    t_my = time.time() - t0

    # Sklearn prep
    X = df[feature_cols]
    pre = build_preprocessor(df, feature_cols)

    rows: list[BenchmarkRow] = []

    def _fit_predict(name: str, pipe: Pipeline) -> None:
        t0_local = time.time()
        pipe.fit(X[train_mask], y[train_mask])
        pred = pipe.predict(X[test_mask])
        dt = time.time() - t0_local
        rows.append(BenchmarkRow(name, regression_metrics(y[test_mask], pred), float(dt)))

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=int(args.rf_trees),
        max_depth=int(args.rf_max_depth) if int(args.rf_max_depth) > 0 else None,
        random_state=int(args.seed),
        n_jobs=-1,
    )
    rf_pipe = Pipeline(steps=[("pre", pre), ("model", rf)])
    _fit_predict("RandomForest", rf_pipe)

    # Gradient Boosting (fast histogram-based)
    gb = HistGradientBoostingRegressor(
        random_state=int(args.seed),
        max_iter=int(args.gb_max_iter),
    )
    gb_pipe = Pipeline(steps=[("pre", pre), ("model", gb)])
    _fit_predict("HistGB", gb_pipe)

    # Decision Tree
    if not bool(args.no_tree):
        tree = DecisionTreeRegressor(
            random_state=int(args.seed),
            max_depth=int(args.tree_max_depth) if int(args.tree_max_depth) > 0 else None,
            min_samples_leaf=int(args.tree_min_samples_leaf),
        )
        tree_pipe = Pipeline(steps=[("pre", pre), ("model", tree)])
        _fit_predict("DecisionTree", tree_pipe)

    # Extra Trees
    if not bool(args.no_extra_trees):
        et = ExtraTreesRegressor(
            n_estimators=int(args.extra_trees),
            max_depth=int(args.extra_max_depth) if int(args.extra_max_depth) > 0 else None,
            random_state=int(args.seed),
            n_jobs=-1,
        )
        et_pipe = Pipeline(steps=[("pre", pre), ("model", et)])
        _fit_predict("ExtraTrees", et_pipe)

    # Linear models (scale for stability)
    if not bool(args.no_linear):
        lin_pipe = Pipeline(steps=[("pre", pre), ("scale", StandardScaler()), ("model", LinearRegression())])
        _fit_predict("LinearRegression", lin_pipe)

        ridge_pipe = Pipeline(
            steps=[("pre", pre), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=int(args.seed)))])
        _fit_predict("Ridge", ridge_pipe)

        enet = ElasticNet(
            alpha=float(args.enet_alpha),
            l1_ratio=float(args.enet_l1),
            random_state=int(args.seed),
            max_iter=5_000,
        )
        enet_pipe = Pipeline(steps=[("pre", pre), ("scale", StandardScaler()), ("model", enet)])
        _fit_predict("ElasticNet", enet_pipe)

    # KNN (needs scaling)
    if not bool(args.no_knn):
        knn = KNeighborsRegressor(n_neighbors=int(args.knn_k), weights="distance")
        knn_pipe = Pipeline(steps=[("pre", pre), ("scale", StandardScaler()), ("model", knn)])
        _fit_predict("KNN", knn_pipe)

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
        _fit_predict("MLP", mlp_pipe)

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
