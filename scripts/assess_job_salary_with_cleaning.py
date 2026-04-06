#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow running this script directly without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from mycelium_app.physics_predictor import (
    PredictorError,
    _train_test_split_mask,
    clean_tabular_dataframe,
    run_physics_prediction,
)
from mycelium_app.presets import PRODUCTION_REGRESSION_KWARGS, PRODUCTION_REGRESSION_PRESET_NAME


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err * err)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_bar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _load_df(path: Path, nrows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=None if not nrows or nrows <= 0 else int(nrows))


def _fit_predict_sklearn_baselines(
    df_clean: pd.DataFrame,
    *,
    target_col: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.metrics import mean_absolute_error
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for baseline assessment. Install feature-engine (brings sklearn) "
            "or install scikit-learn explicitly."
        ) from e

    X = df_clean.drop(columns=[target_col])
    y = pd.to_numeric(df_clean[target_col], errors="coerce").to_numpy(dtype=float)

    X_train = X.loc[train_mask]
    y_train = y[train_mask]
    X_test = X.loc[test_mask]
    y_test = y[test_mask]

    # Match scripts/benchmark_salary_models.py column typing.
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # HistGradientBoostingRegressor requires dense input.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn<1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", ohe),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    models: dict[str, object] = {
        "LinearRegression": Pipeline(steps=[("pre", pre), ("scale", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline(steps=[("pre", pre), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=int(random_seed)))]),
        "HistGB": Pipeline(
            steps=[
                ("pre", pre),
                ("model", HistGradientBoostingRegressor(random_state=int(random_seed), max_iter=250)),
            ]
        ),
    }

    out: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        out[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": _rmse(y_test, pred),
            "r2": _r2(y_test, pred),
        }

    return out


def _fmt(x: float) -> str:
    return f"{x:,.2f}".replace(",", "")


def main() -> int:
    p = argparse.ArgumentParser(description="Assess job salary regression under Mycelium cleaning/outlier strategies")
    p.add_argument(
        "--data",
        default="",
        help="Path to a CSV dataset. If omitted, pass --generate-sample to create a synthetic dataset.",
    )
    p.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a small synthetic salary dataset if --data is missing (or points to a missing file).",
    )
    p.add_argument(
        "--generate-sample-out",
        default="tmp_eval/sample_salary_dataset.csv",
        help="Where to write the generated sample dataset (default: tmp_eval/sample_salary_dataset.csv)",
    )
    p.add_argument(
        "--generate-sample-rows",
        type=int,
        default=8000,
        help="Rows to generate for --generate-sample (default: 8000)",
    )
    p.add_argument("--target", default="salary", help="Regression target column name (default: salary)")
    p.add_argument("--nrows", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)

    p.add_argument("--cleaning-enabled", action="store_true", default=True)
    p.add_argument("--no-cleaning", dest="cleaning_enabled", action="store_false")
    p.add_argument(
        "--outlier-strategy",
        default="winsorize",
        choices=["winsorize", "iqr", "gaussian", "mad", "arbitrary", "feature_engine", "none"],
    )
    p.add_argument("--outlier-fold", type=float, default=1.5)
    p.add_argument("--q-low", type=float, default=0.005)
    p.add_argument("--q-high", type=float, default=0.995)
    p.add_argument("--arbitrary-min", type=float, default=None)
    p.add_argument("--arbitrary-max", type=float, default=None)

    p.add_argument("--cycles", type=int, default=None, help="Override n_cycles for Mycelium (default: preset)")
    p.add_argument("--write-md", default=None, help="Write markdown report to this path")

    args = p.parse_args()

    data_path = Path(str(args.data)) if str(args.data).strip() else Path("")
    if (not str(args.data).strip()) or (data_path and not data_path.exists()):
        if not bool(args.generate_sample):
            raise SystemExit(
                "Dataset not provided. Provide --data /path/to/your.csv, or run with --generate-sample. "
                "You can also generate one explicitly via: python scripts/sample_salary_dataset.py"
            )
        from scripts.sample_salary_dataset import make_sample_salary_dataset

        out_path = Path(str(args.generate_sample_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_gen = make_sample_salary_dataset(int(args.generate_sample_rows), int(args.seed))
        df_gen.to_csv(out_path, index=False)
        data_path = out_path
        print(f"Generated sample dataset -> {data_path}")

    df = _load_df(data_path, args.nrows)

    # Match predictor: pre-split cleanup (dedupe + drop missing target).
    df_pre, diag_pre = clean_tabular_dataframe(
        df,
        target_col=args.target,
        train_mask=None,
        drop_duplicates=True,
        drop_missing_target=True,
        impute_missing=False,
        clip_numeric_outliers=False,
    )

    train_mask, test_mask = _train_test_split_mask(int(df_pre.shape[0]), float(args.train_frac), int(args.seed))

    # Match predictor: post-split cleanup (impute + outlier capping using train stats only).
    if args.cleaning_enabled:
        df_clean, diag_post = clean_tabular_dataframe(
            df_pre,
            target_col=args.target,
            train_mask=train_mask,
            drop_duplicates=False,
            drop_missing_target=False,
            impute_missing=True,
            clip_numeric_outliers=True,
            outlier_strategy=str(args.outlier_strategy),
            outlier_fold=float(args.outlier_fold),
            outlier_q_low=float(args.q_low),
            outlier_q_high=float(args.q_high),
            arbitrary_min=args.arbitrary_min,
            arbitrary_max=args.arbitrary_max,
        )
    else:
        df_clean = df_pre
        diag_post = {"cleaning_enabled": False}

    # Baselines (sklearn)
    baselines = _fit_predict_sklearn_baselines(
        df_clean,
        target_col=args.target,
        train_mask=train_mask,
        test_mask=test_mask,
        random_seed=int(args.seed),
    )

    # Mycelium (locked production regression preset)
    my_kwargs: dict[str, object] = {
        "target_col": args.target,
        "train_fraction": float(args.train_frac),
        "random_seed": int(args.seed),
        "cleaning_enabled": bool(args.cleaning_enabled),
        "cleaning_drop_duplicates": False,
        "cleaning_drop_missing_target": False,
        "cleaning_outlier_strategy": str(args.outlier_strategy),
        "cleaning_outlier_fold": float(args.outlier_fold),
        "cleaning_outlier_q_low": float(args.q_low),
        "cleaning_outlier_q_high": float(args.q_high),
        "cleaning_arbitrary_min": args.arbitrary_min,
        "cleaning_arbitrary_max": args.arbitrary_max,
    }
    my_kwargs.update(dict(PRODUCTION_REGRESSION_KWARGS))
    if args.cycles is not None:
        my_kwargs["n_cycles"] = int(args.cycles)

    my_kwargs["return_predictions"] = True
    pred = run_physics_prediction(df_pre, **my_kwargs)
    if pred.target_kind != "numeric":
        raise PredictorError(f"Expected numeric target_kind, got {pred.target_kind}")

    pairs = [
        (float(a), float(b))
        for a, b in zip((pred.test_actual or []), (pred.test_predicted or []), strict=False)
        if a is not None and b is not None
    ]
    if pairs:
        y_true = np.asarray([p[0] for p in pairs], dtype=float)
        y_pred = np.asarray([p[1] for p in pairs], dtype=float)
        r2 = _r2(y_true, y_pred)
    else:
        r2 = float("nan")

    mycelium = {
        "name": PRODUCTION_REGRESSION_PRESET_NAME,
        "mae": float(pred.metrics.mae),
        "rmse": float(pred.metrics.rmse),
        "r2": float(r2),
    }

    cleaning_report = {
        "pre": diag_pre,
        "post": diag_post,
        "outlier_strategy": str(args.outlier_strategy),
        "outlier_fold": float(args.outlier_fold),
        "winsor_q": [float(args.q_low), float(args.q_high)],
        "arbitrary": [args.arbitrary_min, args.arbitrary_max],
        "n_rows_raw": int(df.shape[0]),
        "n_rows_used": int(df_pre.shape[0]),
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
    }

    lines: list[str] = []
    lines.append("# Job salary regression — cleaned pipeline")
    lines.append("")
    lines.append(f"Dataset: `{data_path}` (nrows={int(df.shape[0])}), target=`{args.target}`")
    lines.append(f"Split: train_fraction={float(args.train_frac):.2f}, seed={int(args.seed)}")
    lines.append(
        "Cleaning: dedupe+drop-missing-target pre-split; impute+outlier-capping post-split using TRAIN stats only"
        if args.cleaning_enabled
        else "Cleaning: disabled"
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| model | MAE | RMSE | R2 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Mycelium ({mycelium['name']}) | {_fmt(mycelium['mae'])} | {_fmt(mycelium['rmse'])} | {mycelium['r2']:.4f} |"
    )
    for name, m in baselines.items():
        lines.append(f"| {name} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {m['r2']:.4f} |")

    lines.append("")
    lines.append("## Cleaning diagnostics")
    lines.append("")
    # Keep it readable (no giant dict dumps)
    lines.append(f"- n_rows_used: {cleaning_report['n_rows_used']} (raw={cleaning_report['n_rows_raw']})")
    lines.append(f"- outlier_strategy: `{cleaning_report['outlier_strategy']}`")
    lines.append(f"- outlier_fold: {cleaning_report['outlier_fold']}")
    lines.append(f"- winsor_q: {cleaning_report['winsor_q']}")
    lines.append(f"- arbitrary: {cleaning_report['arbitrary']}")
    try:
        post = diag_post or {}
        lines.append(f"- imputed_values: {int(post.get('imputed_values', 0))}")
        lines.append(f"- clipped_outliers: {int(post.get('clipped_outliers', 0))}")
    except Exception:
        pass
    lines.append("")

    md = "\n".join(lines)
    print(md)

    if args.write_md:
        Path(args.write_md).write_text(md, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
