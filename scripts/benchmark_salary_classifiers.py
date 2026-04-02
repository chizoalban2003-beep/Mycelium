#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _train_test_split_mask(n: int, train_fraction: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    tf = float(train_fraction)
    if tf >= 0.999:
        train_mask = np.ones(n, dtype=bool)
        test_mask = np.ones(n, dtype=bool)
        return train_mask, test_mask
    if not (0.05 <= tf <= 0.95):
        raise ValueError("train_fraction must be between 0.05 and 0.95 (or 1.0 for no split)")
    if n < 3:
        raise ValueError("Need at least 3 rows")

    rng = np.random.default_rng(int(random_seed))
    idx = rng.permutation(n)
    n_train = int(round(n * tf))
    n_train = max(1, min(n - 1, n_train))

    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Mycelium vs sklearn classifiers on the salary dataset")
    parser.add_argument("--path", default="tmp_eval/job_salary_prediction_dataset.csv")
    parser.add_argument("--nrows", type=int, default=50_000)
    parser.add_argument("--target", default="remote_work", help="Classification target column")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--report", action="store_true", help="Print sklearn classification_report for each model")
    parser.add_argument(
        "--report-max-classes",
        type=int,
        default=30,
        help="If target has more than this many classes, skip per-class report and print only macro/weighted summary.",
    )
    parser.add_argument("--confusion", action="store_true", help="Print confusion matrix (raw + normalized)")
    parser.add_argument(
        "--confusion-max-classes",
        type=int,
        default=12,
        help="Skip printing confusion matrices when class count exceeds this (to avoid huge tables).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.path, nrows=int(args.nrows))
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    train_mask, test_mask = _train_test_split_mask(len(df), args.train_fraction, args.seed)

    # Classification labels
    y_raw = df[args.target].astype("string").fillna("__MISSING__")

    # Features
    X = df.drop(columns=[args.target])

    # ----- Mycelium
    from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction

    t0 = time.perf_counter()
    myc = run_physics_prediction(
        df,
        target_col=args.target,
        plane=PhysicsPlane.solid,
        train_fraction=float(args.train_fraction),
        random_seed=int(args.seed),
        top_k_weights=int(args.top_k),
        cascade_enabled=True,
        competitive_inhibition=True,
        thermal_noise=True,
        return_predictions=True,
    )
    myc_time = time.perf_counter() - t0
    myc_acc = myc.metrics.accuracy

    # ----- sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    le = LabelEncoder()
    y = le.fit_transform(y_raw.to_numpy())

    y_test_labels = y_raw.loc[test_mask].to_numpy(dtype=str)

    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("string")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    results: list[tuple[str, float, float, float]] = []
    reports: dict[str, str] = {}
    confusions: dict[str, str] = {}

    def _build_report(model_name: str, y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> None:
        if not bool(args.report):
            return
        n_classes = int(len(np.unique(y_true_labels)))
        if n_classes <= int(args.report_max_classes):
            rep = classification_report(y_true_labels, y_pred_labels, digits=4, zero_division=0)
            reports[model_name] = rep
            return

        # Too many classes: summarize only.
        rep_dict = classification_report(
            y_true_labels,
            y_pred_labels,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        macro = rep_dict.get("macro avg", {})
        weighted = rep_dict.get("weighted avg", {})
        reports[model_name] = (
            f"(report skipped: {n_classes} classes > {int(args.report_max_classes)})\n"
            f"macro avg:    precision={macro.get('precision', 0.0):.4f} recall={macro.get('recall', 0.0):.4f} f1={macro.get('f1-score', 0.0):.4f}\n"
            f"weighted avg: precision={weighted.get('precision', 0.0):.4f} recall={weighted.get('recall', 0.0):.4f} f1={weighted.get('f1-score', 0.0):.4f}\n"
        )

    def _build_confusion(model_name: str, y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> None:
        if not bool(args.confusion):
            return
        labels = np.unique(y_true_labels)
        n_classes = int(len(labels))
        if n_classes > int(args.confusion_max_classes):
            confusions[model_name] = f"(confusion skipped: {n_classes} classes > {int(args.confusion_max_classes)})"
            return

        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
        cm_norm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels, normalize="true")

        # Pretty print as a small table (rows=true, cols=pred).
        df_cm = pd.DataFrame(cm, index=[f"true:{x}" for x in labels], columns=[f"pred:{x}" for x in labels])
        df_norm = pd.DataFrame(
            np.round(cm_norm, 4),
            index=[f"true:{x}" for x in labels],
            columns=[f"pred:{x}" for x in labels],
        )
        confusions[model_name] = f"Raw:\n{df_cm.to_string()}\n\nNormalized (by true row):\n{df_norm.to_string()}"

    def eval_model(name: str, estimator) -> None:
        t0_local = time.perf_counter()
        pipe = Pipeline([
            ("prep", preprocess),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        dt = time.perf_counter() - t0_local
        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, average="macro"))
        results.append((name, acc, f1, dt))

        if bool(args.report):
            pred_labels = le.inverse_transform(pred)
            _build_report(name, y_test_labels, pred_labels)
        if bool(args.confusion):
            pred_labels = le.inverse_transform(pred)
            _build_confusion(name, y_test_labels, pred_labels)

    # Decision Tree
    eval_model(
        "DecisionTree",
        DecisionTreeClassifier(
            random_state=int(args.seed),
            max_depth=18,
            min_samples_leaf=5,
        ),
    )

    # Random Forest
    eval_model(
        "RandomForest",
        RandomForestClassifier(
            n_estimators=300,
            random_state=int(args.seed),
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        ),
    )

    # Hist Gradient Boosting
    eval_model(
        "HistGB",
        HistGradientBoostingClassifier(
            random_state=int(args.seed),
            max_iter=250,
            learning_rate=0.08,
            max_depth=None,
        ),
    )

    # Neural Net (MLP)
    t0_mlp = time.perf_counter()
    mlp_pipe = Pipeline(
        [
            ("prep", preprocess),
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    random_state=int(args.seed),
                    early_stopping=True,
                    max_iter=60,
                    learning_rate_init=0.001,
                    batch_size=256,
                ),
            ),
        ]
    )
    mlp_pipe.fit(X_train, y_train)
    mlp_pred = mlp_pipe.predict(X_test)
    mlp_time = time.perf_counter() - t0_mlp
    results.append(
        (
            "MLP",
            float(accuracy_score(y_test, mlp_pred)),
            float(f1_score(y_test, mlp_pred, average="macro")),
            float(mlp_time),
        )
    )

    if bool(args.report):
        _build_report("MLP", y_test_labels, le.inverse_transform(mlp_pred))
    if bool(args.confusion):
        _build_confusion("MLP", y_test_labels, le.inverse_transform(mlp_pred))

    if bool(args.report):
        if myc.test_actual is None or myc.test_predicted is None:
            reports["Mycelium v4"] = "(no test predictions returned; cannot build report)"
        else:
            _build_report("Mycelium v4", np.asarray(myc.test_actual, dtype=str), np.asarray(myc.test_predicted, dtype=str))

    if bool(args.confusion):
        if myc.test_actual is None or myc.test_predicted is None:
            confusions["Mycelium v4"] = "(no test predictions returned; cannot build confusion matrix)"
        else:
            _build_confusion(
                "Mycelium v4",
                np.asarray(myc.test_actual, dtype=str),
                np.asarray(myc.test_predicted, dtype=str),
            )

    # ----- Print
    print(f"Dataset: {args.path}")
    print(f"Rows: {len(df)} Train/Test: {int(train_mask.sum())} / {int(test_mask.sum())} Seed: {args.seed}")
    print(f"Target: {args.target}  Classes: {len(le.classes_)}")

    print("\nModel           Accuracy        F1(macro)       Time(s)")
    print(f"Mycelium v4     {_fmt(myc_acc):<14s} {'-':<14s} {myc_time:>10.2f}")
    for name, acc, f1, dt in sorted(results, key=lambda r: r[1], reverse=True):
        print(f"{name:<14s} {_fmt(acc):<14s} {_fmt(f1):<14s} {dt:>10.2f}")

    if bool(args.report):
        print("\n--- Classification reports (test split) ---")
        # Show Mycelium first, then baselines in accuracy order.
        if "Mycelium v4" in reports:
            print("\n[Mycelium v4]")
            print(reports["Mycelium v4"])

        for name, _, _, _ in sorted(results, key=lambda r: r[1], reverse=True):
            rep = reports.get(name)
            if rep is None:
                continue
            print(f"\n[{name}]")
            print(rep)

    if bool(args.confusion):
        print("\n--- Confusion matrices (test split) ---")
        if "Mycelium v4" in confusions:
            print("\n[Mycelium v4]")
            print(confusions["Mycelium v4"])

        for name, _, _, _ in sorted(results, key=lambda r: r[1], reverse=True):
            cm_txt = confusions.get(name)
            if cm_txt is None:
                continue
            print(f"\n[{name}]")
            print(cm_txt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
