#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
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

    def _run_mycelium(label: str, **kwargs: object) -> None:
        t0 = time.perf_counter()
        pred = run_physics_prediction(
            df,
            target_col=target_col,
            train_fraction=train_fraction,
            random_seed=seed,
            top_k_weights=30,
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
            **kwargs,
        )
        dt = time.perf_counter() - t0
        y_true = np.array(pred.test_actual or [], dtype=str)
        y_pred = np.array(pred.test_predicted or [], dtype=str)
        acc = float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan")
        f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true.size else float("nan")
        rows.append(ClsRow(label, acc, f1, dt))

    # Mycelium configs (classification)
    _run_mycelium(
        "Mycelium (tuned gas, n=50)",
        plane=PhysicsPlane.gas,
        n_cycles=50,
        cycle_learning_rate=0.18,
    )

    # Optional PCR-style feature amplification rows.
    pcr_cfg = getattr(_bench_classification, "_pcr_cfg", None)
    if isinstance(pcr_cfg, dict) and bool(pcr_cfg.get("enabled")):
        pcr_kwargs = {
            "pcr_enabled": True,
            "pcr_cycles": int(pcr_cfg.get("cycles", 4)),
            "pcr_pvalue_threshold": float(pcr_cfg.get("p_threshold", 0.05)),
            "pcr_tau": float(pcr_cfg.get("tau", 4.0)),
            "pcr_gain": float(pcr_cfg.get("gain", 0.55)),
            "pcr_strength_cap": float(pcr_cfg.get("strength_cap", 2.5)),
            "pcr_amp_cap": float(pcr_cfg.get("amp_cap", 3.5)),
            "pcr_require_stable": bool(pcr_cfg.get("require_stable", True)),
        }
        _run_mycelium(
            f"Mycelium (tuned gas, n=50, PCR {int(pcr_kwargs['pcr_cycles'])}c)",
            plane=PhysicsPlane.gas,
            n_cycles=50,
            cycle_learning_rate=0.18,
            **pcr_kwargs,
        )

    # Extra Mycelium config mirroring the regression sweep-best knobs (may or may not help classification).
    _run_mycelium(
        "Mycelium (extra: gas, cycles=100, lr=0.25, shear=1.60)",
        plane=PhysicsPlane.gas,
        n_cycles=100,
        cycle_learning_rate=0.25,
        shear_alpha=1.60,
    )

    if isinstance(pcr_cfg, dict) and bool(pcr_cfg.get("enabled")):
        _run_mycelium(
            f"Mycelium (extra: gas, 100c, PCR {int(pcr_cfg.get('cycles', 4))}c)",
            plane=PhysicsPlane.gas,
            n_cycles=100,
            cycle_learning_rate=0.25,
            shear_alpha=1.60,
            pcr_enabled=True,
            pcr_cycles=int(pcr_cfg.get("cycles", 4)),
            pcr_pvalue_threshold=float(pcr_cfg.get("p_threshold", 0.05)),
            pcr_tau=float(pcr_cfg.get("tau", 4.0)),
            pcr_gain=float(pcr_cfg.get("gain", 0.55)),
            pcr_strength_cap=float(pcr_cfg.get("strength_cap", 2.5)),
            pcr_amp_cap=float(pcr_cfg.get("amp_cap", 3.5)),
            pcr_require_stable=bool(pcr_cfg.get("require_stable", True)),
        )

    # Optional: Mycelium with vibrational viscosity (applied to the tuned n=50 config)
    vib_cfg = getattr(_bench_classification, "_vib_cfg", None)
    if isinstance(vib_cfg, dict) and bool(vib_cfg.get("enabled")):
        _run_mycelium(
            "Mycelium (tuned gas, n=50, vib eta)",
            plane=PhysicsPlane.gas,
            n_cycles=50,
            cycle_learning_rate=0.18,
            vibrational_viscosity_enabled=True,
            vibrational_viscosity_period=int(vib_cfg.get("period", 5)),
            vibrational_viscosity_amplitude=float(vib_cfg.get("amplitude", 0.12)),
            vibrational_viscosity_waveform=str(vib_cfg.get("waveform", "square")),
        )

    _run_mycelium(
        "Mycelium (default)",
        plane=PhysicsPlane.solid,
        n_cycles=30,
        cycle_learning_rate=0.18,
    )

    if isinstance(pcr_cfg, dict) and bool(pcr_cfg.get("enabled")):
        _run_mycelium(
            f"Mycelium (default, PCR {int(pcr_cfg.get('cycles', 4))}c)",
            plane=PhysicsPlane.solid,
            n_cycles=30,
            cycle_learning_rate=0.18,
            pcr_enabled=True,
            pcr_cycles=int(pcr_cfg.get("cycles", 4)),
            pcr_pvalue_threshold=float(pcr_cfg.get("p_threshold", 0.05)),
            pcr_tau=float(pcr_cfg.get("tau", 4.0)),
            pcr_gain=float(pcr_cfg.get("gain", 0.55)),
            pcr_strength_cap=float(pcr_cfg.get("strength_cap", 2.5)),
            pcr_amp_cap=float(pcr_cfg.get("amp_cap", 3.5)),
            pcr_require_stable=bool(pcr_cfg.get("require_stable", True)),
        )

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

    def _run_mycelium(label: str, **kwargs: object) -> RegRow:
        t0 = time.perf_counter()
        call_kwargs: dict[str, object] = {
            "target_col": target_col,
            "train_fraction": train_fraction,
            "random_seed": seed,
            "top_k_weights": 30,
            "cascade_enabled": True,
            "competitive_inhibition": True,
            "thermal_noise": False,
            "stage2_cycles": 2,
            "stage2_trigger_cycle": 50,
            "stage2_shatter_complexes": True,
            "inhibition_strength": 0.7,
            "scavenger_cycles": 1,
            "low_confidence_mode": "none",
            "return_predictions": True,
        }
        call_kwargs.update(kwargs)
        pred = run_physics_prediction(df, **call_kwargs)
        dt = time.perf_counter() - t0
        y_true = np.array(pred.test_actual or [], dtype="float64")
        y_pred = np.array(pred.test_predicted or [], dtype="float64")
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(float(mean_squared_error(y_true, y_pred))))
        r2 = float(r2_score(y_true, y_pred))
        return RegRow(label, mae, rmse, r2, dt)

    # Mycelium baseline config (matches the existing table row)
    rows.append(
        _run_mycelium(
            "Mycelium (tuned gas, n=50)",
            plane=PhysicsPlane.gas,
            n_cycles=50,
            cycle_learning_rate=0.18,
        )
    )

    # Optional PCR-style feature amplification rows.
    pcr_cfg = getattr(_bench_regression, "_pcr_cfg", None)
    if isinstance(pcr_cfg, dict) and bool(pcr_cfg.get("enabled")):
        pcr_kwargs = {
            "pcr_enabled": True,
            "pcr_cycles": int(pcr_cfg.get("cycles", 4)),
            "pcr_pvalue_threshold": float(pcr_cfg.get("p_threshold", 0.05)),
            "pcr_tau": float(pcr_cfg.get("tau", 4.0)),
            "pcr_gain": float(pcr_cfg.get("gain", 0.55)),
            "pcr_strength_cap": float(pcr_cfg.get("strength_cap", 2.5)),
            "pcr_amp_cap": float(pcr_cfg.get("amp_cap", 3.5)),
            "pcr_require_stable": bool(pcr_cfg.get("require_stable", True)),
        }
        rows.append(
            _run_mycelium(
                f"Mycelium (tuned gas, n=50, PCR {int(pcr_kwargs['pcr_cycles'])}c)",
                plane=PhysicsPlane.gas,
                n_cycles=50,
                cycle_learning_rate=0.18,
                **pcr_kwargs,
            )
        )

    # Optional target-induced viscosity scaling (buffer shift) rows.
    tiv_cfg = getattr(_bench_regression, "_tiv_cfg", None)
    if isinstance(tiv_cfg, dict) and bool(tiv_cfg.get("enabled")):
        tiv_kwargs = {
            "target_induced_viscosity_enabled": True,
            "target_induced_viscosity_gain": float(tiv_cfg.get("gain", 0.5)),
            "target_induced_viscosity_min_multiplier": float(tiv_cfg.get("min_multiplier", 0.75)),
            "target_induced_viscosity_max_multiplier": float(tiv_cfg.get("max_multiplier", 1.0)),
        }
        rows.append(
            _run_mycelium(
                "Mycelium (tuned gas, n=50, buffer shift)",
                plane=PhysicsPlane.gas,
                n_cycles=50,
                cycle_learning_rate=0.18,
                **tiv_kwargs,
            )
        )

    # Explicit sweep-best row (so markdown can be regenerated from this script)
    rows.append(
        _run_mycelium(
            "Mycelium (sweep best: plane=gas, cycles=100, lr=0.25, shear=1.60)",
            plane=PhysicsPlane.gas,
            n_cycles=100,
            cycle_learning_rate=0.25,
            shear_alpha=1.60,
        )
    )

    if isinstance(tiv_cfg, dict) and bool(tiv_cfg.get("enabled")):
        tiv_kwargs = {
            "target_induced_viscosity_enabled": True,
            "target_induced_viscosity_gain": float(tiv_cfg.get("gain", 0.5)),
            "target_induced_viscosity_min_multiplier": float(tiv_cfg.get("min_multiplier", 0.75)),
            "target_induced_viscosity_max_multiplier": float(tiv_cfg.get("max_multiplier", 1.0)),
        }
        rows.append(
            _run_mycelium(
                "Mycelium (sweep best: gas 100c, buffer shift)",
                plane=PhysicsPlane.gas,
                n_cycles=100,
                cycle_learning_rate=0.25,
                shear_alpha=1.60,
                **tiv_kwargs,
            )
        )

    if isinstance(pcr_cfg, dict) and bool(pcr_cfg.get("enabled")):
        rows.append(
            _run_mycelium(
                f"Mycelium (sweep best: gas 100c, PCR {int(pcr_cfg.get('cycles', 4))}c)",
                plane=PhysicsPlane.gas,
                n_cycles=100,
                cycle_learning_rate=0.25,
                shear_alpha=1.60,
                pcr_enabled=True,
                pcr_cycles=int(pcr_cfg.get("cycles", 4)),
                pcr_pvalue_threshold=float(pcr_cfg.get("p_threshold", 0.05)),
                pcr_tau=float(pcr_cfg.get("tau", 4.0)),
                pcr_gain=float(pcr_cfg.get("gain", 0.55)),
                pcr_strength_cap=float(pcr_cfg.get("strength_cap", 2.5)),
                pcr_amp_cap=float(pcr_cfg.get("amp_cap", 3.5)),
                pcr_require_stable=bool(pcr_cfg.get("require_stable", True)),
            )
        )

    # Optional: Mycelium with vibrational viscosity
    vib_cfg = getattr(_bench_regression, "_vib_cfg", None)
    if isinstance(vib_cfg, dict) and bool(vib_cfg.get("enabled")):
        rows.append(
            _run_mycelium(
                "Mycelium (tuned gas, n=50, vib eta)",
                plane=PhysicsPlane.gas,
                n_cycles=50,
                cycle_learning_rate=0.18,
                vibrational_viscosity_enabled=True,
                vibrational_viscosity_period=int(vib_cfg.get("period", 5)),
                vibrational_viscosity_amplitude=float(vib_cfg.get("amplitude", 0.12)),
                vibrational_viscosity_waveform=str(vib_cfg.get("waveform", "square")),
            )
        )

    # Optional: broader random search (regression only)
    rnd_cfg = getattr(_bench_regression, "_random_cfg", None)
    if isinstance(rnd_cfg, dict) and int(rnd_cfg.get("trials", 0)) > 0:
        trials = int(rnd_cfg.get("trials", 0))
        topk = int(rnd_cfg.get("topk", 1))
        topk = max(1, min(10, topk))
        rng = random.Random(int(seed) + 99173)

        lr_min = float(rnd_cfg.get("lr_min", 0.12))
        lr_max = float(rnd_cfg.get("lr_max", 0.30))
        lr_sampling = str(rnd_cfg.get("lr_sampling", "linear")).lower().strip()
        if lr_sampling not in ("linear", "log"):
            lr_sampling = "linear"
        if not math.isfinite(lr_min) or lr_min <= 0.0:
            lr_min = 0.12
        if not math.isfinite(lr_max) or lr_max <= 0.0:
            lr_max = 0.30
        if lr_max < lr_min:
            lr_min, lr_max = lr_max, lr_min

        anneal_min = float(rnd_cfg.get("anneal_min", 1.0))
        anneal_max = float(rnd_cfg.get("anneal_max", 1.0))
        if not math.isfinite(anneal_min):
            anneal_min = 1.0
        if not math.isfinite(anneal_max):
            anneal_max = 1.0
        anneal_min = float(np.clip(anneal_min, 0.0, 1.0))
        anneal_max = float(np.clip(anneal_max, 0.0, 1.0))
        if anneal_max < anneal_min:
            anneal_min, anneal_max = anneal_max, anneal_min

        include_buffer = bool(rnd_cfg.get("include_buffer", False))
        buffer_gain_min = float(rnd_cfg.get("buffer_gain_min", 0.1))
        buffer_gain_max = float(rnd_cfg.get("buffer_gain_max", 1.0))
        buffer_min_mult_min = float(rnd_cfg.get("buffer_min_mult_min", 0.5))
        buffer_min_mult_max = float(rnd_cfg.get("buffer_min_mult_max", 0.9))
        buffer_max_mult = float(rnd_cfg.get("buffer_max_mult", 1.0))

        def _choice(seq: list[object]) -> object:
            return seq[int(rng.randrange(0, len(seq)))]

        def _u(a: float, b: float) -> float:
            return float(a + (b - a) * rng.random())

        def _u_log(a: float, b: float) -> float:
            a = float(a)
            b = float(b)
            if a <= 0.0 or b <= 0.0:
                return _u(max(1e-6, a), max(1e-6, b))
            la = math.log10(min(a, b))
            lb = math.log10(max(a, b))
            return float(10.0 ** (la + (lb - la) * rng.random()))

        candidates: list[RegRow] = []
        for _ in range(trials):
            plane = _choice([PhysicsPlane.gas, PhysicsPlane.liquid, PhysicsPlane.solid])
            ncy = int(_choice([30, 50, 75, 100, 120]))
            lr = float(_u_log(lr_min, lr_max) if lr_sampling == "log" else _u(lr_min, lr_max))
            shear = float(_u(0.25, 2.50))
            inhib = float(_u(0.30, 1.20))
            stage2_v = float(_u(0.8, 2.8))
            stage2_cycles = int(_choice([0, 1, 2, 3]))
            shatter = bool(_choice([False, True]))

            lr_sched = str(_choice(["constant", "linear_decay", "cosine_decay", "exp_decay"]))
            lr_min_mult = float(_u(0.10, 0.70))
            lr_exp_decay = float(_u(anneal_min, anneal_max))
            shear_sched = str(_choice(["constant", "linear_decay", "cosine_decay"]))
            shear_min_mult = float(_u(0.10, 0.70))

            buf_on = False
            buf_gain = 0.0
            buf_min_mult = 0.75
            buf_max_mult = float(buffer_max_mult)
            if include_buffer and bool(_choice([False, True, True])):
                buf_on = True
                buf_gain = float(_u(buffer_gain_min, buffer_gain_max))
                buf_min_mult = float(_u(buffer_min_mult_min, buffer_min_mult_max))

            vib_on = bool(_choice([False, False, True]))
            vib_period = int(_choice([3, 4, 5, 6, 7, 9]))
            vib_amp = float(_u(0.05, 0.22))
            vib_wave = str(_choice(["square", "sine"]))

            label = (
                "Mycelium (random trial: "
                f"plane={plane.value}, cycles={ncy}, lr={lr:.3f}, shear={shear:.2f}, "
                f"lr_sched={lr_sched}@{lr_min_mult:.2f}, lr_exp={lr_exp_decay:.3f}, shear_sched={shear_sched}@{shear_min_mult:.2f}, "
                f"buf={'y' if buf_on else 'n'}@g{buf_gain:.2f}/min{buf_min_mult:.2f}, "
                f"E2={stage2_v:.2f}, s2={stage2_cycles}, vib={'y' if vib_on else 'n'})"
            )
            try:
                row = _run_mycelium(
                    label,
                    plane=plane,
                    n_cycles=ncy,
                    cycle_learning_rate=lr,
                    cycle_learning_rate_schedule=lr_sched,
                    cycle_learning_rate_min_multiplier=lr_min_mult,
                    cycle_learning_rate_exp_decay=lr_exp_decay,
                    shear_alpha=shear,
                    shear_alpha_schedule=shear_sched,
                    shear_alpha_min_multiplier=shear_min_mult,
                    inhibition_strength=inhib,
                    stage2_voltage_multiplier=stage2_v,
                    stage2_cycles=stage2_cycles,
                    stage2_shatter_complexes=shatter,
                    vibrational_viscosity_enabled=vib_on,
                    vibrational_viscosity_period=vib_period,
                    vibrational_viscosity_amplitude=vib_amp,
                    vibrational_viscosity_waveform=vib_wave,
                    target_induced_viscosity_enabled=bool(buf_on),
                    target_induced_viscosity_gain=float(buf_gain),
                    target_induced_viscosity_min_multiplier=float(buf_min_mult),
                    target_induced_viscosity_max_multiplier=float(buf_max_mult),
                )
                candidates.append(row)
            except Exception:
                continue

        if candidates:
            candidates_sorted = sorted(candidates, key=lambda r: (r.mae, r.rmse))
            best = candidates_sorted[0]
            best_label = "Mycelium (random best)"
            try:
                i = best.model.find("random trial:")
                if i >= 0:
                    summary = best.model[i + len("random trial:") :].strip().strip(")")
                    if summary:
                        best_label = f"Mycelium (random best:{summary})"
            except Exception:
                pass
            rows.append(RegRow(best_label, best.mae, best.rmse, best.r2, best.seconds))

            # Optionally append a few more best candidates for inspection
            for extra in candidates_sorted[1 : 1 + topk]:
                rows.append(RegRow("Mycelium (random topk candidate)", extra.mae, extra.rmse, extra.r2, extra.seconds))

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
    parser.add_argument("--mycelium-vib-visc", action="store_true", help="Enable vibrational viscosity for an extra Mycelium row")
    parser.add_argument("--mycelium-vib-period", type=int, default=5)
    parser.add_argument("--mycelium-vib-amp", type=float, default=0.12)
    parser.add_argument("--mycelium-vib-waveform", choices=["square", "sine"], default="square")
    parser.add_argument("--mycelium-pcr", action="store_true", help="Add Mycelium+PCR rows (primer binding + amplification)")
    parser.add_argument("--mycelium-pcr-cycles", type=int, default=4)
    parser.add_argument("--mycelium-pcr-p", type=float, default=0.05)
    parser.add_argument("--mycelium-pcr-tau", type=float, default=4.0)
    parser.add_argument("--mycelium-pcr-gain", type=float, default=0.55)
    parser.add_argument("--mycelium-pcr-strength-cap", type=float, default=2.5)
    parser.add_argument("--mycelium-pcr-amp-cap", type=float, default=3.5)
    parser.add_argument("--mycelium-pcr-no-req-stable", action="store_true", help="Allow PCR binding on unstable features")
    parser.add_argument(
        "--mycelium-buffer-shift",
        action="store_true",
        help="Add Mycelium buffer-shift rows (target-induced viscosity scaling)",
    )
    parser.add_argument("--mycelium-buffer-gain", type=float, default=0.5)
    parser.add_argument("--mycelium-buffer-min-mult", type=float, default=0.75)
    parser.add_argument("--mycelium-buffer-max-mult", type=float, default=1.0)
    parser.add_argument("--mycelium-random-search", type=int, default=0, help="Run N random Mycelium trials for regression and add a best-row")
    parser.add_argument("--mycelium-random-topk", type=int, default=1, help="If random search enabled, also add up to K additional top candidates")
    parser.add_argument("--mycelium-random-lr-min", type=float, default=0.10)
    parser.add_argument("--mycelium-random-lr-max", type=float, default=0.40)
    parser.add_argument("--mycelium-random-lr-sampling", choices=["linear", "log"], default="log")
    parser.add_argument("--mycelium-random-anneal-min", type=float, default=0.90, help="Min exp-decay factor (1.0 disables annealing)")
    parser.add_argument("--mycelium-random-anneal-max", type=float, default=1.00, help="Max exp-decay factor (1.0 disables annealing)")
    parser.add_argument("--mycelium-random-include-buffer", action="store_true", help="Include buffer-shift params in random search")
    parser.add_argument("--mycelium-random-buffer-gain-min", type=float, default=0.10)
    parser.add_argument("--mycelium-random-buffer-gain-max", type=float, default=1.00)
    parser.add_argument("--mycelium-random-buffer-minmult-min", type=float, default=0.50)
    parser.add_argument("--mycelium-random-buffer-minmult-max", type=float, default=0.90)
    parser.add_argument("--mycelium-random-buffer-maxmult", type=float, default=1.00)
    args = parser.parse_args()

    path = Path(args.path)
    df = pd.read_csv(path, nrows=int(args.nrows) if int(args.nrows) > 0 else None)

    print("Dataset:", str(path), f"(nrows={df.shape[0]})")
    print(f"seed={int(args.seed)}  train_fraction={float(args.train_fraction)}")

    print("\nForced prediction (classification):")
    print(f"Target: {args.cls_target}")
    _bench_classification._vib_cfg = {
        "enabled": bool(args.mycelium_vib_visc),
        "period": int(args.mycelium_vib_period),
        "amplitude": float(args.mycelium_vib_amp),
        "waveform": str(args.mycelium_vib_waveform),
    }
    _bench_classification._pcr_cfg = {
        "enabled": bool(args.mycelium_pcr),
        "cycles": int(args.mycelium_pcr_cycles),
        "p_threshold": float(args.mycelium_pcr_p),
        "tau": float(args.mycelium_pcr_tau),
        "gain": float(args.mycelium_pcr_gain),
        "strength_cap": float(args.mycelium_pcr_strength_cap),
        "amp_cap": float(args.mycelium_pcr_amp_cap),
        "require_stable": (not bool(args.mycelium_pcr_no_req_stable)),
    }
    cls_rows = _bench_classification(df, target_col=str(args.cls_target), seed=int(args.seed), train_fraction=float(args.train_fraction))
    print("| Model | Accuracy | F1 (macro) | Time (s) |")
    print("|---|---|---|---|")
    for r in cls_rows:
        print(f"| {r.model} | {_fmt(r.accuracy)} | {_fmt(r.f1_macro)} | {_fmt(r.seconds, digits=2)} |")

    print("\nForced prediction (regression):")
    print(f"Target: {args.reg_target}")
    _bench_regression._vib_cfg = {
        "enabled": bool(args.mycelium_vib_visc),
        "period": int(args.mycelium_vib_period),
        "amplitude": float(args.mycelium_vib_amp),
        "waveform": str(args.mycelium_vib_waveform),
    }
    _bench_regression._pcr_cfg = {
        "enabled": bool(args.mycelium_pcr),
        "cycles": int(args.mycelium_pcr_cycles),
        "p_threshold": float(args.mycelium_pcr_p),
        "tau": float(args.mycelium_pcr_tau),
        "gain": float(args.mycelium_pcr_gain),
        "strength_cap": float(args.mycelium_pcr_strength_cap),
        "amp_cap": float(args.mycelium_pcr_amp_cap),
        "require_stable": (not bool(args.mycelium_pcr_no_req_stable)),
    }
    _bench_regression._tiv_cfg = {
        "enabled": bool(args.mycelium_buffer_shift),
        "gain": float(args.mycelium_buffer_gain),
        "min_multiplier": float(args.mycelium_buffer_min_mult),
        "max_multiplier": float(args.mycelium_buffer_max_mult),
    }
    _bench_regression._random_cfg = {
        "trials": int(args.mycelium_random_search),
        "topk": int(args.mycelium_random_topk),
        "lr_min": float(args.mycelium_random_lr_min),
        "lr_max": float(args.mycelium_random_lr_max),
        "lr_sampling": str(args.mycelium_random_lr_sampling),
        "anneal_min": float(args.mycelium_random_anneal_min),
        "anneal_max": float(args.mycelium_random_anneal_max),
        "include_buffer": bool(args.mycelium_random_include_buffer),
        "buffer_gain_min": float(args.mycelium_random_buffer_gain_min),
        "buffer_gain_max": float(args.mycelium_random_buffer_gain_max),
        "buffer_min_mult_min": float(args.mycelium_random_buffer_minmult_min),
        "buffer_min_mult_max": float(args.mycelium_random_buffer_minmult_max),
        "buffer_max_mult": float(args.mycelium_random_buffer_maxmult),
    }
    reg_rows = _bench_regression(df, target_col=str(args.reg_target), seed=int(args.seed), train_fraction=float(args.train_fraction))
    print("| Model | MAE | RMSE | R2 | Time (s) |")
    print("|---|---|---|---|---|")
    for r in reg_rows:
        print(f"| {r.model} | {_fmt(r.mae, digits=2)} | {_fmt(r.rmse, digits=2)} | {_fmt(r.r2)} | {_fmt(r.seconds, digits=2)} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
