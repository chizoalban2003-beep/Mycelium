from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, make_blobs, make_moons
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mycelium_app.physics_predictor import (
    PhysicsPlane,
    PredictorRuntimeState,
    load_predictor_state,
    run_physics_prediction,
    save_predictor_state,
    serialize_predictor_state,
)


ENB2012_DATA_PATH = Path("/mnt/chromeos/archive/archive.zip/ENB2012_data.csv")


def _frame_from_sklearn_dataset(loader: Any, *, target_name: str = "target") -> pd.DataFrame:
    bundle = loader(as_frame=True)
    frame = bundle.frame.copy()
    if target_name != "target" and "target" in frame.columns:
        frame = frame.rename(columns={"target": target_name})
    return frame


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _load_enb2012_frame() -> pd.DataFrame:
    if not ENB2012_DATA_PATH.exists():
        raise FileNotFoundError(f"ENB2012 dataset not found at {ENB2012_DATA_PATH}")
    frame = pd.read_csv(ENB2012_DATA_PATH)
    expected = {"X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"ENB2012 dataset is missing expected columns: {sorted(missing)}")
    return frame


def _ecosystem_vitals(result) -> dict[str, Any]:
    viscosities = [float(m.viscosity) for m in result.migration_map if m.viscosity is not None]
    velocities = [abs(float(m.terminal_velocity)) for m in result.migration_map if m.terminal_velocity is not None]
    complexes = [m.complex_id for m in result.migration_map if m.complex_id is not None]
    active_complexes = len(set(int(c) for c in complexes if c is not None))
    return {
        "mean_viscosity": None if not viscosities else float(np.mean(viscosities)),
        "mean_terminal_velocity": None if not velocities else float(np.mean(velocities)),
        "active_complexes": int(active_complexes),
        "mean_band_sharpness": result.metrics.gel_band_sharpness,
        "mean_smearing": result.metrics.gel_smearing,
    }


def _evaluate_regressor_with_plane(
    name: str,
    frame: pd.DataFrame,
    target_col: str,
    *,
    plane: PhysicsPlane,
    squeeze_enabled: bool = False,
    random_seed: int = 42,
) -> dict[str, Any]:
    y = frame[target_col].to_numpy(dtype=float)
    X = frame.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed)

    physics_df = pd.concat([X, pd.Series(y, name=target_col)], axis=1)
    runtime_state = PredictorRuntimeState(metadata={"benchmark": name, "target": target_col, "plane": plane.value})
    physics_result = run_physics_prediction(
        physics_df,
        target_col=target_col,
        plane=plane,
        runtime_state=runtime_state,
        train_fraction=0.75,
        random_seed=random_seed,
        n_cycles=8,
        max_preview_rows=0,
        enable_isotopes=True,
        viscosity_squeeze_enabled=bool(squeeze_enabled),
        return_predictions=True,
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, random_state=random_seed, n_jobs=-1)),
        ]
    )
    et = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=300, random_state=random_seed, n_jobs=-1)),
        ]
    )
    rf.fit(X_train, y_train)
    et.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    et_pred = et.predict(X_test)

    physics_mae = float(physics_result.metrics.mae or 0.0)
    physics_rmse = float(physics_result.metrics.rmse or 0.0)

    return {
        "task": "regression",
        "benchmark": "building_energy",
        "dataset": name,
        "target": target_col,
        "plane": plane.value,
        "viscosity_squeeze_enabled": bool(squeeze_enabled),
        "physics_mae": physics_mae,
        "physics_rmse": physics_rmse,
        "random_forest_mae": float(mean_absolute_error(y_test, rf_pred)),
        "random_forest_rmse": _safe_rmse(y_test, rf_pred),
        "extra_trees_mae": float(mean_absolute_error(y_test, et_pred)),
        "extra_trees_rmse": _safe_rmse(y_test, et_pred),
        "physics_metrics": asdict(physics_result.metrics),
        "ecosystem_vitals": _ecosystem_vitals(physics_result),
        "runtime_state": serialize_predictor_state(runtime_state),
    }


def _evaluate_classifier(name: str, frame: pd.DataFrame, target_col: str) -> dict[str, Any]:
    y = frame[target_col].to_numpy()
    X = frame.drop(columns=[target_col])
    stratify = y if pd.Series(y).nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    physics_df = pd.concat([X, pd.Series(y, name=target_col)], axis=1)
    runtime_state = PredictorRuntimeState(metadata={"benchmark": name})
    physics_result = run_physics_prediction(
        physics_df,
        target_col=target_col,
        plane=PhysicsPlane.liquid,
        runtime_state=runtime_state,
        train_fraction=0.75,
        random_seed=42,
        n_cycles=8,
        max_preview_rows=0,
        enable_isotopes=True,
        return_predictions=True,
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    et = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    rf.fit(X_train, y_train)
    et.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    et_pred = et.predict(X_test)

    scaled_train = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_scaled = scaled_train.fit_transform(X_train)
    X_test_scaled = scaled_train.transform(X_test)
    kmeans = KMeans(n_clusters=int(pd.Series(y).nunique()), n_init=10, random_state=42)
    gmm = GaussianMixture(n_components=int(pd.Series(y).nunique()), random_state=42)
    kmeans_labels = kmeans.fit_predict(X_test_scaled)
    gmm_labels = gmm.fit_predict(X_test_scaled)

    physics_accuracy = float(physics_result.metrics.accuracy or 0.0)
    physics_f1 = (
        float(
            f1_score(
                np.array(physics_result.test_actual or [], dtype=str),
                np.array(physics_result.test_predicted or [], dtype=str),
                average="weighted",
            )
        )
        if physics_result.test_predicted and physics_result.test_actual and len(physics_result.test_actual) == len(physics_result.test_predicted)
        else None
    )

    cluster_silhouette = silhouette_score(X_test_scaled, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else None
    gmm_silhouette = silhouette_score(X_test_scaled, gmm_labels) if len(np.unique(gmm_labels)) > 1 else None

    return {
        "task": "classification",
        "dataset": name,
        "physics_accuracy": physics_accuracy,
        "physics_f1_weighted": physics_f1,
        "random_forest_accuracy": float(accuracy_score(y_test, rf_pred)),
        "extra_trees_accuracy": float(accuracy_score(y_test, et_pred)),
        "physics_metrics": asdict(physics_result.metrics),
        "cluster_ari_kmeans": float(adjusted_rand_score(y_test, kmeans_labels)),
        "cluster_ari_gmm": float(adjusted_rand_score(y_test, gmm_labels)),
        "cluster_silhouette_kmeans": None if cluster_silhouette is None else float(cluster_silhouette),
        "cluster_silhouette_gmm": None if gmm_silhouette is None else float(gmm_silhouette),
        "runtime_state": serialize_predictor_state(runtime_state),
    }


def _evaluate_regressor(name: str, frame: pd.DataFrame, target_col: str) -> dict[str, Any]:
    y = frame[target_col].to_numpy(dtype=float)
    X = frame.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    physics_df = pd.concat([X, pd.Series(y, name=target_col)], axis=1)
    physics_result = run_physics_prediction(
        physics_df,
        target_col=target_col,
        plane=PhysicsPlane.liquid,
        train_fraction=0.75,
        random_seed=42,
        n_cycles=8,
        max_preview_rows=0,
        enable_isotopes=True,
        return_predictions=True,
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    et = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    rf.fit(X_train, y_train)
    et.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    et_pred = et.predict(X_test)

    physics_mae = float(physics_result.metrics.mae or 0.0)
    physics_rmse = float(physics_result.metrics.rmse or 0.0)

    return {
        "task": "regression",
        "dataset": name,
        "physics_mae": physics_mae,
        "physics_rmse": physics_rmse,
        "random_forest_mae": float(mean_absolute_error(y_test, rf_pred)),
        "random_forest_rmse": _safe_rmse(y_test, rf_pred),
        "extra_trees_mae": float(mean_absolute_error(y_test, et_pred)),
        "extra_trees_rmse": _safe_rmse(y_test, et_pred),
        "physics_metrics": asdict(physics_result.metrics),
    }


def run_energy_efficiency_sweep() -> dict[str, Any]:
    frame = _load_enb2012_frame()
    targets = ["Y1", "Y2"]
    planes = [PhysicsPlane.solid, PhysicsPlane.liquid, PhysicsPlane.gas]
    variants = [
        {"name": "baseline", "squeeze_enabled": False},
        {"name": "squeeze", "squeeze_enabled": True},
    ]

    target_reports: list[dict[str, Any]] = []
    for target_col in targets:
        plane_rows: list[dict[str, Any]] = []
        for variant in variants:
            for plane in planes:
                plane_rows.append(
                    _evaluate_regressor_with_plane(
                        "enb2012",
                        frame,
                        target_col,
                        plane=plane,
                        squeeze_enabled=bool(variant["squeeze_enabled"]),
                        random_seed=42,
                    )
                )

        best_row = min(plane_rows, key=lambda row: float(row["physics_rmse"]))
        target_reports.append(
            {
                "target": target_col,
                "planes": plane_rows,
                "best_plane": best_row["plane"],
                "best_variant": "squeeze" if bool(best_row.get("viscosity_squeeze_enabled")) else "baseline",
                "best_physics_rmse": float(best_row["physics_rmse"]),
                "best_physics_mae": float(best_row["physics_mae"]),
            }
        )

    combined_rows = []
    for report in target_reports:
        for row in report["planes"]:
            combined_rows.append(
                {
                    "target": report["target"],
                    "plane": row["plane"],
                    "variant": "squeeze" if bool(row.get("viscosity_squeeze_enabled")) else "baseline",
                    "physics_rmse": float(row["physics_rmse"]),
                    "physics_mae": float(row["physics_mae"]),
                    "mean_viscosity": row["ecosystem_vitals"]["mean_viscosity"],
                    "active_complexes": row["ecosystem_vitals"]["active_complexes"],
                    "mean_terminal_velocity": row["ecosystem_vitals"]["mean_terminal_velocity"],
                }
            )

    aggregate: dict[str, Any] = {
        "benchmark": "building_energy",
        "dataset": "ENB2012_data.csv",
        "path": str(ENB2012_DATA_PATH),
        "variants": [variant["name"] for variant in variants],
        "targets": target_reports,
        "plane_summary": combined_rows,
    }
    return aggregate


def _evaluate_unsupervised() -> dict[str, Any]:
    blob_X, blob_y = make_blobs(n_samples=600, centers=4, cluster_std=1.35, random_state=42)
    moon_X, moon_y = make_moons(n_samples=600, noise=0.07, random_state=42)

    results: list[dict[str, Any]] = []
    for dataset_name, X, y_true in (
        ("blobs", blob_X, blob_y),
        ("moons", moon_X, moon_y),
    ):
        X_scaled = StandardScaler().fit_transform(X)
        for model_name, model in (
            ("kmeans", KMeans(n_clusters=int(len(np.unique(y_true))), n_init=20, random_state=42)),
            ("gmm", GaussianMixture(n_components=int(len(np.unique(y_true))), random_state=42)),
        ):
            labels = model.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else None
            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "ari": float(adjusted_rand_score(y_true, labels)),
                    "silhouette": None if silhouette is None else float(silhouette),
                    "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
                }
            )

    return {"task": "unsupervised", "results": results}


def main() -> None:
    state_path = Path(".benchmark_runtime_state.json")
    runtime_state = PredictorRuntimeState(metadata={"source": "benchmark_models.py"})
    save_predictor_state(runtime_state, state_path)
    runtime_state = load_predictor_state(state_path)
    state_path.unlink(missing_ok=True)

    report = {
        "classification": [
            _evaluate_classifier("breast_cancer", _frame_from_sklearn_dataset(load_breast_cancer), "target"),
            _evaluate_classifier("wine", _frame_from_sklearn_dataset(load_wine), "target"),
        ],
        "building_energy": run_energy_efficiency_sweep(),
        "regression": [
            _evaluate_regressor("diabetes", _frame_from_sklearn_dataset(load_diabetes), "target"),
        ],
        "unsupervised": _evaluate_unsupervised(),
        "runtime_state_roundtrip": serialize_predictor_state(runtime_state),
    }

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
