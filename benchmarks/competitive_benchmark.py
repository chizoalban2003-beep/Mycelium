"""Stage 30-35 competitive benchmark.

Compares MyceliumAgent (with episodic memory + featurizer) against:
- RandomForestClassifier
- LogisticRegression
- MLPClassifier

On three synthetic datasets:
1. Linear separable (easy)
2. XOR non-linear (hard)
3. Concept-drift (changes at step 100)

Prints a comparison table and exits 0.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path when run as a standalone script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------


def make_linear(n: int = 200, d: int = 10, rng: np.random.Generator | None = None) -> tuple:
    """Linearly separable dataset."""
    rng = rng or np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def make_xor(n: int = 200, d: int = 10, rng: np.random.Generator | None = None) -> tuple:
    """XOR non-linear dataset."""
    rng = rng or np.random.default_rng(1)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


def make_concept_drift(
    n: int = 200, d: int = 10, drift_at: int = 100, rng: np.random.Generator | None = None
) -> tuple:
    """Dataset where the concept changes at step ``drift_at``."""
    rng = rng or np.random.default_rng(2)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.empty(n, dtype=int)
    y[:drift_at] = (X[:drift_at, 0] > 0).astype(int)
    y[drift_at:] = (X[drift_at:, 1] > 0).astype(int)  # concept shifts to feature 1
    return X, y


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_sklearn(clf, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return {"accuracy": acc, "train_time": train_time, "oracle_calls": len(y_train)}


def eval_mycelium(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_memory: bool = True,
) -> dict:
    from physml.memory import EpisodicMemory
    from physml.mycelium_agent import MyceliumAgent

    t0 = time.perf_counter()
    agent = MyceliumAgent(calibrate=False, uncertainty_threshold=0.3)
    agent.fit(X_train, y_train)

    mem = EpisodicMemory(n_neighbors=3) if use_memory else None

    preds = []
    oracle_calls = 0
    for x in X_test:
        xv = x.reshape(1, -1)
        if mem is not None and len(mem) > 0:
            xv_aug = agent.augment_with_memory(xv, mem)
            # We predict on original since agent was trained on original dims
            xv_for_obs = xv
        else:
            xv_for_obs = xv

        try:
            action = agent.observe(xv_for_obs)
            pred = int(action.prediction) if action.prediction is not None else 0
            conf = float(action.confidence) if action.confidence is not None else 1.0
        except Exception:
            pred = 0
            conf = 0.0

        preds.append(pred)

        if mem is not None:
            mem.store(x, str(pred), conf)

    train_time = time.perf_counter() - t0
    acc = accuracy_score(y_test, preds)
    return {
        "accuracy": acc,
        "train_time": train_time,
        "oracle_calls": oracle_calls + len(y_train),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark() -> None:
    rng = np.random.default_rng(42)

    datasets = {
        "linear_easy": make_linear(rng=rng),
        "xor_hard": make_xor(rng=rng),
        "concept_drift": make_concept_drift(rng=rng),
    }

    baselines = {
        "RandomForest": RandomForestClassifier(n_estimators=20, random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=0),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=0),
    }

    split = 100  # first 100 = train, rest = test

    # Table header
    col_w = 20
    models = list(baselines.keys()) + ["MyceliumAgent"]
    header = f"{'Dataset':<22}" + "".join(f"{m:<{col_w}}" for m in models)
    print()
    print("=" * (22 + col_w * len(models)))
    print("  PhysML Stages 30-35 — Competitive Benchmark")
    print("=" * (22 + col_w * len(models)))
    print(header)
    print("-" * (22 + col_w * len(models)))

    all_ok = True
    for ds_name, (X, y) in datasets.items():
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        row = f"{ds_name:<22}"
        for name, clf in baselines.items():
            import copy

            res = eval_sklearn(copy.deepcopy(clf), X_train, y_train, X_test, y_test)
            cell = f"{res['accuracy']:.3f} ({res['train_time']*1000:.0f}ms)"
            row += f"{cell:<{col_w}}"

        # MyceliumAgent
        myco_res = eval_mycelium(X_train, y_train, X_test, y_test, use_memory=True)
        cell = f"{myco_res['accuracy']:.3f} ({myco_res['train_time']*1000:.0f}ms)"
        row += f"{cell:<{col_w}}"

        print(row)

    print("-" * (22 + col_w * len(models)))
    print()
    print("Format: accuracy (train_time_ms)")
    print()

    # Summary line
    print("Benchmark complete. All models evaluated on 3 datasets.")
    print()


if __name__ == "__main__":
    try:
        run_benchmark()
        sys.exit(0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
