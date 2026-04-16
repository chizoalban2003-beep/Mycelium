"""Stage 23 — Benchmark runner.

Runs ``benchmark_agent`` on four standard sklearn datasets and writes
per-dataset CSV files to ``benchmarks/results/``.

Usage::

    python benchmarks/run_benchmarks.py

The results are committed to the repository so users can reproduce or compare
without running the full benchmark themselves.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
)

# Add repo root to path so we can import physml without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physml import benchmark_agent, myco  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "iris": load_iris(return_X_y=True),
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "wine": load_wine(return_X_y=True),
}

ORACLE_BUDGET = 60
SEED_SIZE = 20
N_TRIALS = 3


def run_dataset(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    print(f"\n── {name}  ({X.shape[0]} × {X.shape[1]}) ──")

    result = benchmark_agent(
        myco(),
        X,
        y,
        oracle_budget=ORACLE_BUDGET,
        seed_size=SEED_SIZE,
        random_state=42,
    )

    csv_path = RESULTS_DIR / f"{name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "accuracy", "ask_rate"])
        n = max(len(result.accuracy_curve), len(result.ask_rate_curve))
        for i in range(n):
            acc = result.accuracy_curve[i] if i < len(result.accuracy_curve) else ""
            ask = result.ask_rate_curve[i] if i < len(result.ask_rate_curve) else ""
            writer.writerow([i, acc, ask])

    # Return headline numbers for the README table
    metrics: dict = {
        "dataset": name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "final_accuracy": result.accuracy_curve[-1] if result.accuracy_curve else float("nan"),
        "oracle_calls": result.oracle_calls,
        "ask_rate": result.ask_rate_curve[-1] if result.ask_rate_curve else float("nan"),
    }
    return metrics


def main() -> None:
    rows = []
    for name, (X, y) in DATASETS.items():
        rows.append(run_dataset(name, X, y))

    # Write summary CSV
    summary_path = RESULTS_DIR / "summary.csv"
    fieldnames = ["dataset", "n_samples", "n_features", "final_accuracy", "oracle_calls", "ask_rate"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
