"""Stage 23 — Plot benchmark results.

Reads ``benchmarks/results/*.csv`` and produces accuracy-vs-oracle-calls
curves for each dataset, saved as ``benchmarks/results/accuracy_curves.png``.

Usage::

    python benchmarks/plot_benchmarks.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        return

    dataset_files = sorted(RESULTS_DIR.glob("*.csv"))
    dataset_files = [p for p in dataset_files if p.name != "summary.csv"]

    if not dataset_files:
        print("No result CSVs found. Run benchmarks/run_benchmarks.py first.")
        sys.exit(1)

    fig, axes = plt.subplots(1, len(dataset_files), figsize=(5 * len(dataset_files), 4))
    if len(dataset_files) == 1:
        axes = [axes]

    for ax, csv_path in zip(axes, dataset_files):
        rows = load_csv(csv_path)
        steps = [int(r["step"]) for r in rows]
        accuracies = [float(r["accuracy"]) for r in rows if r["accuracy"]]

        if accuracies:
            ax.plot(steps[: len(accuracies)], accuracies, marker="o", linewidth=2)
            ax.set_title(csv_path.stem)
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

    plt.suptitle("myco active-learning — accuracy vs. step", fontsize=13)
    plt.tight_layout()
    out_path = RESULTS_DIR / "accuracy_curves.png"
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
