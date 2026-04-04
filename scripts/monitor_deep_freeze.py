#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BestRow:
    rmse: float
    mae: float
    r2: float
    lr: float
    decay: float
    buffer_enabled: str
    buffer_gain: float
    buffer_min_mult: float


def _latest_csv(pattern: str) -> Path | None:
    files = sorted(Path("tmp_eval").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _parse_best(csv_path: Path) -> tuple[int, BestRow | None]:
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return 0, None

    completed = 0
    best: BestRow | None = None

    with csv_path.open("r", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            completed += 1
            try:
                rmse = float(row["rmse"])
                mae = float(row["mae"])
                r2 = float(row["r2"])
                lr = float(row["lr"])
                decay = float(row["decay"])
                buffer_enabled = str(row["buffer_enabled"])
                buffer_gain = float(row["buffer_gain"])
                buffer_min_mult = float(row["buffer_min_mult"])
            except Exception:
                continue

            if best is None or (rmse, mae) < (best.rmse, best.mae):
                best = BestRow(
                    rmse=rmse,
                    mae=mae,
                    r2=r2,
                    lr=lr,
                    decay=decay,
                    buffer_enabled=buffer_enabled,
                    buffer_gain=buffer_gain,
                    buffer_min_mult=buffer_min_mult,
                )

    return completed, best


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a Deep Freeze sweep CSV and print a summary every N seconds")
    parser.add_argument("--csv", default="", help="Path to sweep CSV. If empty, picks latest tmp_eval/deep_freeze_full_*.csv")
    parser.add_argument("--pattern", default="deep_freeze_full_*.csv")
    parser.add_argument("--total", type=int, default=1295)
    parser.add_argument("--interval", type=int, default=600, help="Seconds between updates (default: 600=10 min)")
    args = parser.parse_args()

    csv_path = Path(args.csv) if str(args.csv).strip() else (_latest_csv(str(args.pattern)) or Path(""))
    if not csv_path or not str(csv_path):
        raise SystemExit("No CSV found. Pass --csv tmp_eval/your_file.csv")

    total = int(args.total)
    interval = int(args.interval)
    if interval <= 0:
        interval = 60

    prev_t: float | None = None
    prev_completed: int | None = None

    while True:
        completed, best = _parse_best(csv_path)
        now = time.time()

        if prev_t is None or prev_completed is None:
            rate = 0.0
        else:
            dt = max(1e-9, now - prev_t)
            dc = float(completed - prev_completed)
            rate = max(0.0, dc / dt)

        remaining = max(0, total - completed)
        eta = remaining / max(1e-9, rate) if rate > 0 else float("inf")

        eta_txt = f"~{eta/60:.1f} min" if math.isfinite(eta) else "(warming up)"
        print(time.strftime("%H:%M:%S"), f"{completed}/{total} ({rate:.2f} cfg/s) ETA {eta_txt}")
        if best:
            print(
                "  best_by_RMSE: "
                f"RMSE={best.rmse:.2f} MAE={best.mae:.2f} R2={best.r2:.6f} "
                f"lr={best.lr:.3f} decay={best.decay:.3f} buf={best.buffer_enabled} "
                f"g={best.buffer_gain:.2f} min={best.buffer_min_mult:.2f}"
            )
        else:
            print("  best_by_RMSE: (none yet)")

        if completed >= total:
            return 0

        prev_t = now
        prev_completed = completed

        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
