#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction


_GLOBAL_DF: pd.DataFrame | None = None


def _init_worker(csv_path: str, nrows: int) -> None:
    global _GLOBAL_DF
    _GLOBAL_DF = pd.read_csv(csv_path, nrows=nrows if nrows > 0 else None)


@dataclass(frozen=True)
class SweepCfg:
    lr: float
    decay: float
    buffer_gain: float
    buffer_min_mult: float
    buffer_enabled: bool


def _eval_one(cfg: SweepCfg, *, seed: int, train_fraction: float) -> dict[str, float | str]:
    df = _GLOBAL_DF
    if df is None:
        raise RuntimeError("Worker dataframe not initialized")

    call_kwargs: dict[str, object] = {
        "target_col": "salary",
        "train_fraction": float(train_fraction),
        "random_seed": int(seed),
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
        "plane": PhysicsPlane.gas,
        "n_cycles": 100,
        "shear_alpha": 1.60,
        "cycle_learning_rate": float(cfg.lr),
        "cycle_learning_rate_schedule": "exp_decay",
        "cycle_learning_rate_exp_decay": float(cfg.decay),
        "cycle_learning_rate_min_multiplier": 0.02,
    }

    if cfg.buffer_enabled:
        call_kwargs.update(
            {
                "target_induced_viscosity_enabled": True,
                "target_induced_viscosity_gain": float(cfg.buffer_gain),
                "target_induced_viscosity_min_multiplier": float(cfg.buffer_min_mult),
                "target_induced_viscosity_max_multiplier": 1.0,
            }
        )
    else:
        call_kwargs.update({"target_induced_viscosity_enabled": False})

    t0 = time.perf_counter()
    pred = run_physics_prediction(df, **call_kwargs)
    seconds = float(time.perf_counter() - t0)

    y_true = np.asarray(pred.test_actual or [], dtype="float64")
    y_pred = np.asarray(pred.test_predicted or [], dtype="float64")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))

    # r2 is optional for sweep output; compute cheaply.
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

    return {
        "lr": float(cfg.lr),
        "decay": float(cfg.decay),
        "buffer_enabled": "1" if cfg.buffer_enabled else "0",
        "buffer_gain": float(cfg.buffer_gain),
        "buffer_min_mult": float(cfg.buffer_min_mult),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "seconds": seconds,
    }


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
    parser = argparse.ArgumentParser(description="Deep Freeze sweep (100c gas) over LR+exp_decay+buffer shift; prioritizes RMSE")
    parser.add_argument("--path", default="tmp_eval/job_salary_prediction_dataset.csv")
    parser.add_argument("--nrows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="tmp_eval/deep_freeze_sweep_results.csv")

    parser.add_argument("--lr-start", type=float, default=0.25)
    parser.add_argument("--lr-stop", type=float, default=0.45)
    parser.add_argument("--lr-step", type=float, default=0.05)

    parser.add_argument("--decay-start", type=float, default=0.985)
    parser.add_argument("--decay-stop", type=float, default=0.998)
    parser.add_argument("--decay-step", type=float, default=0.002)

    parser.add_argument("--buffer-gain-start", type=float, default=0.2)
    parser.add_argument("--buffer-gain-stop", type=float, default=1.2)
    parser.add_argument("--buffer-gain-step", type=float, default=0.2)

    parser.add_argument("--buffer-min-start", type=float, default=0.70)
    parser.add_argument("--buffer-min-stop", type=float, default=0.95)
    parser.add_argument("--buffer-min-step", type=float, default=0.05)

    parser.add_argument("--include-buffer-off", action="store_true", help="Also evaluate buffer_disabled configs (gain/min ignored)")

    args = parser.parse_args()

    csv_path = str(Path(args.path))
    if not Path(csv_path).exists():
        raise SystemExit(f"Missing dataset: {csv_path}")

    lrs = _frange(args.lr_start, args.lr_stop, args.lr_step)
    decays = _frange(args.decay_start, args.decay_stop, args.decay_step)
    gains = _frange(args.buffer_gain_start, args.buffer_gain_stop, args.buffer_gain_step)
    mins = _frange(args.buffer_min_start, args.buffer_min_stop, args.buffer_min_step)

    cfgs: list[SweepCfg] = []
    for lr in lrs:
        for d in decays:
            if bool(args.include_buffer_off):
                cfgs.append(SweepCfg(lr=lr, decay=d, buffer_gain=0.0, buffer_min_mult=0.75, buffer_enabled=False))
            for g in gains:
                for m in mins:
                    cfgs.append(SweepCfg(lr=lr, decay=d, buffer_gain=g, buffer_min_mult=m, buffer_enabled=True))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 4)))

    print(f"Sweep configs: {len(cfgs)}  workers={workers}", flush=True)
    print(f"LRs={len(lrs)} decays={len(decays)} gains={len(gains)} mins={len(mins)}", flush=True)
    print(f"Writing: {out_path}", flush=True)

    fieldnames = [
        "lr",
        "decay",
        "buffer_enabled",
        "buffer_gain",
        "buffer_min_mult",
        "mae",
        "rmse",
        "r2",
        "seconds",
    ]

    best_rmse: dict[str, float | str] | None = None
    best_mae: dict[str, float | str] | None = None

    t0 = time.perf_counter()
    completed = 0

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        f.flush()

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(csv_path, int(args.nrows)),
        ) as ex:
            futs = [
                ex.submit(_eval_one, cfg, seed=int(args.seed), train_fraction=float(args.train_fraction))
                for cfg in cfgs
            ]
            for fut in as_completed(futs):
                row = fut.result()
                w.writerow(row)
                completed += 1

                rmse = float(row["rmse"])  # type: ignore[arg-type]
                mae = float(row["mae"])  # type: ignore[arg-type]

                if best_rmse is None:
                    best_rmse = row
                else:
                    br = float(best_rmse["rmse"])  # type: ignore[arg-type]
                    bm = float(best_rmse["mae"])  # type: ignore[arg-type]
                    if (rmse, mae) < (br, bm):
                        best_rmse = row

                if best_mae is None:
                    best_mae = row
                else:
                    brm = float(best_mae["mae"])  # type: ignore[arg-type]
                    brr = float(best_mae["rmse"])  # type: ignore[arg-type]
                    if (mae, rmse) < (brm, brr):
                        best_mae = row

                if completed % 50 == 0 or completed == len(cfgs):
                    dt = time.perf_counter() - t0
                    rate = completed / max(1e-9, dt)
                    eta = (len(cfgs) - completed) / max(1e-9, rate)
                    print(f"{completed}/{len(cfgs)}  {rate:.2f} cfg/s  ETA~{eta/60:.1f} min", flush=True)
                if completed % 25 == 0:
                    f.flush()

    dt = time.perf_counter() - t0
    print(f"Done in {dt/60:.2f} min", flush=True)

    def _fmt_best(tag: str, r: dict[str, float | str] | None) -> None:
        if not r:
            return
        print(
            f"{tag}: RMSE={float(r['rmse']):.2f}  MAE={float(r['mae']):.2f}  R2={float(r['r2']):.6f}  "
            f"lr={float(r['lr']):.3f} decay={float(r['decay']):.3f} "
            f"buf={r['buffer_enabled']} g={float(r['buffer_gain']):.2f} min={float(r['buffer_min_mult']):.2f} "
            f"t={float(r['seconds']):.2f}s"
        )

    _fmt_best("BEST_RMSE", best_rmse)
    _fmt_best("BEST_MAE", best_mae)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
