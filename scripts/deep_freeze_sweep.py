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
    field_enabled: bool
    field_alpha: float
    field_start_cycle: int
    field_coupling: str
    field_alpha_exp_decay: float
    multibuffer_enabled: bool
    multibuffer_q_low: float
    multibuffer_q_high: float
    multibuffer_low_visc: float
    multibuffer_high_visc: float
    multibuffer_high_alpha_mult: float
    multibuffer_transition_frac: float


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

    if cfg.field_enabled:
        call_kwargs.update(
            {
                "field_effect_enabled": True,
                "field_effect_alpha": float(cfg.field_alpha),
                "field_effect_start_cycle": int(cfg.field_start_cycle),
                "field_effect_use_abs_corr": True,
                "field_effect_coupling": str(cfg.field_coupling),
                "field_effect_alpha_exp_decay": float(cfg.field_alpha_exp_decay),
            }
        )
    else:
        call_kwargs.update({"field_effect_enabled": False})

    if cfg.multibuffer_enabled:
        call_kwargs.update(
            {
                "multibuffer_enabled": True,
                "multibuffer_q_low": float(cfg.multibuffer_q_low),
                "multibuffer_q_high": float(cfg.multibuffer_q_high),
                "multibuffer_low_viscosity_multiplier": float(cfg.multibuffer_low_visc),
                "multibuffer_mid_viscosity_multiplier": 1.0,
                "multibuffer_high_viscosity_multiplier": float(cfg.multibuffer_high_visc),
                "multibuffer_low_field_alpha_multiplier": 1.0,
                "multibuffer_mid_field_alpha_multiplier": 1.0,
                "multibuffer_high_field_alpha_multiplier": float(cfg.multibuffer_high_alpha_mult),
                "multibuffer_transition_frac": float(cfg.multibuffer_transition_frac),
            }
        )
    else:
        call_kwargs.update({"multibuffer_enabled": False})

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
        "field_enabled": "1" if cfg.field_enabled else "0",
        "field_alpha": float(cfg.field_alpha),
        "field_start_cycle": int(cfg.field_start_cycle),
        "field_coupling": str(cfg.field_coupling),
        "field_alpha_exp_decay": float(cfg.field_alpha_exp_decay),
        "multibuffer_enabled": "1" if cfg.multibuffer_enabled else "0",
        "multibuffer_q_low": float(cfg.multibuffer_q_low),
        "multibuffer_q_high": float(cfg.multibuffer_q_high),
        "multibuffer_low_visc": float(cfg.multibuffer_low_visc),
        "multibuffer_high_visc": float(cfg.multibuffer_high_visc),
        "multibuffer_high_alpha_mult": float(cfg.multibuffer_high_alpha_mult),
        "multibuffer_transition_frac": float(cfg.multibuffer_transition_frac),
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
    parser = argparse.ArgumentParser(
        description="Deep Freeze sweep (100c gas) over LR+exp_decay+buffer shift (+ optional Field-Effect coupling); prioritizes RMSE"
    )
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

    # Field-Effect sweep knobs (v4.5+). These are multiplied onto the Deep Freeze grid.
    parser.add_argument("--field-enabled", action="store_true", help="Include Field-Effect configs in the sweep")
    parser.add_argument("--field-alpha-start", type=float, default=0.01)
    parser.add_argument("--field-alpha-stop", type=float, default=0.25)
    parser.add_argument("--field-alpha-step", type=float, default=0.03)
    parser.add_argument("--field-start-start", type=int, default=40)
    parser.add_argument("--field-start-stop", type=int, default=90)
    parser.add_argument("--field-start-step", type=int, default=10)
    parser.add_argument(
        "--field-coupling-types",
        default="linear,r_squared",
        help="Comma-separated: linear,r_squared",
    )
    parser.add_argument(
        "--field-alpha-exp-decay-values",
        default="1.0",
        help="Comma-separated floats; 1.0=constant, <1.0 decays, >1.0 grows after activation",
    )
    parser.add_argument(
        "--field-decay-values",
        dest="field_alpha_exp_decay_values",
        default=None,
        help="Alias for --field-alpha-exp-decay-values (interpreted as per-cycle multiplier after activation)",
    )
    parser.add_argument(
        "--include-field-off",
        action="store_true",
        help="Also include field_disabled rows even when --field-enabled is set",
    )

    # Multi-Buffer (v4.6) sweep knobs. These are multiplied onto the grid.
    parser.add_argument("--multibuffer-enabled", action="store_true", help="Include Multi-Buffer configs in the sweep")
    parser.add_argument("--multibuffer-q-low", type=float, default=0.33, help="Single q_low value (overridden by --multibuffer-q-low-values)")
    parser.add_argument("--multibuffer-q-high", type=float, default=0.67, help="Single q_high value (overridden by --multibuffer-q-high-values)")
    parser.add_argument(
        "--multibuffer-q-low-values",
        default="",
        help="Comma-separated floats; sweep q_low values (e.g., 0.20,0.25,0.33,0.40)",
    )
    parser.add_argument(
        "--multibuffer-q-high-values",
        default="",
        help="Comma-separated floats; sweep q_high values (e.g., 0.60,0.67,0.75,0.80)",
    )
    parser.add_argument(
        "--multibuffer-low-visc-values",
        default="1.0,1.1,1.2",
        help="Comma-separated floats; low-zone viscosity multiplier (>1 slows low-salary updates)",
    )
    parser.add_argument(
        "--multibuffer-high-visc-values",
        default="1.0",
        help="Comma-separated floats; high-zone viscosity multiplier (<1 speeds high-salary updates)",
    )
    parser.add_argument(
        "--multibuffer-high-alpha-values",
        default="1.0,1.1,1.2,1.3",
        help="Comma-separated floats; high-zone field-alpha multiplier (>1 strengthens coupling for high-salary rows)",
    )
    parser.add_argument(
        "--multibuffer-transition-frac-values",
        default="0.0",
        help="Comma-separated floats; soft-zone transition width as fraction of (t_high-t_low). 0.0=hard zones.",
    )
    parser.add_argument(
        "--include-multibuffer-off",
        action="store_true",
        help="Also include multibuffer_disabled rows even when --multibuffer-enabled is set",
    )

    args = parser.parse_args()

    csv_path = str(Path(args.path))
    if not Path(csv_path).exists():
        raise SystemExit(f"Missing dataset: {csv_path}")

    lrs = _frange(args.lr_start, args.lr_stop, args.lr_step)
    decays = _frange(args.decay_start, args.decay_stop, args.decay_step)
    gains = _frange(args.buffer_gain_start, args.buffer_gain_stop, args.buffer_gain_step)
    mins = _frange(args.buffer_min_start, args.buffer_min_stop, args.buffer_min_step)

    # Allow alias flag to override when provided.
    if args.field_alpha_exp_decay_values is None:
        args.field_alpha_exp_decay_values = "1.0"

    field_alphas: list[float] = []
    field_starts: list[int] = []
    field_couplings: list[str] = []
    field_decay_vals: list[float] = []
    if bool(args.field_enabled):
        field_alphas = _frange(args.field_alpha_start, args.field_alpha_stop, args.field_alpha_step)
        field_starts = list(range(int(args.field_start_start), int(args.field_start_stop) + 1, int(args.field_start_step)))
        raw_types = [t.strip().lower() for t in str(args.field_coupling_types).split(",") if t.strip()]
        field_couplings = [t for t in raw_types if t in ("linear", "r_squared")]
        if not field_couplings:
            field_couplings = ["linear"]
        raw_decays = [x.strip() for x in str(args.field_alpha_exp_decay_values).split(",") if x.strip()]
        for x in raw_decays:
            try:
                v = float(x)
            except Exception:
                continue
            if math.isfinite(v) and v > 0:
                field_decay_vals.append(v)
        if not field_decay_vals:
            field_decay_vals = [1.0]

    def _parse_csv_floats(raw: str) -> list[float]:
        vals: list[float] = []
        for x in [s.strip() for s in str(raw).split(",") if s.strip()]:
            try:
                v = float(x)
            except Exception:
                continue
            if math.isfinite(v):
                vals.append(float(v))
        return vals

    mb_q_low_single = float(args.multibuffer_q_low)
    mb_q_high_single = float(args.multibuffer_q_high)
    if not math.isfinite(mb_q_low_single):
        mb_q_low_single = 0.33
    if not math.isfinite(mb_q_high_single):
        mb_q_high_single = 0.67

    mb_q_low_vals = [v for v in _parse_csv_floats(str(args.multibuffer_q_low_values)) if 0.0 < v < 1.0]
    mb_q_high_vals = [v for v in _parse_csv_floats(str(args.multibuffer_q_high_values)) if 0.0 < v < 1.0]
    if not mb_q_low_vals:
        mb_q_low_vals = [float(mb_q_low_single)]
    if not mb_q_high_vals:
        mb_q_high_vals = [float(mb_q_high_single)]

    mb_q_pairs: list[tuple[float, float]] = []
    for ql in mb_q_low_vals:
        for qh in mb_q_high_vals:
            ql2 = float(np.clip(float(ql), 0.01, 0.99))
            qh2 = float(np.clip(float(qh), 0.01, 0.99))
            if ql2 < qh2:
                mb_q_pairs.append((ql2, qh2))
    if not mb_q_pairs:
        mb_q_pairs = [(0.33, 0.67)]
    mb_low_visc_vals: list[float] = []
    mb_high_visc_vals: list[float] = []
    mb_high_alpha_vals: list[float] = []
    mb_transition_fracs: list[float] = []
    if bool(args.multibuffer_enabled):
        for v in _parse_csv_floats(str(args.multibuffer_low_visc_values)):
            if math.isfinite(v) and v > 0:
                mb_low_visc_vals.append(float(v))
        for v in _parse_csv_floats(str(args.multibuffer_high_visc_values)):
            if math.isfinite(v) and v > 0:
                mb_high_visc_vals.append(float(v))
        for v in _parse_csv_floats(str(args.multibuffer_high_alpha_values)):
            if math.isfinite(v) and v > 0:
                mb_high_alpha_vals.append(float(v))
        for v in _parse_csv_floats(str(args.multibuffer_transition_frac_values)):
            if math.isfinite(v) and v >= 0:
                mb_transition_fracs.append(float(v))
        if not mb_low_visc_vals:
            mb_low_visc_vals = [1.0]
        if not mb_high_visc_vals:
            mb_high_visc_vals = [1.0]
        if not mb_high_alpha_vals:
            mb_high_alpha_vals = [1.0]
        if not mb_transition_fracs:
            mb_transition_fracs = [0.0]

    cfgs: list[SweepCfg] = []
    for lr in lrs:
        for d in decays:
            buffer_cfgs: list[tuple[bool, float, float]] = []
            if bool(args.include_buffer_off):
                buffer_cfgs.append((False, 0.0, 0.75))
            for g in gains:
                for m in mins:
                    buffer_cfgs.append((True, float(g), float(m)))

            field_cfgs: list[tuple[bool, float, int, str, float]] = []
            if bool(args.field_enabled):
                if bool(args.include_field_off):
                    field_cfgs.append((False, 0.0, 0, "linear", 1.0))
                for a in field_alphas:
                    for s in field_starts:
                        for t in field_couplings:
                            for fd in field_decay_vals:
                                field_cfgs.append((True, float(a), int(s), str(t), float(fd)))
            else:
                field_cfgs.append((False, 0.0, 0, "linear", 1.0))

            mb_cfgs: list[tuple[bool, float, float, float, float, float, float]] = []
            # (enabled, q_low, q_high, low_visc, high_visc, high_alpha_mult, transition_frac)
            if bool(args.multibuffer_enabled):
                if bool(args.include_multibuffer_off):
                    ql0, qh0 = mb_q_pairs[0]
                    mb_cfgs.append((False, float(ql0), float(qh0), 1.0, 1.0, 1.0, 0.0))
                for ql0, qh0 in mb_q_pairs:
                    for lv in mb_low_visc_vals:
                        for hv in mb_high_visc_vals:
                            for ha in mb_high_alpha_vals:
                                for tf in mb_transition_fracs:
                                    mb_cfgs.append(
                                        (True, float(ql0), float(qh0), float(lv), float(hv), float(ha), float(tf))
                                    )
            else:
                ql0, qh0 = mb_q_pairs[0]
                mb_cfgs.append((False, float(ql0), float(qh0), 1.0, 1.0, 1.0, 0.0))

            for buf_enabled, buf_gain, buf_min in buffer_cfgs:
                for f_enabled, f_alpha, f_start, f_type, f_decay in field_cfgs:
                    for mb_enabled0, mbql, mbqh, mblv, mbhv, mbha, mbtf in mb_cfgs:
                        cfgs.append(
                            SweepCfg(
                                lr=lr,
                                decay=d,
                                buffer_gain=float(buf_gain),
                                buffer_min_mult=float(buf_min),
                                buffer_enabled=bool(buf_enabled),
                                field_enabled=bool(f_enabled),
                                field_alpha=float(f_alpha),
                                field_start_cycle=int(f_start),
                                field_coupling=str(f_type),
                                field_alpha_exp_decay=float(f_decay),
                                multibuffer_enabled=bool(mb_enabled0),
                                multibuffer_q_low=float(mbql),
                                multibuffer_q_high=float(mbqh),
                                multibuffer_low_visc=float(mblv),
                                multibuffer_high_visc=float(mbhv),
                                multibuffer_high_alpha_mult=float(mbha),
                                multibuffer_transition_frac=float(mbtf),
                            )
                        )

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
        "field_enabled",
        "field_alpha",
        "field_start_cycle",
        "field_coupling",
        "field_alpha_exp_decay",
        "multibuffer_enabled",
        "multibuffer_q_low",
        "multibuffer_q_high",
        "multibuffer_low_visc",
        "multibuffer_high_visc",
        "multibuffer_high_alpha_mult",
        "multibuffer_transition_frac",
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
        f_enabled = str(r.get("field_enabled", "0"))
        if f_enabled == "1":
            field_txt = (
                f" field=1 a={float(r.get('field_alpha', 0.0)):.3f} "
                f"start={int(float(r.get('field_start_cycle', 0)))} "
                f"type={str(r.get('field_coupling', 'linear'))} "
                f"fdecay={float(r.get('field_alpha_exp_decay', 1.0)):.4f}"
            )
        else:
            field_txt = " field=0"

        mb_txt = ""
        if str(r.get("multibuffer_enabled", "0")) == "1":
            mb_txt = (
                f" mb=1 q=({float(r.get('multibuffer_q_low', 0.33)):.2f},{float(r.get('multibuffer_q_high', 0.67)):.2f})"
                f" low_visc={float(r.get('multibuffer_low_visc', 1.0)):.3f}"
                f" high_visc={float(r.get('multibuffer_high_visc', 1.0)):.3f}"
                f" high_alpha={float(r.get('multibuffer_high_alpha_mult', 1.0)):.3f}"
                f" tf={float(r.get('multibuffer_transition_frac', 0.0)):.3f}"
            )
        else:
            mb_txt = " mb=0"

        print(
            f"{tag}: RMSE={float(r['rmse']):.2f} MAE={float(r['mae']):.2f} R2={float(r['r2']):.6f} "
            f"lr={float(r['lr']):.3f} decay={float(r['decay']):.3f} "
            f"buf={r['buffer_enabled']} g={float(r['buffer_gain']):.2f} min={float(r['buffer_min_mult']):.2f}"
            f"{field_txt}{mb_txt} t={float(r['seconds']):.2f}s"
        )

    _fmt_best("BEST_RMSE", best_rmse)
    _fmt_best("BEST_MAE", best_mae)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
