#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{float(v):.4f}"


def _print_run(name: str, pred, seconds: float) -> None:
    m = pred.metrics
    print(f"\n== {name} ==")
    print(f"time_s={seconds:.2f}")
    print(f"accuracy={_fmt(m.accuracy)}")
    if m.coverage is not None:
        print(f"coverage={_fmt(m.coverage)}  abstain_rate={_fmt(m.abstain_rate)}  selective_acc={_fmt(m.selective_accuracy)}")

    trapped = [x for x in pred.migration_map if getattr(x, "state", "") == "trapped"]
    trapped_sorted = sorted(trapped, key=lambda x: float(getattr(x, "viscosity", 0.0)), reverse=True)[:12]
    if trapped_sorted:
        print("top_trapped_features:")
        for x in trapped_sorted:
            pv = getattr(x, "p_value", None)
            pv_s = "-" if pv is None else f"{float(pv):.3g}"
            print(
                f"  {x.feature:18s} state=trapped kind={x.feature_kind:11s} ion={x.ionization:12s} "
                f"visc={float(x.viscosity):.3f} vel={float(x.terminal_velocity):.3f} charge={float(x.charge):.3f} p={pv_s}"
            )

    try:
        diag = getattr(pred, "diagnostics", None) or {}
        sel = diag.get("selective")
        if sel and isinstance(sel, dict):
            thresholds = (sel.get("thresholds") or {})
            sieve = thresholds.get("secondary_sieve") if isinstance(thresholds, dict) else None
            if sieve and isinstance(sieve, dict) and sieve.get("enabled"):
                print(
                    "secondary_sieve: "
                    f"events={sieve.get('events')} rows_total={sieve.get('rows_total')} "
                    f"cycles={sieve.get('cycles')} reverse={sieve.get('reverse_multiplier')} "
                    f"noise_std={sieve.get('noise_std')} inst_min={sieve.get('instability_min')} "
                    f"update_max={sieve.get('update_norm_max')}"
                )
            stages = (sel.get("test_stages") or {})
            reasons = (sel.get("final_abstain_reasons") or {})
            if stages:
                print("abstain_stages:")
                for k in ("pre_reionization", "post_reionization", "final"):
                    st = stages.get(k)
                    if not st:
                        continue
                    ar = st.get("abstain_rate", None)
                    cv = st.get("coverage", None)
                    print(f"  {k:16s} abstain_rate={_fmt(ar)} coverage={_fmt(cv)} n_abstain={st.get('n_abstain')} n_test={st.get('n_test')}")
            if reasons and reasons.get("n_abstain", 0):
                print("abstain_reasons_within_abstained:")
                for key in ("conf_low", "smear_high", "ion_gate_blocked", "base_low_and_ion_gate"):
                    v = reasons.get(key)
                    if not v:
                        continue
                    print(
                        f"  {key:22s} count={v.get('count')} pct_of_abstain={float(v.get('pct_of_abstain', 0.0)):.3f}"
                    )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark selective + secondary ionization (cascade expansion)")
    parser.add_argument("--path", default="tmp_eval/job_salary_prediction_dataset.csv")
    parser.add_argument("--nrows", type=int, default=8000)
    parser.add_argument("--target", default="remote_work")
    parser.add_argument("--plane", default="gas", choices=["solid", "liquid", "gas"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)

    # Baseline selective settings (match model_performance_table.md defaults)
    parser.add_argument("--low-mode", default="abstain", choices=["none", "flag", "abstain"])
    parser.add_argument("--conf", type=float, default=0.0, help="0 = auto")
    parser.add_argument("--smear", type=float, default=0.0, help="0 = auto")
    parser.add_argument("--conf-q", type=float, default=0.63)
    parser.add_argument("--smear-q", type=float, default=0.80)
    parser.add_argument("--ion-gate", action="store_true", default=True)
    parser.add_argument("--ion-z", type=float, default=0.25)
    parser.add_argument("--ion-p", type=float, default=0.05)

    # Tuned gas settings
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.18)
    parser.add_argument("--cascade", action="store_true", default=True)
    parser.add_argument("--inhibit", action="store_true", default=True)
    parser.add_argument("--thermal", action="store_true", default=False)
    parser.add_argument("--stage2-cycles", type=int, default=2)
    parser.add_argument("--stage2-trigger", type=int, default=50)
    parser.add_argument("--stage2-shatter", action="store_true", default=True)
    parser.add_argument("--inhibition-strength", type=float, default=0.7)
    parser.add_argument("--scavenger", type=int, default=1)

    # Secondary ionization settings
    parser.add_argument("--sec", action="store_true", default=True)
    parser.add_argument("--sec-cycles", type=int, default=2)
    parser.add_argument("--sec-visc", type=float, default=0.65, help="Secondary viscosity multiplier end value")
    parser.add_argument("--sec-visc-anneal", action="store_true", default=False)
    parser.add_argument("--sec-visc-start", type=float, default=None, help="Secondary viscosity multiplier start value")
    parser.add_argument("--sec-shear", type=float, default=1.10)
    parser.add_argument("--sec-inhib", type=float, default=0.85)
    parser.add_argument("--sec-promote-votes", type=int, default=3)
    parser.add_argument("--sec-promote-z", type=float, default=0.35)
    parser.add_argument("--sec-promote-conf", type=float, default=0.40)

    # Reciprocating Sieve (v4.2)
    parser.add_argument("--sec-sieve", action="store_true", default=False)
    parser.add_argument("--sec-sieve-cycles", type=int, default=2)
    parser.add_argument("--sec-sieve-reverse", type=float, default=0.75)
    parser.add_argument("--sec-sieve-noise", type=float, default=0.04)
    parser.add_argument("--sec-sieve-inst", type=float, default=0.65)
    parser.add_argument("--sec-sieve-update-max", type=float, default=0.003)

    args = parser.parse_args()

    df = pd.read_csv(args.path, nrows=int(args.nrows) if int(args.nrows) > 0 else None)

    base_kwargs = dict(
        target_col=str(args.target),
        plane=PhysicsPlane(str(args.plane)),
        train_fraction=float(args.train_fraction),
        random_seed=int(args.seed),
        top_k_weights=int(args.top_k),
        n_cycles=int(args.cycles),
        cycle_learning_rate=float(args.lr),
        cascade_enabled=bool(args.cascade),
        competitive_inhibition=bool(args.inhibit),
        thermal_noise=bool(args.thermal),
        stage2_cycles=int(args.stage2_cycles),
        stage2_trigger_cycle=int(args.stage2_trigger),
        stage2_shatter_complexes=bool(args.stage2_shatter),
        inhibition_strength=float(args.inhibition_strength),
        scavenger_cycles=int(args.scavenger),
        low_confidence_mode=str(args.low_mode),
        low_confidence_threshold=float(args.conf),
        low_confidence_entropy_threshold=float(args.smear),
        low_confidence_auto_conf_quantile=float(args.conf_q),
        low_confidence_auto_smear_quantile=float(args.smear_q),
        low_confidence_require_ionized=bool(args.ion_gate),
        low_confidence_ionization_pvalue=float(args.ion_p),
        low_confidence_ionization_z_min=float(args.ion_z),
        return_predictions=True,
    )

    t0 = time.perf_counter()
    pred_base = run_physics_prediction(df, **base_kwargs, low_confidence_secondary_enabled=False, low_confidence_secondary_cycles=0)
    dt = time.perf_counter() - t0
    _print_run("baseline_selective", pred_base, dt)

    t1 = time.perf_counter()
    pred_sec = run_physics_prediction(
        df,
        **base_kwargs,
        low_confidence_secondary_enabled=bool(args.sec),
        low_confidence_secondary_cycles=int(args.sec_cycles),
        low_confidence_secondary_viscosity_multiplier=float(args.sec_visc),
        low_confidence_secondary_viscosity_anneal=bool(args.sec_visc_anneal),
        low_confidence_secondary_viscosity_multiplier_start=args.sec_visc_start,
        low_confidence_secondary_shear_multiplier=float(args.sec_shear),
        low_confidence_secondary_inhibition_multiplier=float(args.sec_inhib),
        low_confidence_secondary_promote_min_zone_votes=int(args.sec_promote_votes),
        low_confidence_secondary_promote_z_min=float(args.sec_promote_z),
        low_confidence_secondary_promote_conf_min=float(args.sec_promote_conf),
        low_confidence_secondary_sieve_enabled=bool(args.sec_sieve),
        low_confidence_secondary_sieve_cycles=int(args.sec_sieve_cycles),
        low_confidence_secondary_sieve_reverse_multiplier=float(args.sec_sieve_reverse),
        low_confidence_secondary_sieve_noise_std=float(args.sec_sieve_noise),
        low_confidence_secondary_sieve_instability_min=float(args.sec_sieve_inst),
        low_confidence_secondary_sieve_update_norm_max=float(args.sec_sieve_update_max),
    )
    dt2 = time.perf_counter() - t1
    _print_run("secondary_ionization", pred_sec, dt2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
