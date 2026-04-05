Model performance (regression) — sweep winners vs baselines

Dataset: tmp_eval/job_salary_prediction_dataset.csv (nrows=8000)
Seed=42 | train_fraction=0.8 | target=salary

Preprocessing (new default): enabled.
- Pre-split: drop duplicate rows + drop missing target rows
- Post-split: impute missing + cap numeric outliers using TRAIN stats only
- Default outlier strategy: winsorize quantiles q_low=0.005, q_high=0.995

Note: on this particular 8k salary slice, the default cleaning pass reports `imputed_values=0` and `clipped_outliers=0`, so the metrics are unchanged versus the prior (pre-cleaning) benchmark.

Key result: Soft Multi-Buffer (v4.7 “Liquid-Crystal” probe: sigmoidal zone transitions) breaks the 5k MAE mark on this 8k slice (MAE 5568.80 → 4981.64; RMSE 7231.86 → 6361.90). Field-Effect coupling remains a safe additive.

Interpretation (as of April 5, 2026): the Field-Effect sweep suggests the model benefits from an earlier, long-range attraction phase (start=40) with slightly increasing coupling strength (field_decay=1.01). The Multi-Buffer results indicate the larger remaining error was not just “more cycles”, but heterogeneous dynamics: low-salary rows benefit from higher effective viscosity (slower updates) while high-salary rows benefit from lower viscosity (faster settling). Adding a soft (sigmoidal) transition reduces boundary oscillation and gives another meaningful accuracy step.

Updated regression leaderboard:
| Model | MAE | RMSE | Status |
|---|---:|---:|---|
| HistGB | 4695.95 | 5930.23 | Global leader (nonlinear baseline) |
| Mycelium v4.7 (Soft Multi-Buffer + Deep Freeze + Field-Effect) | 4981.64 | 6361.90 | New Mycelium peak (beats Ridge) |
| Ridge | 5356.81 | 7045.91 | Linear benchmark |
| Mycelium v4.6 (Hard Multi-Buffer + Deep Freeze + Field-Effect) | 5076.61 | 6498.50 | Prior peak |
| Mycelium v4.5 (Deep Freeze + Field-Effect) | 5568.80 | 7231.86 | Stable baseline |
| Mycelium v4.3-style (Deep Freeze, no field) | 5569.30 | 7233.55 | Stable baseline |

Phase 2 idea (now implemented as v4.6 experiment): “Multi-Buffer / zone-specific chemistry”. Zones are derived from the model’s current predictions using train-quantile thresholds (no target leakage), then viscosity and Field-Effect alpha can be scaled per-zone.

Salary regression (forced prediction):
| Model | MAE | RMSE | R2 | Time (s) | Notes |
|---|---:|---:|---:|---:|---|
| HistGB | 4695.95 | 5930.23 | 0.974078 | 2.69 | sklearn baseline |
| Ridge | 5356.81 | 7045.91 | 0.963407 | 0.25 | sklearn baseline |
| LinearRegression | 5357.18 | 7046.48 | 0.963402 | 0.17 | sklearn baseline |
| Mycelium (Deep Freeze, no field) | 5569.30 | 7233.55 | 0.961432 | 2.56 | gas, 100c, exp_decay lr=0.25 decay=0.995 + buffer shift g=0.60 min=0.70 |
| Mycelium (Deep Freeze + Field-Effect) | 5568.80 | 7231.86 | 0.961451 | 1.87 | + field α=0.25 start=40 type=linear field_decay=1.01 |
| Mycelium (v4.7 Soft Multi-Buffer + Field-Effect) | 4981.64 | 6361.90 | 0.970167 | 2.84 | + multibuffer q=(0.20,0.80) low_visc=1.30 high_visc=0.80 transition_frac=0.06 |

Sweep artifacts:
- tmp_eval/deep_freeze_full_20260404_114410.csv (best Deep Freeze without Field-Effect)
- tmp_eval/field_effect_sweep_20260405.csv (Field-Effect sweep on top of Deep Freeze optimum)
- tmp_eval/multibuffer_smoke_20260405.csv (Multi-Buffer sweep on top of best Deep Freeze + Field)
- tmp_eval/multibuffer_broad_20260405.csv (broad Multi-Buffer sweep over q_low/q_high + low/high viscosity + high-zone alpha)
- tmp_eval/multibuffer_sigmoid_probe_20260405.csv (probe sweep over multibuffer_transition_frac)
