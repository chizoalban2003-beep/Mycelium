Model performance (regression) — sweep winners vs baselines

Dataset: tmp_eval/job_salary_prediction_dataset.csv (nrows=8000)
Seed=42 | train_fraction=0.8 | target=salary

Key result: Field-Effect coupling is a safe additive in the Deep Freeze regime; best-by-RMSE improves from 7232.82 → 7231.07 with no RMSE blow-up.

Interpretation (as of April 5, 2026): the Field-Effect sweep suggests the model benefits from an earlier, long-range attraction phase (start=40) with slightly increasing coupling strength (field_decay=1.01), but we are likely nearing the performance floor of the current single-plane “Gas + global buffer” physics on this 8k slice.

Updated regression leaderboard:
| Model | MAE | RMSE | Status |
|---|---:|---:|---|
| HistGB | 4695.95 | 5930.23 | Global leader (nonlinear baseline) |
| Ridge | 5356.81 | 7045.91 | Linear benchmark |
| Mycelium v4.5 (Deep Freeze + Field-Effect best-by-RMSE) | 5566.60 | 7231.07 | New Mycelium peak |
| Mycelium v4.3-style (Deep Freeze best-by-RMSE, no field) | 5567.26 | 7232.82 | Stable baseline |

Phase 2 idea (optional next experiment): “Multi-Buffer / zone-specific chemistry”. Instead of one global viscosity/coupling schedule, make viscosity and/or coupling depend on target-density zones (e.g., low-salary vs high-salary regimes) so the system can behave differently where the mapping is more nonlinear.

Salary regression (forced prediction):
| Model | MAE | RMSE | R2 | Time (s) | Notes |
|---|---:|---:|---:|---:|---|
| HistGB | 4695.95 | 5930.23 | 0.974078 | 2.69 | sklearn baseline |
| Ridge | 5356.81 | 7045.91 | 0.963407 | 0.25 | sklearn baseline |
| LinearRegression | 5357.18 | 7046.48 | 0.963402 | 0.17 | sklearn baseline |
| Mycelium (Deep Freeze best-by-RMSE) | 5567.26 | 7232.82 | 0.961440 | 2.40 | gas, 100c, exp_decay lr=0.25 decay=0.995 + buffer shift g=0.60 min=0.70 |
| Mycelium (Deep Freeze + Field-Effect best-by-RMSE) | 5566.60 | 7231.07 | 0.961459 | 2.20 | + field α=0.25 start=40 type=linear field_decay=1.01 |

Sweep artifacts:
- tmp_eval/deep_freeze_full_20260404_114410.csv (best Deep Freeze without Field-Effect)
- tmp_eval/field_effect_sweep_20260405.csv (Field-Effect sweep on top of Deep Freeze optimum)
