Mycelium vs baselines (classification + regression) — updated comparisons

Dataset: tmp_eval/job_salary_prediction_dataset.csv (nrows=8000)
Seed=42 | train_fraction=0.8

Note on PCR Stimulation: PCR 4c amplifies classification bands, increasing coverage (+4.4%) and selective accuracy (0.3739 → 0.4309) by reinforcing dominant categorical signals.
Regression is turbulence-sensitive: PCR 4c degrades MAE/RMSE; micro-PCR (1c, gain=0.20) slightly helps the tuned 50c run but still fails to improve the 100c sweep-best (and primer-threshold tightening did not help). Buffer shift (target-induced viscosity scaling) did not improve the 100c sweep-best by itself; on tuned 50c it roughly matches micro-PCR. Cooling (exp LR decay) + buffer shift produces a small but real improvement in the 100c regime; the full Deep Freeze sweep’s best-by-RMSE is recorded below. A v4.5 Field-Effect (covariance-weighted coupling in cycles 80+) improves slightly again; freezing LR during the coupling phase did not help.

Forced prediction (classification; apples-to-apples with sklearn):
Target: remote_work
| Model | Accuracy | F1 (macro) | Time (s) |
|---|---|---|---|
| LinearSVC | 0.4525 | 0.4357 | 0.44 |
| LogReg | 0.4519 | 0.4426 | 0.62 |
| Mycelium (extra: gas, 100c, PCR 4c) | 0.3831 | 0.3802 | 8.26 |
| Mycelium (tuned gas, n=50, PCR 4c) | 0.3775 | 0.3760 | 8.51 |
| HistGB | 0.3719 | 0.3717 | 3.83 |
| Mycelium (default, PCR 4c) | 0.3656 | 0.3649 | 5.49 |
| RandomForest | 0.3581 | 0.3582 | 4.40 |
| Mycelium (extra: gas, cycles=100, lr=0.25, shear=1.60) | 0.3581 | 0.3566 | 8.54 |
| Mycelium (tuned gas, n=50) | 0.3519 | 0.3515 | 9.26 |
| Mycelium (default) | 0.3519 | 0.3510 | 5.58 |
| ExtraTrees | 0.3481 | 0.3480 | 6.40 |
| DecisionTree | 0.3375 | 0.3377 | 0.13 |
| Dummy (most_frequent) | 0.3331 | 0.1666 | 0.05 |
| KNN | 0.3312 | 0.3296 | 0.32 |

Forced prediction (regression):
Target: salary
| Model | MAE | RMSE | R2 | Time (s) |
|---|---|---|---|---|
| HistGB | 5117.52 | 6460.61 | 0.9692 | 0.79 |
| Ridge | 5353.88 | 7042.21 | 0.9634 | 0.20 |
| Mycelium (v4.5 field-effect: gas 100c, exp_decay lr=0.25, decay=0.995 + buffer shift g=0.60 min=0.70 + field α=0.10 @ cycle≥80) | 5566.94 | 7232.28 | 0.9614 | 2.13 |
| Mycelium (deep-freeze best: gas 100c, exp_decay lr=0.25, decay=0.995 + buffer shift g=0.60 min=0.70) | 5567.26 | 7232.82 | 0.9614 | 3.13 |
| Mycelium (sweep best: plane=gas, cycles=100, lr=0.25, shear=1.60) | 5568.86 | 7234.75 | 0.9614 | 2.19 |
| Mycelium (tuned gas, n=50) | 5581.93 | 7241.46 | 0.9613 | 2.46 |
| Mycelium (tuned gas, n=50, micro-PCR 1c, gain=0.20) | 5567.53 | 7231.50 | 0.9615 | 1.32 |
| Mycelium (tuned gas, n=50, buffer shift gain=0.50, min=0.75) | 5567.49 | 7232.27 | 0.9614 | 1.32 |
| Mycelium (tuned gas, n=50, PCR 4c) | 5642.86 | 7385.33 | 0.9598 | 1.61 |
| Mycelium (sweep best: gas 100c, PCR 4c) | 5857.11 | 7741.72 | 0.9558 | 1.75 |
| RandomForest | 9983.86 | 12710.04 | 0.8809 | 6.59 |
| ExtraTrees | 12263.74 | 15182.33 | 0.8301 | 9.88 |
| DecisionTree | 14717.64 | 19139.22 | 0.7300 | 0.11 |
| KNN | 18440.47 | 22967.44 | 0.6112 | 0.17 |
| Dummy (mean) | 29473.24 | 36852.00 | -0.0010 | 0.05 |

Selective prediction (Mycelium-only abstain):
| Mycelium Selective Mode (tuned) | Coverage | Selective Acc | Abstain Rate | Overall Acc |
|---|---|---|---|---|
| keep top ~10% confidence (q=0.90) | 0.0787 | 0.3730 | 0.9213 | 0.0294 |
| keep ~42.1% (q=0.60 + ionized gate + secondary 4c (visc=0.65) + promote votes=3 @ conf≥0.45 + sieve v4.2) | 0.4213 | 0.3739 | 0.5787 | 0.1575 |
| keep ~46.6% (PCR 4c + q=0.60 + ionized gate + secondary 4c (visc=0.65) + promote votes=3 @ conf≥0.45 + sieve v4.2) | 0.4656 | 0.4309 | 0.5344 | 0.2006 |
| keep ~38.7% (q=0.60 + ionized gate + secondary (3c, anneal visc 1.00→0.65) + promote votes=3 @ conf≥0.45) | 0.3869 | 0.3716 | 0.6131 | 0.1437 |
| keep ~38.5% (q=0.60 + ionized gate + secondary (3c, visc=0.75) + promote votes=3 @ conf≥0.45) | 0.3850 | 0.3718 | 0.6150 | 0.1431 |
| keep ~38.4% (q=0.52 + ionized gate, ion_z_min=0.25) | 0.3844 | 0.3496 | 0.6156 | 0.1344 |
| keep ~29.4% (q=0.63 + ionized gate, ion_z_min=0.25) | 0.2937 | 0.3723 | 0.7063 | 0.1094 |
