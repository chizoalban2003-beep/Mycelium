Mycelium vs baselines (classification + regression) — updated comparisons

Dataset: tmp_eval/job_salary_prediction_dataset.csv (nrows=8000)
Seed=42 | train_fraction=0.8

Forced prediction (classification; apples-to-apples with sklearn):
Target: remote_work
| Model | Accuracy | F1 (macro) | Time (s) |
|---|---|---|---|
| LinearSVC | 0.4525 | 0.4357 | 0.21 |
| LogReg | 0.4519 | 0.4426 | 0.24 |
| HistGB | 0.3719 | 0.3717 | 2.04 |
| RandomForest | 0.3581 | 0.3582 | 3.95 |
| Mycelium (tuned gas, n=50) | 0.3519 | 0.3515 | 8.37 |
| Mycelium (default) | 0.3519 | 0.3510 | 6.84 |
| ExtraTrees | 0.3481 | 0.3480 | 6.56 |
| DecisionTree | 0.3375 | 0.3377 | 0.11 |
| Dummy (most_frequent) | 0.3331 | 0.1666 | 0.06 |
| KNN | 0.3312 | 0.3296 | 0.39 |

Forced prediction (regression):
Target: salary
| Model | MAE | RMSE | R2 | Time (s) |
|---|---|---|---|---|
| HistGB | 5117.52 | 6460.61 | 0.9692 | 1.35 |
| Ridge | 5353.88 | 7042.21 | 0.9634 | 0.35 |
| Mycelium (tuned gas, n=50) | 5581.93 | 7241.46 | 0.9613 | 2.28 |
| RandomForest | 9983.86 | 12710.04 | 0.8809 | 6.69 |
| ExtraTrees | 12263.74 | 15182.33 | 0.8301 | 9.13 |
| DecisionTree | 14717.64 | 19139.22 | 0.7300 | 0.10 |
| KNN | 18440.47 | 22967.44 | 0.6112 | 0.17 |
| Dummy (mean) | 29473.24 | 36852.00 | -0.0010 | 0.11 |

Selective prediction (Mycelium-only abstain):
| Mycelium Selective Mode (tuned) | Coverage | Selective Acc | Abstain Rate | Overall Acc |
|---|---|---|---|---|
| keep top 10% confidence (q=0.90) | 0.1000 | 0.4062 | 0.9000 | 0.0406 |
| keep ~43.4% (q=0.60 + ionized gate + secondary (4c, visc=0.65) + promote votes=3 @ conf≥0.45 + sieve v4.2 (shake=4, reverse=1.0, noise=0.12, inst≥0.50, confΔ≤0.003)) | 0.4337 | 0.3732 | 0.5663 | 0.1619 |
| keep ~38.6% (q=0.60 + ionized gate + secondary (3c, anneal visc 1.00→0.65) + promote votes=3 @ conf≥0.45) | 0.3856 | 0.3728 | 0.6144 | 0.1437 |
| keep ~38.4% (q=0.60 + ionized gate + secondary (3c, visc=0.75) + promote votes=3 @ conf≥0.45) | 0.3838 | 0.3730 | 0.6162 | 0.1431 |
| keep ~38% (q=0.52 + ionized gate, ion_z_min=0.25) | 0.3840 | 0.3659 | 0.6160 | 0.1406 |
| keep ~30% (q=0.63 + ionized gate, ion_z_min=0.25) | 0.3020 | 0.3727 | 0.6980 | 0.1125 |
