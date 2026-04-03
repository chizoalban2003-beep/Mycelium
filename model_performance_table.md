Mycelium vs baselines (classification) — updated abstention + tuning

Dataset: tmp_eval/job_salary_prediction_dataset.csv (nrows=8000)
Target: remote_work | seed=42 | train_fraction=0.8

Forced prediction (apples-to-apples with sklearn):
| Model | Accuracy | F1 (macro) | Time (s) |
|---|---|---|---|
| Mycelium (tuned gas, n=50) | 0.3650 | 0.3642 | 8.33 |
| HistGB | 0.3606 | 0.3606 | 3.63 |
| KNN | 0.3538 | 0.3502 | 0.94 |
| Mycelium (default) | 0.3519 | 0.3510 | 6.94 |
| LogReg | 0.3431 | 0.3403 | 54.54 |
| RandomForest | 0.3425 | 0.3423 | 1.93 |
| MLP | 0.3394 | 0.1689 | 1.80 |
| DecisionTree | 0.3356 | 0.3355 | 0.17 |
| LinearSVC | 0.3275 | 0.1645 | 0.13 |
| ExtraTrees | 0.3125 | 0.3120 | 3.84 |

Selective prediction (Mycelium-only abstain):
| Mycelium Selective Mode (tuned) | Coverage | Selective Acc | Abstain Rate | Overall Acc |
|---|---|---|---|---|
| keep top 10% confidence (q=0.90) | 0.1000 | 0.4062 | 0.9000 | 0.0406 |
| keep ~43.4% (q=0.60 + ionized gate + secondary (4c, visc=0.65) + promote votes=3 @ conf≥0.45 + sieve v4.2 (shake=4, reverse=1.0, noise=0.12, inst≥0.50, confΔ≤0.003)) | 0.4337 | 0.3732 | 0.5663 | 0.1619 |
| keep ~38.6% (q=0.60 + ionized gate + secondary (3c, anneal visc 1.00→0.65) + promote votes=3 @ conf≥0.45) | 0.3856 | 0.3728 | 0.6144 | 0.1437 |
| keep ~38.4% (q=0.60 + ionized gate + secondary (3c, visc=0.75) + promote votes=3 @ conf≥0.45) | 0.3838 | 0.3730 | 0.6162 | 0.1431 |
| keep ~38% (q=0.52 + ionized gate, ion_z_min=0.25) | 0.3840 | 0.3659 | 0.6160 | 0.1406 |
| keep ~30% (q=0.63 + ionized gate, ion_z_min=0.25) | 0.3020 | 0.3727 | 0.6980 | 0.1125 |
