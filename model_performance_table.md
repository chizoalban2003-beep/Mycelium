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
| keep ~38% (q=0.52 + ionized gate, ion_z_min=0.25) | 0.3840 | 0.3659 | 0.6160 | 0.1406 |
| keep ~30% (q=0.63 + ionized gate, ion_z_min=0.25) | 0.3020 | 0.3727 | 0.6980 | 0.1125 |
