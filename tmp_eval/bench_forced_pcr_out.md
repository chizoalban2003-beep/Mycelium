Dataset: tmp_eval/job_salary_prediction_dataset.csv (nrows=8000)
seed=42  train_fraction=0.8

Forced prediction (classification):
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
| Mycelium (sweep best: plane=gas, cycles=100, lr=0.25, shear=1.60) | 5568.86 | 7234.75 | 0.9614 | 2.19 |
| Mycelium (tuned gas, n=50) | 5581.93 | 7241.46 | 0.9613 | 2.46 |
| Mycelium (tuned gas, n=50, PCR 4c) | 5642.86 | 7385.33 | 0.9598 | 1.61 |
| Mycelium (sweep best: gas 100c, PCR 4c) | 5857.11 | 7741.72 | 0.9558 | 1.75 |
| RandomForest | 9983.86 | 12710.04 | 0.8809 | 6.59 |
| ExtraTrees | 12263.74 | 15182.33 | 0.8301 | 9.88 |
| DecisionTree | 14717.64 | 19139.22 | 0.7300 | 0.11 |
| KNN | 18440.47 | 22967.44 | 0.6112 | 0.17 |
| Dummy (mean) | 29473.24 | 36852.00 | -0.0010 | 0.05 |
