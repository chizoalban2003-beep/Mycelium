# PhysML — Physics-Inspired Machine Learning for Tabular Data

A standalone ML model that frames supervised learning as a **gel electrophoresis simulation**.
Features are treated as charged particles migrating through a viscous medium; their "charge"
is their statistical association with the target, and "viscosity" is modulated by collinearity,
distribution shape, and iterative PCR-style amplification.

## How It Works

```
Raw tabular data
      │
      ▼
  Cleaning & imputation  (rolling median, MAD, winsorize, …)
      │
      ▼
  Feature scoring        (Pearson/Spearman/Cramér-V / KL-divergence)
      │
      ▼
  Electrophoresis        (n_cycles × learning_rate updates, viscosity field)
      │
      ▼
  Bonding & complexes    (multicollinearity suppression)
      │
      ▼
  PCR amplification      (boost statistically significant features)
      │
      ▼
  PredictionResult       (test accuracy/R², feature weights, diagnostics)
```

## Install

```bash
pip install numpy pandas scipy scikit-learn
# optional: richer outlier cleaning
pip install feature-engine
```

## Quick Start

### scikit-learn API

```python
from physml import PhysicsPredictor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine(as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.25, random_state=42, stratify=wine.target
)

clf = PhysicsPredictor(plane="liquid", n_cycles=20)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
```

### Low-level functional API

```python
import pandas as pd
from physml import run_physics_prediction, PhysicsPlane

df = pd.read_csv("my_data.csv")
result = run_physics_prediction(
    df,
    target_col="price",          # or any classification column
    plane=PhysicsPlane.solid,
    n_cycles=30,
    return_predictions=True,
)
print(f"R²={result.metrics.rmse:.4f}  features={len(result.weights)}")
```

### Explicit train/test control

```python
import numpy as np

n = len(df)
mask = np.zeros(n, dtype=bool)
mask[:int(0.8*n)] = True           # first 80 % → train

result = run_physics_prediction(
    df,
    target_col="label",
    explicit_train_mask=mask,      # bypass random split
    n_cycles=25,
    return_predictions=True,
)
```

## Benchmark / Evaluation

Run the comprehensive benchmark to compare PhysML against RF, GBT, MLP, KNN, SVM, and more:

```bash
python evaluate.py                    # all tasks (classification + regression)
python evaluate.py --tasks classification
python evaluate.py --tasks regression
python evaluate.py --quick            # faster run with fewer cycles
python evaluate.py --output results.json
```

Baselines included:

- **Random Forest** (RF)
- **Extra Trees** (ET)
- **Gradient Boosting** (GB)
- **Histogram Gradient Boosting** (HGB)
- **Neural Network** (MLP)
- **K-Nearest Neighbours** (KNN)
- **SVM / SVR**
- **Logistic Regression / Ridge**
- **AdaBoost**

Datasets: iris, breast_cancer, wine (classification); diabetes, california_housing (regression).

## Package Structure

```
physml/
  __init__.py        Public API exports
  predictor.py       Core physics engine
  estimator.py       scikit-learn compatible PhysicsPredictor class
evaluate.py          Stand-alone benchmark script
tests/
  test_predictor.py  Physics engine unit tests
  test_estimator.py  Estimator / sklearn compatibility tests
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `plane` | `"liquid"` | Medium preset: `solid` / `liquid` / `gas` |
| `n_cycles` | 30 | Number of electrophoresis iterations |
| `cycle_learning_rate` | 0.18 | Per-cycle charge update rate |
| `cascade_enabled` | `True` | Multicollinearity complex suppression |
| `pcr_enabled` | `False` | PCR amplification of strong features |
| `enable_isotopes` | `True` | Auto-generate interaction features |
| `explicit_train_mask` | `None` | Override random split with boolean array |

## Running Tests

```bash
python -m pytest tests/ -q
```

## License

MIT
