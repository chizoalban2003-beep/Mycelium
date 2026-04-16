# How the Physics Works

PhysML frames supervised learning as a **gel electrophoresis simulation**.
Each feature is a charged particle migrating through a viscous medium toward
a target electrode.

## Core Analogy

| Physics concept | ML concept |
|---|---|
| Particle charge | Feature–target association (Pearson/Spearman/Cramér-V) |
| Viscosity field | Collinearity, distribution skewness |
| Migration velocity | Feature weight update per cycle |
| Bonding / complexes | Multicollinearity suppression |
| PCR amplification | Boosting statistically significant features |

## Pipeline

```
Raw tabular data
      │
      ▼
  Cleaning & imputation  (rolling median, MAD, winsorize)
      │
      ▼
  Feature scoring        (Pearson/Spearman/Cramér-V / KL-divergence)
      │
      ▼
  Electrophoresis        (n_cycles × learning_rate updates, viscosity)
      │
      ▼
  Bonding & complexes    (multicollinearity suppression)
      │
      ▼
  PCR amplification      (boost statistically significant features)
      │
      ▼
  PredictionResult       (accuracy/R², feature weights, diagnostics)
```

## Cycle Dynamics

In each electrophoresis cycle the feature weight vector **w** is updated:

```
w_{t+1} = w_t + lr × (charge_i / viscosity_i) × error_t
```

where:
- `charge_i` is the statistical association score of feature *i*
- `viscosity_i` is raised by collinearity with already-strong features
- `error_t` is the residual from the current ensemble prediction

After all cycles, features whose weight did not clear a minimum threshold are
suppressed (bonded into a "complex"), and the top surviving features are
PCR-amplified.

## Uncertainty Quantification

The neural backend exposes softmax probabilities.  The agent uses
**prediction entropy** as the primary uncertainty signal:

```
H = -Σ p_k log p_k
```

When `query_strategy="gp"`, a sparse Gaussian process is fitted on the
labelled set; the acquisition function switches to **GP predictive variance**,
which is better calibrated for very small labelled sets.

When `policy="ensemble"`, five bootstrap copies of the MLP are trained and
**committee disagreement** (vote entropy across members) is used as the
ask-signal.  This is orthogonal to the entropy measure and can be combined
with drift detection.

## Temperature Calibration

After fitting, a temperature scalar *T* is estimated on a 20% held-out split
by minimising **Expected Calibration Error (ECE)** via 1-D optimisation.
Softmax logits are then divided by *T* before converting to probabilities,
correcting the overconfidence typical of MLP classifiers.

## Adaptive Threshold Policy

The agent maintains a sliding window of recent prediction errors.  When the
rolling error rate rises, the ask-threshold is lowered (agent asks more); when
it falls, the threshold is raised (agent relies on itself more).  This
homeostatic mechanism keeps the oracle call rate proportional to actual
uncertainty.

## Drift Detection

PhysML supports two drift-detection algorithms:

- **Page-Hinkley** — sequential change-point detection on the running mean of
  prediction residuals.  Sensitive to gradual drift.
- **ADWIN** — adaptive windowing; adjusts the window size automatically and
  is more robust to abrupt concept shifts.

When drift is detected, the agent resets its homeostasis state and emits a
burst of oracle queries to relabel the shifted distribution quickly.
