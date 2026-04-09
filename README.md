# Myco — Grow with Data

A local-first, physics-inspired prediction engine and digital companion for tabular, sensor, and electricity data.

It frames feature selection as a physical system: strong signals migrate, weak signals slow down, correlated features bond, and noisy measurements are cleaned without erasing real transients.

## At a Glance

- **Explainable** feature scoring and selection
- **Transient-aware** cleaning for noisy signal streams
- **Memory-friendly** dataclasses with `slots=True`
- **Diagnostics-rich** output for debugging and visualization
- **Local-first** workflow that runs comfortably in VS Code

## Architecture

```mermaid
flowchart LR
    A[Raw electricity / sensor data] --> B[_clean_dataframe_for_prediction()]
    B --> C[Rolling sedimentation\nmedian + MAD + cadence inference]
    C --> D[Feature scoring\nWeightInfo + KL divergence]
    D --> E[Migration field\nviscosity + mass + terminal velocity]
    E --> F[Bonding + complexes\nBondInfo + EquilibriumZone]
    F --> G[PCR amplification\n_pcr_amplification_factor()]
    G --> H[PredictionResult]
    H --> I[Diagnostics + visualization]

    C --> C1[Warm-up backfill\nfirst stable median]
    C --> C2[Cadence inference\nDatetimeIndex or time column]
    E --> E1[Viscosity field\n_calculate_viscosity_field()]
    G --> G1[Selective amplification\nstable features only when enabled]
```

## Code Map

The diagram maps directly to these implementation points in `mycelium_app/physics_predictor.py`:

- **Fluid / cleaning layer**
  - `_clean_dataframe_for_prediction(...)`
  - rolling-window sedimentation helper
  - cadence inference diagnostics in `PredictionResult.diagnostics`

- **Gravity / weighting layer**
  - `WeightInfo`
  - feature scoring inside `run_physics_prediction(...)`

- **Viscosity / migration layer**
  - `_calculate_viscosity_field(...)`
  - `_migration_state(...)`
  - `MigrationInfo`

- **Bonding / manifold layer**
  - `BondInfo`
  - `EquilibriumZone`
  - `_build_bonding_map(...)`
  - `_collinearity_complexes(...)`
  - `_build_equilibrium_zones(...)`

- **PCR / amplification layer**
  - `_pcr_amplification_factor(...)`
  - PCR logic inside `run_physics_prediction(...)`
  - `MigrationInfo.terminal_velocity`
  - `PredictionMetrics.gel_band_sharpness`
  - `PredictionMetrics.gel_smearing`

## Workflow

The engine follows a simple sequence:

- **Clean** the stream without destroying transients.
- **Score** each feature against the target.
- **Move** features through the viscosity field.
- **Group** correlated features into complexes.
- **Amplify** statistically useful signals with PCR.
- **Report** the result with diagnostics for visualization.

## Data Cleaning Strategy

The cleaner supports two modes:

- **Static outlier handling**
  - winsorize
  - IQR
  - Gaussian
  - MAD
  - arbitrary clipping

- **Rolling-window sedimentation**
  - rolling median + rolling MAD
  - preserves clustered transients
  - clips isolated spikes
  - infers cadence from datetime-like data when rolling mode is enabled
  - uses a stable warm-up backfill at the start of the stream

This is especially useful for electricity data, where startup spikes and device on/off events should often be preserved rather than treated as noise.

### High-Rate Knobs

When rolling mode infers cadence automatically, the cleaner uses sensible defaults:

- **High-rate / waveform data:** short transient window by default
- **Low-rate / telemetry data:** longer stability window by default

These defaults are exposed through the cleaning parameters, so you can tune them without manually converting cadence into a window size.

## Configuration Reference

The most useful knobs for day-to-day runs are:

### Cleaning

- `cleaning_outlier_strategy` — `winsorize`, `iqr`, `gaussian`, `mad`, `arbitrary`, `feature_engine`, `rolling`, or `none`
- `cleaning_rolling_window` — explicit rolling window size when you already know the cadence
- `cleaning_rolling_window_cadence_hz` — inferred cadence override for rolling mode
- `cleaning_rolling_window_seconds` — time span used when cadence is inferred automatically
- `cleaning_rolling_mad_fold` — robust clipping strength for rolling median/MAD filtering
- `cleaning_rolling_cluster_min_size` — minimum cluster size before a deviation is preserved as a transient

### PCR / Amplification

- `pcr_enabled` — turns selective amplification on or off
- `pcr_cycles` — number of amplification cycles to apply
- `pcr_pvalue_threshold` — primer-binding threshold for target affinity
- `pcr_gain` — amplification strength per cycle
- `pcr_require_stable` — gates amplification to stable features only

### Practical Defaults

- Use `rolling` for electricity or sensor streams with timestamps.
- Use `winsorize` if you want simple, global clipping.
- Keep `pcr_enabled=False` until you want selective amplification of high-affinity features.
- Start with a short rolling window for waveform data and a longer one for telemetry.

## Hardware-Friendly Design

The core dataclasses use `slots=True` to reduce per-instance memory overhead:

- `WeightInfo`
- `MigrationInfo`
- `PredictionMetrics`
- `BondInfo`
- `IterationInfo`
- `EquilibriumZone`
- `PredictionResult`

That makes the feature-tracking layer lighter when you have many harmonics, phases, or derived signal features.

## Quick Start

Run the demo starter:

```bash
/home/chizoalban2003/Mycelium/.venv/bin/python crew_ai_starter.py
```

Or call the engine directly from Python:

```python
import pandas as pd
from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction

df = pd.DataFrame(
    {
        "study_hours": [1.0, 2.5, 3.5, 4.0, 5.5, 6.0],
        "sleep_hours": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
        "target": [52, 58, 63, 68, 74, 79],
    }
)

result = run_physics_prediction(
    df,
    target_col="target",
    plane=PhysicsPlane.solid,
    train_fraction=0.67,
    random_seed=42,
    n_cycles=5,
)

print(result.metrics)
```

## Benchmarking

Run the built-in benchmark suite to compare the physics predictor against ensemble baselines and unsupervised clustering models:

```bash
/home/chizoalban2003/Mycelium/.venv/bin/python benchmark_models.py
```

The script reports:

- classification accuracy and F1 against random forest and extra-trees baselines
- regression MAE/RMSE against random forest and extra-trees baselines
- unsupervised clustering quality on no-target synthetic datasets
- runtime-state serialization / reload smoke test

The benchmark suite also includes a permanent ENB2012 / Energy Efficiency sweep that:

- evaluates Heating Load (`Y1`) and Cooling Load (`Y2`) separately
- sweeps `PhysicsPlane.solid`, `PhysicsPlane.liquid`, and `PhysicsPlane.gas`
- compares baseline physics runs against the `Viscosity Squeeze` variant
- records ecosystem vitals such as mean viscosity, terminal velocity, and active complexes

This makes ENB2012 a repeatable wind tunnel for checking whether future physics changes help on structured building-energy tabular data.

## Suggested Visual Legend

If you want to reuse the diagram as documentation, this legend matches the visual panels:

- **Top-left:** correlation structure and viscosity field
- **Top-right:** weighted feature influence and blending points
- **Bottom-left:** manifold dynamics, sedimentation, and turbulence
- **Bottom-right:** electrophoresis, PCR amplification, and iterative convergence

## Notes

- SciPy is used when available for statistical routines.
- The engine is designed to be explainable, not just predictive.
- Diagnostics are surfaced in `PredictionResult.diagnostics` for debugging and visualization.
- Runtime state helpers support serialization, pruning, homeostatic gain updates, and abstention-aware reruns.
- The ENB2012 benchmark is tracked as a permanent regression/plane-sweep fixture.
- The repository currently includes a small CrewAI starter script for demo runs and narrative summaries.
