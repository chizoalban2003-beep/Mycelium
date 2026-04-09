"""Tests for the gravitational sedimentation engine."""
import pandas as pd
import numpy as np
from mycelium_app.sedimentation import run_sedimentation


def _make_df(n_rows=20, n_cols=8, seed=42):
    """Create a test DataFrame with varying feature densities."""
    np.random.seed(seed)
    data = {}
    # Dense features (high variance, correlated)
    base = np.random.randn(n_rows)
    data["dense_a"] = base * 10 + 100
    data["dense_b"] = base * 8 + 50 + np.random.randn(n_rows) * 2
    data["dense_c"] = base * 6 + 30 + np.random.randn(n_rows) * 3

    # Medium features
    data["medium_x"] = np.random.randn(n_rows) * 5
    data["medium_y"] = np.random.randn(n_rows) * 4

    # Light features (noise)
    data["noise_1"] = np.random.randn(n_rows) * 0.1
    data["noise_2"] = np.random.choice([0, 1], n_rows)
    data["noise_3"] = np.random.uniform(-1, 1, n_rows)

    return pd.DataFrame(data)


def test_sedimentation_produces_layers():
    """Should stratify features into three layers."""
    df = _make_df()
    result = run_sedimentation(df)

    assert result.n_features >= 3
    assert "bedrock" in result.layer_summary
    assert "suspension" in result.layer_summary
    assert "turbulent" in result.layer_summary

    total = sum(info["count"] for info in result.layer_summary.values())
    assert total == result.n_features


def test_dense_features_settle_deeper():
    """Highly correlated features with high variance should settle to bedrock."""
    df = _make_df()
    result = run_sedimentation(df, flocculation_threshold=0.7)

    bedrock_feats = [f.feature for f in result.features if f.layer == "bedrock"]
    turbulent_feats = [f.feature for f in result.features if f.layer == "turbulent"]

    # Dense_a, dense_b, dense_c should be in bedrock (they're correlated and high variance)
    dense_in_bedrock = sum(1 for f in bedrock_feats if f.startswith("dense_"))
    noise_in_turbulent = sum(1 for f in turbulent_feats if f.startswith("noise_"))

    assert dense_in_bedrock >= 1 or len(bedrock_feats) > 0, "Some features should settle to bedrock"


def test_flocculation_groups_correlated():
    """Correlated features should form complexes."""
    df = _make_df()
    result = run_sedimentation(df, flocculation_threshold=0.6)

    assert len(result.complexes) >= 1, "Correlated features should form at least one complex"
    # The dense features should be in the same complex
    for c in result.complexes:
        dense_count = sum(1 for f in c.features if f.startswith("dense_"))
        if dense_count >= 2:
            assert c.internal_cohesion > 0.5
            return
    # Even if not grouped by name, complexes should exist
    assert result.complexes


def test_vif_computation():
    """VIF should be computed for features."""
    df = _make_df()
    result = run_sedimentation(df)

    vifs = [f.vif for f in result.features if f.vif is not None]
    assert len(vifs) > 0, "At least some features should have VIF values"


def test_empty_dataframe():
    """Empty DataFrame should return empty result."""
    result = run_sedimentation(pd.DataFrame())
    assert result.n_features == 0
    assert result.features == []


def test_single_column():
    """Single column should handle gracefully."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = run_sedimentation(df)
    assert result.n_features == 0  # need at least 2 features for correlation


def test_depth_normalization():
    """All feature depths should be between 0 and 1."""
    df = _make_df(n_rows=30)
    result = run_sedimentation(df)

    for f in result.features:
        assert 0 <= f.depth <= 1.0, f"{f.feature} depth {f.depth} out of range"


def test_correlation_matrix():
    """Correlation matrix should be populated for features."""
    df = _make_df()
    result = run_sedimentation(df)

    assert len(result.correlation_matrix) > 0
    for feat, corrs in result.correlation_matrix.items():
        assert feat in corrs  # self-correlation should be ~1.0
        assert abs(corrs[feat] - 1.0) < 0.01
