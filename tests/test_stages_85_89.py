"""Tests for Stages 85–89:
  GraphLearner, ClusterEngine, BandpassFilter, DataValidator, PipelineBuilder.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clf_data(n=200, n_features=8, random_state=0):
    n_informative = min(4, n_features - 1)
    n_redundant = min(2, n_features - n_informative - 1)
    return make_classification(
        n_samples=n, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, random_state=random_state,
    )


def _reg_data(n=150, n_features=6, random_state=0):
    return make_regression(
        n_samples=n, n_features=n_features, noise=5.0, random_state=random_state,
    )


def _correlated_X(n=200, n_features=6, rng_seed=0):
    """Generate data with intentional inter-feature correlations."""
    rng = np.random.default_rng(rng_seed)
    base = rng.standard_normal((n, 2))
    cols = [base[:, 0]]
    for i in range(1, n_features):
        noise = rng.standard_normal(n) * 0.1
        cols.append(base[:, i % 2] + noise)
    return np.column_stack(cols)


# ===========================================================================
# Stage 85 — GraphLearner
# ===========================================================================

class TestGraphLearner:

    def test_import(self):
        from physml.graph_learner import GraphLearner, GraphResult
        assert GraphLearner
        assert GraphResult

    def test_graph_result_fields(self):
        from physml.graph_learner import GraphResult
        r = GraphResult(feature_a=0, feature_b=3, weight=0.75)
        assert r.feature_a == 0
        assert r.feature_b == 3
        assert r.weight == pytest.approx(0.75)

    def test_graph_result_as_dict(self):
        from physml.graph_learner import GraphResult
        r = GraphResult(feature_a=1, feature_b=2, weight=-0.5)
        d = r.as_dict()
        assert d["feature_a"] == 1
        assert d["feature_b"] == 2
        assert d["weight"] == pytest.approx(-0.5)

    def test_graph_result_repr(self):
        from physml.graph_learner import GraphResult
        r = GraphResult(feature_a=0, feature_b=1, weight=0.3)
        assert "GraphResult" in repr(r)

    def test_fit_basic(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X()
        gl = GraphLearner(threshold=0.3)
        result = gl.fit(X)
        assert result is gl  # returns self
        assert gl._fitted

    def test_n_features_after_fit(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X(n_features=6)
        gl = GraphLearner()
        gl.fit(X)
        assert gl.n_features == 6

    def test_edges_found_on_correlated_data(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X()
        gl = GraphLearner(threshold=0.3)
        gl.fit(X)
        assert gl.n_edges > 0

    def test_high_threshold_fewer_edges(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X()
        gl_low = GraphLearner(threshold=0.1).fit(X)
        gl_high = GraphLearner(threshold=0.9).fit(X)
        assert gl_low.n_edges >= gl_high.n_edges

    def test_get_graph_returns_list(self):
        from physml.graph_learner import GraphLearner, GraphResult
        X = _correlated_X()
        gl = GraphLearner(threshold=0.3).fit(X)
        edges = gl.get_graph()
        assert isinstance(edges, list)
        assert all(isinstance(e, GraphResult) for e in edges)

    def test_get_graph_sorted_by_weight(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X()
        gl = GraphLearner(threshold=0.1).fit(X)
        edges = gl.get_graph()
        weights = [abs(e.weight) for e in edges]
        assert weights == sorted(weights, reverse=True)

    def test_most_connected_returns_list(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X()
        gl = GraphLearner(threshold=0.1).fit(X)
        top = gl.most_connected(n=3)
        assert isinstance(top, list)
        assert len(top) <= 3

    def test_most_connected_all_valid_indices(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X(n_features=6)
        gl = GraphLearner(threshold=0.1).fit(X)
        for idx in gl.most_connected(n=6):
            assert 0 <= idx < 6

    def test_threshold_zero_finds_all_pairs(self):
        from physml.graph_learner import GraphLearner
        X = _correlated_X(n_features=4)
        gl = GraphLearner(threshold=0.0).fit(X)
        max_edges = 4 * 3 // 2
        assert gl.n_edges <= max_edges

    def test_invalid_threshold_raises(self):
        from physml.graph_learner import GraphLearner
        with pytest.raises(ValueError):
            GraphLearner(threshold=1.5)

    def test_not_fitted_raises(self):
        from physml.graph_learner import GraphLearner
        gl = GraphLearner()
        with pytest.raises(RuntimeError):
            gl.get_graph()

    def test_repr(self):
        from physml.graph_learner import GraphLearner
        gl = GraphLearner(threshold=0.4)
        assert "GraphLearner" in repr(gl)
        assert "0.4" in repr(gl)


# ===========================================================================
# Stage 86 — ClusterEngine
# ===========================================================================

class TestClusterEngine:

    def test_import(self):
        from physml.cluster_engine import ClusterEngine, ClusterReport
        assert ClusterEngine
        assert ClusterReport

    def test_cluster_report_fields(self):
        from physml.cluster_engine import ClusterReport
        r = ClusterReport(n_clusters=3, inertia=100.0, silhouette_score=0.5)
        assert r.n_clusters == 3
        assert r.inertia == pytest.approx(100.0)
        assert r.silhouette_score == pytest.approx(0.5)

    def test_cluster_report_as_dict(self):
        from physml.cluster_engine import ClusterReport
        r = ClusterReport(n_clusters=2, inertia=50.0, silhouette_score=0.4)
        d = r.as_dict()
        assert d["n_clusters"] == 2
        assert "inertia" in d
        assert "silhouette_score" in d

    def test_cluster_report_repr(self):
        from physml.cluster_engine import ClusterReport
        r = ClusterReport(n_clusters=3, inertia=10.0, silhouette_score=0.6)
        assert "ClusterReport" in repr(r)

    def test_fit_basic(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data()
        ce = ClusterEngine(n_clusters=3)
        result = ce.fit(X)
        assert result is ce
        assert ce._fitted

    def test_labels_shape(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data(n=100)
        ce = ClusterEngine(n_clusters=3).fit(X)
        assert ce.labels_.shape == (100,)

    def test_labels_values_in_range(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data()
        ce = ClusterEngine(n_clusters=4).fit(X)
        assert set(ce.labels_).issubset(set(range(4)))

    def test_predict_returns_labels(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data()
        ce = ClusterEngine(n_clusters=3).fit(X)
        preds = ce.predict(X[:20])
        assert preds.shape == (20,)
        assert set(preds).issubset(set(range(3)))

    def test_cluster_centers_shape(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data(n_features=8)
        ce = ClusterEngine(n_clusters=3).fit(X)
        assert ce.cluster_centers_.shape == (3, 8)

    def test_report_returns_correct_type(self):
        from physml.cluster_engine import ClusterEngine, ClusterReport
        X, _ = _clf_data()
        ce = ClusterEngine(n_clusters=3).fit(X)
        r = ce.report()
        assert isinstance(r, ClusterReport)

    def test_report_n_clusters(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data()
        ce = ClusterEngine(n_clusters=4).fit(X)
        r = ce.report()
        assert r.n_clusters == 4

    def test_report_inertia_positive(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data()
        ce = ClusterEngine(n_clusters=3).fit(X)
        assert ce.report().inertia > 0

    def test_report_silhouette_in_range(self):
        from physml.cluster_engine import ClusterEngine
        X, _ = _clf_data()
        ce = ClusterEngine(n_clusters=3).fit(X)
        r = ce.report()
        if not math.isnan(r.silhouette_score):
            assert -1.0 <= r.silhouette_score <= 1.0

    def test_invalid_n_clusters_raises(self):
        from physml.cluster_engine import ClusterEngine
        with pytest.raises(ValueError):
            ClusterEngine(n_clusters=0)

    def test_not_fitted_raises(self):
        from physml.cluster_engine import ClusterEngine
        ce = ClusterEngine()
        with pytest.raises(RuntimeError):
            ce.report()

    def test_repr(self):
        from physml.cluster_engine import ClusterEngine
        ce = ClusterEngine(n_clusters=5)
        assert "ClusterEngine" in repr(ce)
        assert "5" in repr(ce)


# ===========================================================================
# Stage 87 — BandpassFilter
# ===========================================================================

class TestBandpassFilter:

    def test_import(self):
        from physml.bandpass_filter import BandpassFilter, FilterResult
        assert BandpassFilter
        assert FilterResult

    def test_filter_result_fields(self):
        from physml.bandpass_filter import FilterResult
        r = FilterResult(n_original=10, n_kept=6, kept_indices=[0, 2, 3, 5, 7, 9])
        assert r.n_original == 10
        assert r.n_kept == 6
        assert len(r.kept_indices) == 6

    def test_filter_result_as_dict(self):
        from physml.bandpass_filter import FilterResult
        r = FilterResult(n_original=5, n_kept=3, kept_indices=[0, 2, 4])
        d = r.as_dict()
        assert d["n_original"] == 5
        assert d["n_kept"] == 3
        assert d["kept_indices"] == [0, 2, 4]

    def test_filter_result_repr(self):
        from physml.bandpass_filter import FilterResult
        r = FilterResult(n_original=8, n_kept=4, kept_indices=[0, 1, 2, 3])
        assert "FilterResult" in repr(r)

    def test_fit_basic(self):
        from physml.bandpass_filter import BandpassFilter
        X, _ = _clf_data()
        bf = BandpassFilter(low_var=0.0)
        result = bf.fit(X)
        assert result is bf
        assert bf._fitted

    def test_transform_shape(self):
        from physml.bandpass_filter import BandpassFilter
        X, _ = _clf_data(n=100, n_features=8)
        bf = BandpassFilter(low_var=0.0).fit(X)
        X_t = bf.transform(X)
        assert X_t.ndim == 2
        assert X_t.shape[0] == 100

    def test_fit_transform_equivalent(self):
        from physml.bandpass_filter import BandpassFilter
        X, _ = _clf_data()
        bf1 = BandpassFilter(low_var=0.1)
        bf2 = BandpassFilter(low_var=0.1)
        X_t1 = bf1.fit_transform(X)
        bf2.fit(X)
        X_t2 = bf2.transform(X)
        np.testing.assert_array_equal(X_t1, X_t2)

    def test_constant_feature_removed(self):
        from physml.bandpass_filter import BandpassFilter
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        X[:, 2] = 3.14  # constant
        bf = BandpassFilter(low_var=1e-10)
        bf.fit(X)
        assert 2 not in bf.kept_indices_

    def test_high_var_removes_noisy_feature(self):
        from physml.bandpass_filter import BandpassFilter
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        X[:, 0] = rng.standard_normal(200) * 100  # very high variance
        bf = BandpassFilter(low_var=0.0, high_var=10.0)
        bf.fit(X)
        assert 0 not in bf.kept_indices_

    def test_result_method(self):
        from physml.bandpass_filter import BandpassFilter, FilterResult
        X, _ = _clf_data()
        bf = BandpassFilter(low_var=0.0).fit(X)
        r = bf.result()
        assert isinstance(r, FilterResult)
        assert r.n_original == X.shape[1]
        assert r.n_kept == len(bf.kept_indices_)

    def test_variances_property(self):
        from physml.bandpass_filter import BandpassFilter
        X, _ = _clf_data(n_features=6)
        bf = BandpassFilter().fit(X)
        assert bf.variances_.shape == (6,)
        assert np.all(bf.variances_ >= 0)

    def test_invalid_low_var(self):
        from physml.bandpass_filter import BandpassFilter
        with pytest.raises(ValueError):
            BandpassFilter(low_var=-0.1)

    def test_invalid_high_var_lt_low_var(self):
        from physml.bandpass_filter import BandpassFilter
        with pytest.raises(ValueError):
            BandpassFilter(low_var=1.0, high_var=0.5)

    def test_not_fitted_raises(self):
        from physml.bandpass_filter import BandpassFilter
        with pytest.raises(RuntimeError):
            BandpassFilter().transform(np.ones((10, 3)))

    def test_all_filtered_empty_array(self):
        from physml.bandpass_filter import BandpassFilter
        X = np.ones((20, 4))  # all constant
        bf = BandpassFilter(low_var=1e-6).fit(X)
        X_t = bf.transform(X)
        assert X_t.shape == (20, 0)

    def test_repr(self):
        from physml.bandpass_filter import BandpassFilter
        bf = BandpassFilter(low_var=0.1, high_var=5.0)
        assert "BandpassFilter" in repr(bf)
        assert "0.1" in repr(bf)


# ===========================================================================
# Stage 88 — DataValidator
# ===========================================================================

class TestDataValidator:

    def test_import(self):
        from physml.data_validator import DataValidator, ValidationReport
        assert DataValidator
        assert ValidationReport

    def test_validation_report_fields(self):
        from physml.data_validator import ValidationReport
        r = ValidationReport(
            n_rows=100, n_cols=5, missing_count=0,
            constant_features=[], duplicate_rows=0,
            infinite_count=0, is_valid=True,
        )
        assert r.n_rows == 100
        assert r.n_cols == 5
        assert r.is_valid is True

    def test_validation_report_as_dict(self):
        from physml.data_validator import ValidationReport
        r = ValidationReport(
            n_rows=50, n_cols=3, missing_count=2,
            constant_features=[1], duplicate_rows=5,
            infinite_count=1, is_valid=False,
        )
        d = r.as_dict()
        assert d["missing_count"] == 2
        assert d["is_valid"] is False

    def test_validation_report_repr(self):
        from physml.data_validator import ValidationReport
        r = ValidationReport(10, 3, 0, [], 0, 0, True)
        assert "ValidationReport" in repr(r)

    def test_validate_clean_data(self):
        from physml.data_validator import DataValidator
        X, _ = _clf_data()
        dv = DataValidator()
        r = dv.validate(X)
        assert r.missing_count == 0
        assert r.infinite_count == 0
        assert r.is_valid is True

    def test_validate_shape(self):
        from physml.data_validator import DataValidator
        X, _ = _clf_data(n=120, n_features=7)
        r = DataValidator().validate(X)
        assert r.n_rows == 120
        assert r.n_cols == 7

    def test_detects_missing_values(self):
        from physml.data_validator import DataValidator
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        X[0, 1] = np.nan
        X[3, 2] = np.nan
        r = DataValidator().validate(X)
        assert r.missing_count == 2
        assert r.is_valid is False

    def test_detects_infinite_values(self):
        from physml.data_validator import DataValidator
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        X[5, 0] = np.inf
        r = DataValidator().validate(X)
        assert r.infinite_count >= 1
        assert r.is_valid is False

    def test_detects_constant_features(self):
        from physml.data_validator import DataValidator
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 5))
        X[:, 2] = 0.0
        r = DataValidator().validate(X)
        assert 2 in r.constant_features

    def test_detects_duplicate_rows(self):
        from physml.data_validator import DataValidator
        rng = np.random.default_rng(0)
        X_unique = rng.standard_normal((40, 4))
        X = np.vstack([X_unique, X_unique[:5]])  # 5 duplicates
        r = DataValidator().validate(X)
        assert r.duplicate_rows == 5

    def test_is_valid_false_when_missing(self):
        from physml.data_validator import DataValidator
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        r = DataValidator().validate(X)
        assert r.is_valid is False

    def test_is_valid_false_when_infinite(self):
        from physml.data_validator import DataValidator
        X = np.array([[1.0, np.inf], [2.0, 3.0]])
        r = DataValidator().validate(X)
        assert r.is_valid is False

    def test_no_issues_valid(self):
        from physml.data_validator import DataValidator
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        r = DataValidator().validate(X)
        assert r.is_valid is True
        assert r.missing_count == 0
        assert r.infinite_count == 0

    def test_check_disabled_skips(self):
        from physml.data_validator import DataValidator
        X = np.array([[np.nan, 1.0], [2.0, 3.0]])
        r = DataValidator(check_missing=False).validate(X)
        assert r.missing_count == 0

    def test_repr(self):
        from physml.data_validator import DataValidator
        dv = DataValidator()
        assert "DataValidator" in repr(dv)


# ===========================================================================
# Stage 89 — PipelineBuilder
# ===========================================================================

class TestPipelineBuilder:

    def test_import(self):
        from physml.pipeline_builder import PipelineBuilder, PipelineStep
        assert PipelineBuilder
        assert PipelineStep

    def test_pipeline_step_fields(self):
        from physml.pipeline_builder import PipelineStep
        scaler = StandardScaler()
        ps = PipelineStep(name="scaler", component=scaler, is_estimator=False)
        assert ps.name == "scaler"
        assert ps.component is scaler
        assert ps.is_estimator is False

    def test_pipeline_step_as_dict(self):
        from physml.pipeline_builder import PipelineStep
        lr = LogisticRegression()
        ps = PipelineStep(name="clf", component=lr, is_estimator=True)
        d = ps.as_dict()
        assert d["name"] == "clf"
        assert d["is_estimator"] is True

    def test_pipeline_step_repr(self):
        from physml.pipeline_builder import PipelineStep
        ps = PipelineStep(name="scaler", component=StandardScaler(), is_estimator=False)
        assert "PipelineStep" in repr(ps)
        assert "scaler" in repr(ps)

    def test_add_step_returns_self(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = PipelineBuilder()
        result = pb.add_step("scaler", StandardScaler())
        assert result is pb

    def test_add_estimator_returns_self(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = PipelineBuilder()
        result = pb.add_estimator("clf", LogisticRegression())
        assert result is pb

    def test_step_names_order(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = (
            PipelineBuilder()
            .add_step("scaler", StandardScaler())
            .add_estimator("clf", LogisticRegression())
        )
        assert pb.step_names == ["scaler", "clf"]

    def test_build_returns_pipeline(self):
        from physml.pipeline_builder import PipelineBuilder
        from sklearn.pipeline import Pipeline
        pb = (
            PipelineBuilder()
            .add_step("scaler", StandardScaler())
            .add_estimator("clf", LogisticRegression(max_iter=300))
        )
        pipe = pb.build()
        assert isinstance(pipe, Pipeline)

    def test_build_pipeline_can_fit_predict(self):
        from physml.pipeline_builder import PipelineBuilder
        X, y = _clf_data()
        pipe = (
            PipelineBuilder()
            .add_step("scaler", StandardScaler())
            .add_estimator("clf", LogisticRegression(max_iter=300, random_state=0))
            .build()
        )
        pipe.fit(X[:150], y[:150])
        preds = pipe.predict(X[150:])
        assert preds.shape == (50,)

    def test_empty_builder_raises_on_build(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = PipelineBuilder()
        with pytest.raises(ValueError):
            pb.build()

    def test_duplicate_transformer_name_raises(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = PipelineBuilder().add_step("scaler", StandardScaler())
        with pytest.raises(ValueError):
            pb.add_step("scaler", StandardScaler())

    def test_add_estimator_replaces_previous(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = (
            PipelineBuilder()
            .add_step("scaler", StandardScaler())
            .add_estimator("clf1", LogisticRegression())
            .add_estimator("clf2", Ridge())
        )
        estimator_steps = [s for s in pb.steps if s.is_estimator]
        assert len(estimator_steps) == 1
        assert estimator_steps[0].name == "clf2"

    def test_get_step(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = PipelineBuilder().add_step("scaler", StandardScaler())
        step = pb.get_step("scaler")
        assert step.name == "scaler"

    def test_get_step_missing_raises(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = PipelineBuilder()
        with pytest.raises(KeyError):
            pb.get_step("nonexistent")

    def test_clear_removes_all_steps(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = (
            PipelineBuilder()
            .add_step("scaler", StandardScaler())
            .add_estimator("clf", LogisticRegression())
        )
        pb.clear()
        assert pb.step_names == []

    def test_repr(self):
        from physml.pipeline_builder import PipelineBuilder
        pb = PipelineBuilder().add_step("scaler", StandardScaler())
        assert "PipelineBuilder" in repr(pb)
        assert "scaler" in repr(pb)
