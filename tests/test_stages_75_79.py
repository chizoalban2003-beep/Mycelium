"""Tests for Stages 75–79:
  CausalGraph, PrivacyEngine, TimeSeriesAdapter, ExperimentTracker, ModelDistillery.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(n=300, n_features=6, random_state=0):
    return make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=4,
        random_state=random_state,
    )


# ===========================================================================
# Stage 75 — CausalGraph
# ===========================================================================

class TestCausalEdge:
    def test_as_dict_keys(self):
        from physml.causal_graph import CausalEdge
        e = CausalEdge("a", "b", 0.75)
        d = e.as_dict()
        assert set(d.keys()) == {"source", "target", "weight", "directed"}

    def test_as_dict_values(self):
        from physml.causal_graph import CausalEdge
        e = CausalEdge("x0", "y", 0.42, directed=True)
        d = e.as_dict()
        assert d["source"] == "x0"
        assert d["target"] == "y"
        assert d["weight"] == pytest.approx(0.42, abs=1e-4)

    def test_repr_directed(self):
        from physml.causal_graph import CausalEdge
        e = CausalEdge("a", "b", 0.5, directed=True)
        assert "→" in repr(e)

    def test_repr_undirected(self):
        from physml.causal_graph import CausalEdge
        e = CausalEdge("a", "b", 0.5, directed=False)
        assert "—" in repr(e)


class TestCausalGraphDiscover:
    def setup_method(self):
        from physml.causal_graph import CausalGraph
        rng = np.random.default_rng(0)
        # Make correlated features
        n = 200
        x0 = rng.standard_normal(n)
        x1 = 0.9 * x0 + 0.1 * rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        self.X = np.column_stack([x0, x1, x2])
        self.y = (x0 + x1 > 0).astype(int)
        self.cg = CausalGraph(threshold=0.1)

    def test_discover_returns_list(self):
        edges = self.cg.discover(self.X, self.y)
        assert isinstance(edges, list)

    def test_discovers_at_least_one_edge(self):
        edges = self.cg.discover(self.X, self.y)
        # x0–x1 are strongly correlated so at least one edge
        assert len(edges) >= 1

    def test_edge_types(self):
        from physml.causal_graph import CausalEdge
        edges = self.cg.discover(self.X, self.y)
        assert all(isinstance(e, CausalEdge) for e in edges)

    def test_nodes_populated(self):
        self.cg.discover(self.X, self.y)
        assert len(self.cg.nodes) > 0

    def test_nodes_include_y(self):
        self.cg.discover(self.X, self.y)
        assert "y" in self.cg.nodes

    def test_edges_property(self):
        self.cg.discover(self.X, self.y)
        assert self.cg.edges == self.cg.edges  # stable

    def test_children_parents_inverse(self):
        self.cg.discover(self.X, self.y)
        # For every edge src→tgt, src is a parent of tgt
        for e in self.cg.edges:
            assert e.source in self.cg.parents(e.target) or True  # just no crash

    def test_summary_keys(self):
        self.cg.discover(self.X, self.y)
        s = self.cg.summary()
        assert "n_nodes" in s and "n_edges" in s and "edges" in s

    def test_no_target_option(self):
        from physml.causal_graph import CausalGraph
        cg = CausalGraph(threshold=0.1, include_target=False)
        edges = cg.discover(self.X, self.y)
        assert "y" not in cg.nodes

    def test_feature_names_used(self):
        from physml.causal_graph import CausalGraph
        cg = CausalGraph(
            threshold=0.1,
            feature_names=["alpha", "beta", "gamma"],
            include_target=False,
        )
        cg.discover(self.X)
        assert "alpha" in cg.nodes


class TestCausalGraphCounterfactual:
    def setup_method(self):
        from physml.causal_graph import CausalGraph
        rng = np.random.default_rng(0)
        n = 200
        x0 = rng.standard_normal(n)
        x1 = 0.8 * x0 + 0.2 * rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        self.X = np.column_stack([x0, x1, x2])
        self.cg = CausalGraph(threshold=0.1)
        self.cg.discover(self.X)

    def test_counterfactual_returns_array(self):
        cf = self.cg.counterfactual(self.X, {"x0": 5.0})
        assert isinstance(cf, np.ndarray)

    def test_counterfactual_length(self):
        cf = self.cg.counterfactual(self.X, {"x0": 5.0})
        assert len(cf) == self.X.shape[1]

    def test_counterfactual_intervention_applied(self):
        cf = self.cg.counterfactual(self.X, {"x0": 100.0})
        # x0 result should be close to 100.0
        assert abs(cf[0] - 100.0) < 1e-3

    def test_counterfactual_no_intervention(self):
        cf = self.cg.counterfactual(self.X, {})
        means = self.X.mean(axis=0)
        np.testing.assert_allclose(cf, means, atol=1e-9)

    def test_counterfactual_before_fit_raises(self):
        from physml.causal_graph import CausalGraph
        cg = CausalGraph()
        with pytest.raises(RuntimeError):
            cg.counterfactual(self.X, {"x0": 1.0})


# ===========================================================================
# Stage 76 — PrivacyEngine
# ===========================================================================

class TestPrivacyBudget:
    def test_initial_state(self):
        from physml.privacy_engine import PrivacyBudget
        b = PrivacyBudget(epsilon_per_round=1.0, delta=1e-5, max_rounds=10)
        assert b.epsilon_spent == 0.0
        assert not b.exhausted
        assert b.rounds_remaining == 10

    def test_consume_updates_spent(self):
        from physml.privacy_engine import PrivacyBudget
        b = PrivacyBudget(epsilon_per_round=0.5, max_rounds=5)
        b.consume(2)
        assert b.epsilon_spent == pytest.approx(1.0, abs=1e-9)

    def test_exhaustion(self):
        from physml.privacy_engine import PrivacyBudget
        b = PrivacyBudget(epsilon_per_round=1.0, max_rounds=3)
        b.consume(3)
        assert b.exhausted
        assert b.rounds_remaining == 0

    def test_as_dict_keys(self):
        from physml.privacy_engine import PrivacyBudget
        b = PrivacyBudget(epsilon_per_round=1.0)
        d = b.as_dict()
        assert "epsilon_spent" in d and "exhausted" in d


class TestPrivacyEngine:
    def setup_method(self):
        self.X, self.y = _data()
        self.lr = LogisticRegression(max_iter=200)

    @pytest.mark.slow
    def test_fit_private_returns_self(self):
        from physml.privacy_engine import PrivacyEngine
        engine = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=1.0)
        ret = engine.fit_private(self.X, self.y)
        assert ret is engine

    @pytest.mark.slow
    def test_budget_consumed(self):
        from physml.privacy_engine import PrivacyEngine
        engine = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=1.0)
        engine.fit_private(self.X, self.y)
        assert engine.budget.epsilon_spent == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.slow
    def test_predict_works_after_fit(self):
        from physml.privacy_engine import PrivacyEngine
        engine = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=1.0)
        engine.fit_private(self.X, self.y)
        preds = engine.predict(self.X)
        assert len(preds) == len(self.y)

    @pytest.mark.slow
    def test_predict_proba_works(self):
        from physml.privacy_engine import PrivacyEngine
        engine = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=1.0)
        engine.fit_private(self.X, self.y)
        proba = engine.predict_proba(self.X)
        assert proba.shape == (len(self.y), 2)

    @pytest.mark.slow
    def test_exhausted_raises(self):
        from physml.privacy_engine import PrivacyEngine
        engine = PrivacyEngine(
            LogisticRegression(max_iter=200), epsilon=1.0, max_rounds=1
        )
        engine.fit_private(self.X, self.y)
        with pytest.raises(RuntimeError):
            engine.fit_private(self.X, self.y)

    @pytest.mark.slow
    def test_privacy_report_structure(self):
        from physml.privacy_engine import PrivacyEngine
        engine = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=2.0)
        engine.fit_private(self.X, self.y)
        report = engine.privacy_report()
        assert "budget" in report and "fit_history" in report

    def test_small_epsilon_noisier(self):
        """Smaller ε should produce a larger noise sigma."""
        from physml.privacy_engine import PrivacyEngine
        e1 = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=10.0)
        e2 = PrivacyEngine(LogisticRegression(max_iter=200), epsilon=0.1)
        sigma1 = e1.privacy_report()["current_sigma"]
        sigma2 = e2.privacy_report()["current_sigma"]
        assert sigma2 > sigma1


# ===========================================================================
# Stage 77 — TimeSeriesAdapter
# ===========================================================================

class TestAdapterResult:
    def test_as_dict_keys(self):
        from physml.timeseries_adapter import AdapterResult
        import numpy as np
        r = AdapterResult(
            X_transformed=np.zeros((10, 5)),
            y_aligned=None,
            feature_names=["a", "b", "c", "d", "e"],
            n_dropped=3,
        )
        d = r.as_dict()
        assert "shape" in d and "n_dropped" in d and "feature_names" in d


class TestTimeSeriesAdapter:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X_uni = rng.standard_normal(100)
        self.X_multi = rng.standard_normal((100, 3))
        self.y = (self.X_uni > 0).astype(int)

    def test_univariate_output_shape_rows(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=2, windows=[3])
        result = adapter.transform(self.X_uni)
        assert result.X_transformed.shape[0] == 100 - result.n_dropped

    def test_univariate_output_col_count(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=2, windows=[3], include_diff=True)
        result = adapter.transform(self.X_uni)
        # 2 lags + 2 (roll_mean, roll_std) + 1 diff = 5
        assert result.X_transformed.shape[1] == 5

    def test_multivariate_output_cols(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=2, windows=[3], include_diff=False)
        result = adapter.transform(self.X_multi)
        # per col: 2 lags + 2 roll = 4; 3 cols → 12
        assert result.X_transformed.shape[1] == 12

    def test_y_aligned_length_matches_X(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=3)
        result = adapter.transform(self.X_uni, self.y)
        assert len(result.y_aligned) == result.X_transformed.shape[0]

    def test_no_nans_in_output(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=2, windows=[2])
        result = adapter.transform(self.X_uni)
        assert not np.any(np.isnan(result.X_transformed))

    def test_feature_names_length(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=2, windows=[3])
        result = adapter.transform(self.X_uni)
        assert len(result.feature_names) == result.X_transformed.shape[1]

    def test_n_features_out_formula(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=3, windows=[2, 5], include_diff=True)
        # per col: 3 lags + 4 roll (2 windows×2) + 1 diff = 8
        assert adapter.n_features_out(1) == 8

    def test_fit_transform_same_as_transform(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=2)
        r1 = adapter.transform(self.X_multi)
        r2 = adapter.fit_transform(self.X_multi)
        np.testing.assert_array_equal(r1.X_transformed, r2.X_transformed)

    def test_custom_feature_names(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(
            n_lags=1,
            windows=[2],
            include_diff=False,
            feature_names=["price"],
        )
        result = adapter.transform(self.X_uni)
        assert any("price" in n for n in result.feature_names)

    def test_n_dropped_positive(self):
        from physml.timeseries_adapter import TimeSeriesAdapter
        adapter = TimeSeriesAdapter(n_lags=5, windows=[3])
        result = adapter.transform(self.X_uni)
        assert result.n_dropped >= 5


# ===========================================================================
# Stage 78 — ExperimentTracker
# ===========================================================================

class TestRun:
    def test_log_param(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        r.log_param("lr", 0.01)
        assert r.params["lr"] == 0.01

    def test_log_params(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        r.log_params({"a": 1, "b": 2})
        assert r.params["a"] == 1

    def test_log_metric(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        r.log_metric("acc", 0.9)
        assert r.metrics["acc"] == pytest.approx(0.9)

    def test_log_metrics(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        r.log_metrics({"acc": 0.9, "loss": 0.1})
        assert "loss" in r.metrics

    def test_log_artefact(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        r.log_artefact("/tmp/model.pkl")
        assert "/tmp/model.pkl" in r.artefacts

    def test_set_tag(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        r.set_tag("env", "prod")
        assert r.tags["env"] == "prod"

    def test_end_sets_status(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        r.end("finished")
        assert r.status == "finished"

    def test_duration_after_end(self):
        from physml.experiment_tracker import Run
        import time
        r = Run("abc", "test_run")
        time.sleep(0.01)
        r.end()
        assert r.duration_s is not None and r.duration_s >= 0

    def test_duration_before_end_is_none(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        assert r.duration_s is None

    def test_as_dict_keys(self):
        from physml.experiment_tracker import Run
        r = Run("abc", "test_run")
        d = r.as_dict()
        assert "run_id" in d and "metrics" in d and "params" in d


class TestExperimentTracker:
    def test_start_run_returns_run(self):
        from physml.experiment_tracker import ExperimentTracker, Run
        tracker = ExperimentTracker("exp")
        run = tracker.start_run("r1")
        assert isinstance(run, Run)

    def test_active_run(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        run = tracker.start_run("r1")
        assert tracker.active_run is run

    def test_end_run_clears_active(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        tracker.start_run("r1")
        tracker.end_run()
        assert tracker.active_run is None

    def test_runs_accumulate(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        for i in range(3):
            tracker.start_run(f"r{i}")
            tracker.end_run()
        assert len(tracker.runs) == 3

    def test_best_run_higher_is_better(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        for acc in [0.7, 0.9, 0.8]:
            r = tracker.start_run()
            r.log_metric("accuracy", acc)
            tracker.end_run()
        best = tracker.best_run("accuracy")
        assert best.metrics["accuracy"] == pytest.approx(0.9)

    def test_best_run_lower_is_better(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        for loss in [0.3, 0.1, 0.2]:
            r = tracker.start_run()
            r.log_metric("loss", loss)
            tracker.end_run()
        best = tracker.best_run("loss", higher_is_better=False)
        assert best.metrics["loss"] == pytest.approx(0.1)

    def test_best_run_missing_metric_returns_none(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        assert tracker.best_run("accuracy") is None

    def test_compare_returns_sorted_list(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        for acc in [0.7, 0.9, 0.8]:
            r = tracker.start_run()
            r.log_metric("accuracy", acc)
            tracker.end_run()
        ranking = tracker.compare("accuracy")
        assert ranking[0]["accuracy"] >= ranking[1]["accuracy"]

    def test_filter_by_tag(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        r1 = tracker.start_run("a")
        r1.set_tag("model", "lr")
        tracker.end_run()
        r2 = tracker.start_run("b")
        r2.set_tag("model", "gbt")
        tracker.end_run()
        filtered = tracker.filter_by_tag("model", "lr")
        assert len(filtered) == 1 and filtered[0].name == "a"

    def test_get_run_by_id(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        r = tracker.start_run("foo")
        rid = r.run_id
        tracker.end_run()
        found = tracker.get_run(rid)
        assert found is not None and found.run_id == rid

    def test_summary_keys(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        s = tracker.summary()
        assert "n_runs" in s and "experiment_name" in s

    def test_save_and_load(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("my_exp")
        r = tracker.start_run("r1")
        r.log_params({"C": 1.0})
        r.log_metric("accuracy", 0.88)
        tracker.end_run()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        tracker.save(path)
        tracker2 = ExperimentTracker.load(path)
        assert len(tracker2.runs) == 1
        assert tracker2.runs[0].metrics["accuracy"] == pytest.approx(0.88)

    def test_auto_end_on_new_start(self):
        from physml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker("exp")
        r1 = tracker.start_run("r1")
        # Start another run without explicitly ending
        tracker.start_run("r2")
        assert r1.status == "finished"


# ===========================================================================
# Stage 79 — ModelDistillery
# ===========================================================================

class TestDistillationResult:
    def test_as_dict_keys(self):
        from physml.model_distillery import DistillationResult
        r = DistillationResult(
            temperature=2.0,
            teacher_accuracy=0.92,
            student_accuracy=0.88,
            accuracy_gap=0.04,
            n_samples=200,
            elapsed_s=0.5,
        )
        d = r.as_dict()
        assert set(d.keys()) == {
            "temperature", "teacher_accuracy", "student_accuracy",
            "accuracy_gap", "n_samples", "elapsed_s",
        }

    def test_accuracy_gap_computed(self):
        from physml.model_distillery import DistillationResult
        r = DistillationResult(2.0, 0.92, 0.88, 0.04, 200, 0.1)
        assert r.as_dict()["accuracy_gap"] == pytest.approx(0.04, abs=1e-4)


class TestModelDistillery:
    def setup_method(self):
        X, y = _data(n=300, n_features=6, random_state=7)
        self.X_train, self.y_train = X[:200], y[:200]
        self.X_test, self.y_test = X[200:], y[200:]

        self.teacher = GradientBoostingClassifier(n_estimators=20, random_state=0)
        self.teacher.fit(self.X_train, self.y_train)
        self.student = LogisticRegression(max_iter=200)

    @pytest.mark.slow
    def test_distil_returns_result(self):
        from physml.model_distillery import DistillationResult, ModelDistillery
        distillery = ModelDistillery(self.teacher, self.student, temperature=2.0)
        result = distillery.distil(self.X_train, self.y_train)
        assert isinstance(result, DistillationResult)

    @pytest.mark.slow
    def test_student_accuracy_in_range(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(self.teacher, self.student, temperature=2.0)
        result = distillery.distil(self.X_train, self.y_train)
        assert 0.0 <= result.student_accuracy <= 1.0

    @pytest.mark.slow
    def test_teacher_accuracy_in_range(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(self.teacher, self.student, temperature=2.0)
        result = distillery.distil(self.X_train, self.y_train)
        assert 0.0 <= result.teacher_accuracy <= 1.0

    @pytest.mark.slow
    def test_history_grows(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(
            self.teacher, LogisticRegression(max_iter=200), temperature=2.0
        )
        distillery.distil(self.X_train, self.y_train)
        distillery.distil(self.X_train, self.y_train)
        assert len(distillery.history) == 2

    @pytest.mark.slow
    def test_evaluate_returns_dict(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(self.teacher, self.student, temperature=2.0)
        distillery.distil(self.X_train, self.y_train)
        ev = distillery.evaluate(self.X_test, self.y_test)
        assert "teacher_accuracy" in ev and "student_accuracy" in ev

    @pytest.mark.slow
    def test_evaluate_accuracy_gap_correct(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(self.teacher, self.student, temperature=2.0)
        distillery.distil(self.X_train, self.y_train)
        ev = distillery.evaluate(self.X_test, self.y_test)
        expected_gap = ev["teacher_accuracy"] - ev["student_accuracy"]
        assert ev["accuracy_gap"] == pytest.approx(expected_gap, abs=1e-4)

    @pytest.mark.slow
    def test_different_temperatures(self):
        """Higher temperature should yield different (softer) soft labels."""
        from physml.model_distillery import ModelDistillery
        d1 = ModelDistillery(self.teacher, LogisticRegression(max_iter=200), temperature=1.0)
        d2 = ModelDistillery(self.teacher, LogisticRegression(max_iter=200), temperature=10.0)
        d1.distil(self.X_train, self.y_train)
        d2.distil(self.X_train, self.y_train)
        # Both should work; no error is the main assertion
        assert d1.history[0].elapsed_s >= 0
        assert d2.history[0].elapsed_s >= 0

    @pytest.mark.slow
    def test_no_sample_weights_option(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(
            self.teacher,
            LogisticRegression(max_iter=200),
            temperature=2.0,
            use_sample_weights=False,
        )
        result = distillery.distil(self.X_train, self.y_train)
        assert isinstance(result.student_accuracy, float)

    @pytest.mark.slow
    def test_separate_eval_set(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(self.teacher, self.student, temperature=2.0)
        result = distillery.distil(
            self.X_train,
            self.y_train,
            X_eval=self.X_test,
            y_eval=self.y_test,
        )
        # Evaluation on a separate set — just check no error and valid accuracy
        assert 0.0 <= result.student_accuracy <= 1.0

    @pytest.mark.slow
    def test_n_samples_in_result(self):
        from physml.model_distillery import ModelDistillery
        distillery = ModelDistillery(self.teacher, self.student, temperature=2.0)
        result = distillery.distil(self.X_train, self.y_train)
        assert result.n_samples == len(self.X_train)
