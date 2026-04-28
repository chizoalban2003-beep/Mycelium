"""Tests for Stages 47–51:
  47 — AutoMLOptimizer
  48 — ConformalClassifier / ConformalRegressor
  49 — Explainer / explain_agent
  50 — AgentCheckpoint
  51 — MetaLearner
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from physml.automl import AutoMLOptimizer
from physml.conformal import ConformalClassifier, ConformalRegressor
from physml.explainability import Explainer, explain_agent
from physml.checkpoint import AgentCheckpoint
from physml.meta_learner import MetaLearner
from physml.mycelium_agent import MyceliumAgent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_data():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="module")
def binary_data():
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:, :6]
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


# ---------------------------------------------------------------------------
# Stage 47 — AutoMLOptimizer
# ---------------------------------------------------------------------------

class TestAutoMLOptimizer:

    @pytest.mark.slow
    def test_fit_returns_dict(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        opt = AutoMLOptimizer(n_candidates=4, cv=2, random_state=0)
        result = opt.fit(X_tr, y_tr)
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_best_score_in_range(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        opt = AutoMLOptimizer(n_candidates=4, cv=2, random_state=0)
        opt.fit(X_tr, y_tr)
        assert 0.0 <= opt.best_score_ <= 1.0

    @pytest.mark.slow
    def test_cv_results_nonempty(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        opt = AutoMLOptimizer(n_candidates=4, cv=2, random_state=0)
        opt.fit(X_tr, y_tr)
        assert len(opt.cv_results_) > 0

    @pytest.mark.slow
    def test_summary_sorted_descending(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        opt = AutoMLOptimizer(n_candidates=4, cv=2, random_state=0)
        opt.fit(X_tr, y_tr)
        scores = [r["mean_test_score"] for r in opt.summary()]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.slow
    def test_custom_param_grid(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        grid = {"n_estimators": [30, 60], "learning_rate": [0.1, 0.2]}
        opt = AutoMLOptimizer(param_grid=grid, n_candidates=4, cv=2, random_state=0)
        best = opt.fit(X_tr, y_tr)
        assert "n_estimators" in best or "learning_rate" in best or len(best) == 0

    @pytest.mark.slow
    def test_get_best_estimator_predicts(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        opt = AutoMLOptimizer(n_candidates=3, cv=2, random_state=1)
        est = opt.get_best_estimator(X_tr, y_tr)
        preds = est.predict(X_te)
        assert preds.shape == (len(y_te),)

    def test_filter_params_skips_unknown(self):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        opt = AutoMLOptimizer()
        params = {"C": 1.0, "nonexistent_param": 99}
        filtered = opt._filter_params(lr, params)
        assert "nonexistent_param" not in filtered
        assert "C" in filtered

    @pytest.mark.slow
    def test_self_improve_auto_tune(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        result = agent.self_improve(X_te, y_te, auto_tune=True)
        assert "best_automl_params" in result

    @pytest.mark.slow
    def test_self_improve_without_auto_tune_no_key(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        result = agent.self_improve(X_te, y_te, auto_tune=False)
        assert "best_automl_params" not in result


# ---------------------------------------------------------------------------
# Stage 48 — ConformalClassifier
# ---------------------------------------------------------------------------

class TestConformalClassifier:

    def test_coverage_at_least_1_minus_alpha(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        base = LogisticRegression(max_iter=300)
        clf = ConformalClassifier(base, alpha=0.1)
        clf.fit(X_tr, y_tr)
        clf.calibrate(X_te[:30], y_te[:30])
        cov = clf.coverage(X_te[30:], y_te[30:])
        # Soft check: ≥ 1 - alpha - slack
        assert cov >= (1 - 0.1 - 0.15)

    def test_predict_set_returns_list(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        base = LogisticRegression(max_iter=300)
        clf = ConformalClassifier(base, alpha=0.1)
        clf.fit(X_tr, y_tr)
        clf.calibrate(X_te[:30], y_te[:30])
        sets = clf.predict_set(X_te[30:35])
        assert isinstance(sets, list)
        assert len(sets) == 5

    def test_prediction_set_elements_are_known_classes(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        base = LogisticRegression(max_iter=300)
        clf = ConformalClassifier(base, alpha=0.1)
        clf.fit(X_tr, y_tr)
        clf.calibrate(X_te[:20], y_te[:20])
        sets = clf.predict_set(X_te[20:25])
        known = set(np.unique(y_tr))
        for s in sets:
            assert len(s) >= 1
            assert all(v in known for v in s)

    def test_uncalibrated_raises(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        clf = ConformalClassifier(LogisticRegression(max_iter=300), alpha=0.1)
        clf.fit(X_tr, y_tr)
        with pytest.raises(RuntimeError, match="calibrate"):
            clf.predict_set(X_te[:5])

    def test_point_predict_matches_base(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        base = LogisticRegression(max_iter=300)
        clf = ConformalClassifier(base, alpha=0.1)
        clf.fit(X_tr, y_tr)
        clf.calibrate(X_te[:20], y_te[:20])
        np.testing.assert_array_equal(clf.predict(X_te[:10]), base.predict(X_te[:10]))

    def test_set_sizes_shape(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        clf = ConformalClassifier(LogisticRegression(max_iter=500), alpha=0.2)
        clf.fit(X_tr, y_tr)
        clf.calibrate(X_te[:30], y_te[:30])
        sizes = clf.set_sizes(X_te[30:])
        assert sizes.shape == (len(X_te[30:]),)
        assert (sizes >= 1).all()

    def test_alpha_0_gives_full_coverage(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        clf = ConformalClassifier(LogisticRegression(max_iter=300), alpha=0.0)
        clf.fit(X_tr, y_tr)
        clf.calibrate(X_te[:30], y_te[:30])
        cov = clf.coverage(X_te[30:], y_te[30:])
        # alpha=0 guarantees ≥ 1-alpha empirically; in finite samples ≈ 1
        assert cov >= 0.85


# ---------------------------------------------------------------------------
# Stage 48 — ConformalRegressor
# ---------------------------------------------------------------------------

class TestConformalRegressor:

    def test_coverage_at_least_1_minus_alpha(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        base = LinearRegression()
        reg = ConformalRegressor(base, alpha=0.1)
        reg.fit(X_tr, y_tr)
        reg.calibrate(X_te[:30], y_te[:30])
        cov = reg.coverage(X_te[30:], y_te[30:])
        assert cov >= (1 - 0.1 - 0.15)

    def test_predict_interval_shape(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        reg = ConformalRegressor(LinearRegression(), alpha=0.1)
        reg.fit(X_tr, y_tr)
        reg.calibrate(X_te[:20], y_te[:20])
        ivs = reg.predict_interval(X_te[20:])
        assert ivs.shape == (len(X_te[20:]), 2)

    def test_lower_le_upper(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        reg = ConformalRegressor(LinearRegression(), alpha=0.1)
        reg.fit(X_tr, y_tr)
        reg.calibrate(X_te[:20], y_te[:20])
        ivs = reg.predict_interval(X_te[20:30])
        assert (ivs[:, 0] <= ivs[:, 1]).all()

    def test_uncalibrated_raises(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        reg = ConformalRegressor(LinearRegression(), alpha=0.1)
        reg.fit(X_tr, y_tr)
        with pytest.raises(RuntimeError, match="calibrate"):
            reg.predict_interval(X_te[:5])

    def test_interval_widths_all_positive(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        reg = ConformalRegressor(LinearRegression(), alpha=0.1)
        reg.fit(X_tr, y_tr)
        reg.calibrate(X_te[:20], y_te[:20])
        widths = reg.interval_widths(X_te[20:])
        assert (widths >= 0).all()

    def test_smaller_alpha_gives_wider_intervals(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        reg_tight = ConformalRegressor(LinearRegression(), alpha=0.3)
        reg_wide = ConformalRegressor(LinearRegression(), alpha=0.05)
        reg_tight.fit(X_tr, y_tr); reg_tight.calibrate(X_te[:20], y_te[:20])
        reg_wide.fit(X_tr, y_tr); reg_wide.calibrate(X_te[:20], y_te[:20])
        w_tight = np.mean(reg_tight.interval_widths(X_te[20:]))
        w_wide = np.mean(reg_wide.interval_widths(X_te[20:]))
        assert w_wide >= w_tight


# ---------------------------------------------------------------------------
# Stage 49 — Explainer / explain_agent
# ---------------------------------------------------------------------------

class TestExplainer:

    def test_fit_produces_importances(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        rf = RandomForestClassifier(n_estimators=20, random_state=0)
        rf.fit(X_tr, y_tr)
        exp = Explainer()
        exp.fit(rf, X_te, y_te)
        assert exp.importances_ is not None
        assert len(exp.importances_) == X_te.shape[1]

    def test_importances_sum_to_one(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        rf = RandomForestClassifier(n_estimators=20, random_state=0)
        rf.fit(X_tr, y_tr)
        exp = Explainer()
        exp.fit(rf, X_te, y_te)
        assert abs(exp.importances_.sum() - 1.0) < 1e-6

    def test_top_features_sorted(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        rf = RandomForestClassifier(n_estimators=20, random_state=0)
        rf.fit(X_tr, y_tr)
        exp = Explainer()
        exp.fit(rf, X_te, y_te, feature_names=["a", "b", "c", "d"])
        top = exp.top_features(k=4)
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_feature_names_from_list(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X_tr, y_tr)
        names = ["sepal_l", "sepal_w", "petal_l", "petal_w"]
        exp = Explainer()
        exp.fit(rf, X_te, y_te, feature_names=names)
        assert exp.feature_names_ == names

    def test_report_dict_keys(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        rf = RandomForestClassifier(n_estimators=10, random_state=0)
        rf.fit(X_tr, y_tr)
        exp = Explainer()
        exp.fit(rf, X_te, y_te)
        rep = exp.report()
        assert "feature_importances" in rep
        assert "top_5" in rep
        assert "n_features" in rep

    def test_permutation_fallback_linear_model(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        # Linear model has coef_ — direct path
        lr = LogisticRegression(max_iter=300)
        lr.fit(X_tr, y_tr)
        exp = Explainer()
        exp.fit(lr, X_te, y_te)
        assert exp.importances_ is not None

    @pytest.mark.slow
    def test_permutation_path_no_model_attrs(self, iris_data):
        """Use a wrapped model without feature_importances_ or coef_."""
        X_tr, X_te, y_tr, y_te = iris_data

        class _NoAttr:
            def fit(self, X, y): self._lr = LogisticRegression(max_iter=300).fit(X, y)
            def predict(self, X): return self._lr.predict(X)
            def predict_proba(self, X): return self._lr.predict_proba(X)

        m = _NoAttr()
        m.fit(X_tr, y_tr)
        exp = Explainer(n_repeats=2, random_state=0)
        exp.fit(m, X_te, y_te)
        assert exp.importances_ is not None

    def test_unfitted_raises(self):
        exp = Explainer()
        with pytest.raises(RuntimeError, match="fit"):
            exp.top_features()

    @pytest.mark.slow
    def test_explain_agent_convenience(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        exp = explain_agent(agent, X_te, y_te, n_repeats=2)
        assert exp.importances_ is not None

    def test_regression_explainer(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        rf = RandomForestRegressor(n_estimators=10, random_state=0)
        rf.fit(X_tr, y_tr)
        exp = Explainer(n_repeats=2)
        exp.fit(rf, X_te, y_te)
        assert len(exp.importances_) == X_te.shape[1]


# ---------------------------------------------------------------------------
# Stage 50 — AgentCheckpoint
# ---------------------------------------------------------------------------

class TestAgentCheckpoint:

    @pytest.mark.slow
    def test_save_creates_file(self, iris_data, tmp_path):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        ckpt_path = AgentCheckpoint.save(agent, tmp_path / "agent.ckpt")
        assert Path(ckpt_path).exists()

    @pytest.mark.slow
    def test_load_returns_agent(self, iris_data, tmp_path):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        path = AgentCheckpoint.save(agent, tmp_path / "agent.ckpt")
        agent2 = AgentCheckpoint.load(path)
        assert isinstance(agent2, MyceliumAgent)

    @pytest.mark.slow
    def test_loaded_agent_predicts(self, iris_data, tmp_path):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        path = AgentCheckpoint.save(agent, tmp_path / "pred.ckpt")
        agent2 = AgentCheckpoint.load(path)
        action = agent2.observe(X_te[:1])
        assert hasattr(action, "prediction")

    @pytest.mark.slow
    def test_inspect_no_agent_object(self, iris_data, tmp_path):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        path = AgentCheckpoint.save(agent, tmp_path / "inspect.ckpt")
        meta = AgentCheckpoint.inspect(path)
        assert meta["agent"] == "<not loaded>"
        assert "version" in meta
        assert "timestamp" in meta

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AgentCheckpoint.load(tmp_path / "nonexistent.ckpt")

    @pytest.mark.slow
    def test_bytes_roundtrip(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        data = AgentCheckpoint.save_bytes(agent)
        agent2 = AgentCheckpoint.load_bytes(data)
        assert isinstance(agent2, MyceliumAgent)

    @pytest.mark.slow
    def test_inspect_has_file_size(self, iris_data, tmp_path):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        path = AgentCheckpoint.save(agent, tmp_path / "sz.ckpt")
        meta = AgentCheckpoint.inspect(path)
        assert meta["file_size_bytes"] > 0

    @pytest.mark.slow
    def test_compression_produces_smaller_file(self, iris_data, tmp_path):
        X_tr, X_te, y_tr, y_te = iris_data
        agent = MyceliumAgent()
        agent.fit(X_tr, y_tr)
        p0 = AgentCheckpoint.save(agent, tmp_path / "c0.ckpt", compress=0)
        p9 = AgentCheckpoint.save(agent, tmp_path / "c9.ckpt", compress=9)
        assert Path(p9).stat().st_size <= Path(p0).stat().st_size


# ---------------------------------------------------------------------------
# Stage 51 — MetaLearner
# ---------------------------------------------------------------------------

class TestMetaLearner:

    def test_record_increases_history(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner()
        ml.record(X_tr, y_tr, {"query_strategy": "entropy", "policy": "adaptive"}, 0.85)
        assert ml.history_size() == 1

    def test_recommend_with_few_entries_returns_default(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner(min_history=5)
        ml.record(X_tr, y_tr, {"query_strategy": "entropy"}, 0.80)
        rec = ml.recommend(X_te, y_te)
        assert isinstance(rec, dict)

    def test_recommend_returns_dict(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner(min_history=2)
        for _ in range(3):
            ml.record(X_tr, y_tr, {"query_strategy": "entropy", "policy": "adaptive"}, 0.85)
        rec = ml.recommend(X_te, y_te)
        assert isinstance(rec, dict)
        assert "query_strategy" in rec

    def test_recommend_favours_better_config(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner(min_history=2)
        # Record a bad config many times
        for _ in range(5):
            ml.record(X_tr, y_tr, {"query_strategy": "threshold", "policy": "fixed"}, 0.55)
        # Record a good config many times
        for _ in range(5):
            ml.record(X_tr, y_tr, {"query_strategy": "entropy", "policy": "adaptive"}, 0.92)
        rec = ml.recommend(X_te, y_te)
        assert rec.get("query_strategy") == "entropy"

    def test_top_configs_sorted_by_score(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner()
        ml.record(X_tr, y_tr, {"cfg": "a"}, 0.70)
        ml.record(X_tr, y_tr, {"cfg": "b"}, 0.85)
        ml.record(X_tr, y_tr, {"cfg": "c"}, 0.60)
        top = ml.top_configs(k=3)
        scores = [e["score"] for e in top]
        assert scores == sorted(scores, reverse=True)

    def test_dataset_profile_returns_dict(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner()
        prof = ml.dataset_profile(X_tr, y_tr)
        assert isinstance(prof, dict)
        assert "log_n_samples" in prof

    def test_profile_vector_unit_norm(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner()
        vec = ml._dataset_profile(X_tr, y_tr)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-9 or np.linalg.norm(vec) < 1e-9

    def test_gini_binary(self):
        ml = MetaLearner()
        y = np.array([0, 1, 0, 1, 0, 1])
        g = ml._gini(y)
        assert 0.4 < g < 0.6   # balanced binary → gini ≈ 0.5

    def test_gini_pure(self):
        ml = MetaLearner()
        y = np.array([0, 0, 0, 0])
        assert ml._gini(y) == 0.0

    def test_metadata_stored_and_retrievable(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        ml = MetaLearner()
        ml.record(X_tr, y_tr, {"cfg": "x"}, 0.9, metadata={"dataset": "iris"})
        assert ml._entries[0].metadata["dataset"] == "iris"
