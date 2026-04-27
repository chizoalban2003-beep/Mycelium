"""Tests for Stages 80–84:
  ActiveLearner, FeatureEngineer, ImbalancedHandler, OnlineEvaluator, ModelZoo.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clf_data(n=300, n_features=8, random_state=0):
    n_informative = min(4, n_features - 1)
    n_redundant = min(2, n_features - n_informative - 1)
    return make_classification(
        n_samples=n, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, random_state=random_state
    )


def _reg_data(n=200, n_features=6, random_state=0):
    return make_regression(
        n_samples=n, n_features=n_features, noise=10.0, random_state=random_state
    )


def _imbalanced_clf(n_majority=150, n_minority=30, n_features=4, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_majority + n_minority, n_features))
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


# ===========================================================================
# Stage 80 — ActiveLearner
# ===========================================================================

class TestActiveLearner:

    def _setup(self, strategy="entropy", n_query=10):
        from physml.active_learner import ActiveLearner
        X, y = _clf_data(n=300)
        estimator = LogisticRegression(max_iter=300, random_state=0)
        learner = ActiveLearner(estimator, strategy=strategy, n_query=n_query)
        learner.initialize(X[:60], y[:60], X[60:])
        return learner, X, y

    def test_import(self):
        from physml.active_learner import ActiveLearner, QueryResult
        assert ActiveLearner
        assert QueryResult

    def test_init_basic(self):
        from physml.active_learner import ActiveLearner
        learner = ActiveLearner(LogisticRegression(), strategy="entropy", n_query=5)
        assert learner.strategy == "entropy"
        assert learner.n_query == 5

    def test_invalid_strategy(self):
        from physml.active_learner import ActiveLearner
        with pytest.raises(ValueError):
            ActiveLearner(LogisticRegression(), strategy="bogus")

    @pytest.mark.slow
    def test_initialize(self):
        learner, X, y = self._setup()
        assert learner.n_labelled == 60
        assert learner.n_pool == 240

    @pytest.mark.slow
    def test_query_returns_correct_type(self):
        from physml.active_learner import QueryResult
        learner, _, _ = self._setup()
        result = learner.query()
        assert isinstance(result, QueryResult)

    @pytest.mark.slow
    def test_query_n_indices(self):
        learner, _, _ = self._setup(n_query=15)
        result = learner.query()
        assert len(result.query_indices) == 15

    @pytest.mark.slow
    def test_query_indices_in_range(self):
        learner, _, _ = self._setup()
        result = learner.query()
        pool_size = learner.n_pool  # before update
        # indices should be within the pool size at time of query (240)
        assert all(0 <= i < 240 for i in result.query_indices)

    @pytest.mark.slow
    def test_query_result_fields(self):
        learner, _, _ = self._setup()
        result = learner.query()
        assert result.strategy == "entropy"
        assert len(result.scores) == len(result.query_indices)
        assert result.elapsed_s >= 0

    @pytest.mark.slow
    def test_query_as_dict(self):
        learner, _, _ = self._setup()
        d = learner.query().as_dict()
        assert "strategy" in d
        assert "n_selected" in d
        assert "n_labelled" in d
        assert "n_unlabelled" in d

    @pytest.mark.slow
    def test_update_grows_labelled(self):
        learner, X, y = self._setup(n_query=10)
        result = learner.query()
        labelled_before = learner.n_labelled
        pool_before = learner.n_pool
        learner.update(result.query_indices, y[60 + np.array(result.query_indices)])
        assert learner.n_labelled == labelled_before + 10
        assert learner.n_pool == pool_before - 10

    @pytest.mark.slow
    def test_score_method(self):
        learner, X, y = self._setup()
        score = learner.score(X[:30], y[:30])
        assert 0.0 <= score <= 1.0

    @pytest.mark.slow
    def test_history_accumulates(self):
        learner, X, y = self._setup()
        for _ in range(3):
            result = learner.query()
            learner.update(result.query_indices, y[60:][result.query_indices[:len(result.query_indices)]])
        assert len(learner.history) == 3

    @pytest.mark.slow
    def test_strategy_least_confident(self):
        learner, _, _ = self._setup(strategy="least_confident")
        result = learner.query()
        assert result.strategy == "least_confident"
        assert len(result.query_indices) == 10

    @pytest.mark.slow
    def test_strategy_margin(self):
        learner, _, _ = self._setup(strategy="margin")
        result = learner.query()
        assert result.strategy == "margin"
        assert len(result.query_indices) == 10

    @pytest.mark.slow
    def test_strategy_qbc(self):
        from physml.active_learner import ActiveLearner
        X, y = _clf_data(n=200)
        learner = ActiveLearner(
            LogisticRegression(max_iter=200), strategy="qbc", n_query=5,
            committee_size=3
        )
        learner.initialize(X[:50], y[:50], X[50:])
        result = learner.query()
        assert len(result.query_indices) == 5

    def test_no_init_raises(self):
        from physml.active_learner import ActiveLearner
        learner = ActiveLearner(LogisticRegression())
        with pytest.raises(RuntimeError):
            learner.query()

    def test_public_api_via_init(self):
        from physml import ActiveLearner, QueryResult
        assert ActiveLearner
        assert QueryResult


# ===========================================================================
# Stage 81 — FeatureEngineer
# ===========================================================================

class TestFeatureEngineer:

    def test_import(self):
        from physml.feature_engineer import FeatureEngineer, EngineeredFeatures
        assert FeatureEngineer
        assert EngineeredFeatures

    def test_fit_transform_shape(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=200, n_features=5)
        fe = FeatureEngineer(top_k=10)
        X_new, result = fe.fit_transform(X, y)
        assert X_new.shape[0] == 200
        assert X_new.shape[1] == result.n_selected == 10

    def test_result_fields(self):
        from physml.feature_engineer import FeatureEngineer, EngineeredFeatures
        X, y = _clf_data(n=100)
        fe = FeatureEngineer(top_k=8)
        _, result = fe.fit_transform(X, y)
        assert isinstance(result, EngineeredFeatures)
        assert result.n_original == 8
        assert result.n_selected == 8
        assert len(result.feature_names) == 8
        assert len(result.mi_scores) == 8
        assert result.elapsed_s >= 0

    def test_result_as_dict(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=100)
        _, result = FeatureEngineer(top_k=5).fit_transform(X, y)
        d = result.as_dict()
        assert "n_original" in d
        assert "n_selected" in d
        assert "feature_names" in d

    def test_polynomial_features(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=100, n_features=4)
        fe = FeatureEngineer(polynomial=True, interactions=False, log_transform=False, top_k=None)
        _, result = fe.fit_transform(X, y)
        # should have original + squared = 8
        assert result.n_generated >= 8

    def test_interactions_features(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=100, n_features=4)
        fe = FeatureEngineer(polynomial=False, interactions=True, log_transform=False, top_k=None)
        _, result = fe.fit_transform(X, y)
        # 4 originals + 6 interaction pairs = 10
        assert result.n_generated >= 10

    def test_log_features(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=100, n_features=4)
        fe = FeatureEngineer(polynomial=False, interactions=False, log_transform=True, top_k=None)
        _, result = fe.fit_transform(X, y)
        assert result.n_generated >= 8

    def test_ratios_features(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=100, n_features=3)
        fe = FeatureEngineer(polynomial=False, interactions=False, log_transform=False,
                             ratios=True, top_k=None)
        _, result = fe.fit_transform(X, y)
        assert result.n_generated > 3

    def test_transform_consistency(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=200, n_features=5)
        fe = FeatureEngineer(top_k=8)
        X_new, _ = fe.fit_transform(X[:100], y[:100])
        X_test = fe.transform(X[100:])
        assert X_test.shape[1] == X_new.shape[1]

    def test_transform_before_fit_raises(self):
        from physml.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        X, _ = _clf_data(n=50)
        with pytest.raises(RuntimeError):
            fe.transform(X)

    def test_feature_names(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=100, n_features=4)
        fe = FeatureEngineer(top_k=6, polynomial=True, interactions=False, log_transform=False)
        _, result = fe.fit_transform(X, y, feature_names=["a", "b", "c", "d"])
        # Feature names should use the provided base names
        assert any("a" in name or "b" in name for name in result.feature_names)

    def test_top_k_none_keeps_all(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _clf_data(n=100, n_features=3)
        fe = FeatureEngineer(polynomial=True, interactions=True, log_transform=True, top_k=None)
        X_new, result = fe.fit_transform(X, y)
        assert X_new.shape[1] == result.n_generated

    def test_regression_task(self):
        from physml.feature_engineer import FeatureEngineer
        X, y = _reg_data(n=150)
        fe = FeatureEngineer(task="regression", top_k=10)
        X_new, result = fe.fit_transform(X, y)
        assert result.n_selected == 10

    def test_public_api_via_init(self):
        from physml import FeatureEngineer, EngineeredFeatures
        assert FeatureEngineer
        assert EngineeredFeatures


# ===========================================================================
# Stage 82 — ImbalancedHandler
# ===========================================================================

class TestImbalancedHandler:

    def test_import(self):
        from physml.imbalanced import ImbalancedHandler, ImbalanceReport
        assert ImbalancedHandler
        assert ImbalanceReport

    def test_invalid_strategy(self):
        from physml.imbalanced import ImbalancedHandler
        with pytest.raises(ValueError):
            ImbalancedHandler(strategy="magic")

    def test_oversample_balances(self):
        from physml.imbalanced import ImbalancedHandler
        X, y = _imbalanced_clf()
        handler = ImbalancedHandler(strategy="oversample")
        X_res, y_res, report = handler.resample(X, y)
        counts = {c: (y_res == c).sum() for c in np.unique(y_res)}
        # Majority and minority should be equal (or close)
        assert report.imbalance_ratio_after <= 1.05
        assert report.n_added > 0

    def test_undersample_balances(self):
        from physml.imbalanced import ImbalancedHandler
        X, y = _imbalanced_clf()
        handler = ImbalancedHandler(strategy="undersample")
        X_res, y_res, report = handler.resample(X, y)
        assert report.imbalance_ratio_after <= 1.05
        assert report.n_removed > 0

    def test_weights_no_data_change(self):
        from physml.imbalanced import ImbalancedHandler
        X, y = _imbalanced_clf()
        handler = ImbalancedHandler(strategy="weights")
        X_res, y_res, report = handler.resample(X, y)
        assert len(X_res) == len(X)
        assert report.n_added == 0
        assert report.n_removed == 0

    def test_report_fields(self):
        from physml.imbalanced import ImbalancedHandler, ImbalanceReport
        X, y = _imbalanced_clf()
        handler = ImbalancedHandler(strategy="oversample")
        _, _, report = handler.resample(X, y)
        assert isinstance(report, ImbalanceReport)
        assert "imbalance_ratio_before" in report.as_dict()
        assert report.imbalance_ratio_before > 1.0

    def test_report_as_dict(self):
        from physml.imbalanced import ImbalancedHandler
        X, y = _imbalanced_clf()
        _, _, report = ImbalancedHandler(strategy="undersample").resample(X, y)
        d = report.as_dict()
        assert "strategy" in d
        assert "class_counts_before" in d
        assert "class_counts_after" in d

    def test_compute_weights_shape(self):
        from physml.imbalanced import ImbalancedHandler
        y = np.array([0] * 80 + [1] * 20)
        handler = ImbalancedHandler()
        weights = handler.compute_weights(y)
        assert weights.shape == (100,)
        # minority class should have higher weight
        assert weights[80:].mean() > weights[:80].mean()

    def test_compute_weights_mean_one(self):
        from physml.imbalanced import ImbalancedHandler
        y = np.array([0] * 60 + [1] * 40)
        weights = ImbalancedHandler().compute_weights(y)
        assert abs(weights.mean() - 1.0) < 1e-6

    def test_oversample_preserves_original_rows(self):
        from physml.imbalanced import ImbalancedHandler
        X, y = _imbalanced_clf()
        _, y_res, _ = ImbalancedHandler(strategy="oversample").resample(X, y)
        assert (y_res == 0).sum() >= 150  # majority class unchanged

    def test_undersample_result_smaller(self):
        from physml.imbalanced import ImbalancedHandler
        X, y = _imbalanced_clf()
        X_res, _, _ = ImbalancedHandler(strategy="undersample").resample(X, y)
        assert len(X_res) < len(X)

    def test_multiclass_oversample(self):
        from physml.imbalanced import ImbalancedHandler
        rng = np.random.default_rng(42)
        X = rng.standard_normal((120, 4))
        y = np.array([0] * 60 + [1] * 40 + [2] * 20)
        handler = ImbalancedHandler(strategy="oversample")
        X_res, y_res, report = handler.resample(X, y)
        assert report.imbalance_ratio_after <= 1.05

    def test_target_ratio_partial(self):
        from physml.imbalanced import ImbalancedHandler
        X, y = _imbalanced_clf(n_majority=150, n_minority=30)
        handler = ImbalancedHandler(strategy="oversample", target_ratio=0.5)
        _, _, report = handler.resample(X, y)
        # With ratio 0.5, minority target = 150*0.5=75 (not fully balanced)
        assert report.n_added >= 0

    def test_public_api_via_init(self):
        from physml import ImbalancedHandler, ImbalanceReport
        assert ImbalancedHandler
        assert ImbalanceReport


# ===========================================================================
# Stage 83 — OnlineEvaluator
# ===========================================================================

class TestOnlineEvaluator:

    def test_import(self):
        from physml.online_evaluator import OnlineEvaluator, EvalWindow
        assert OnlineEvaluator
        assert EvalWindow

    def test_invalid_task(self):
        from physml.online_evaluator import OnlineEvaluator
        with pytest.raises(ValueError):
            OnlineEvaluator(task="segmentation")

    def test_classification_update_emits_window(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="classification", window_size=50, step_size=25)
        rng = np.random.default_rng(0)
        y_pred = rng.integers(0, 2, 30)
        y_true = rng.integers(0, 2, 30)
        windows = ev.update(y_pred, y_true)
        # 30 >= 25, so at least one window should be emitted
        assert len(windows) >= 1

    def test_window_fields(self):
        from physml.online_evaluator import OnlineEvaluator, EvalWindow
        ev = OnlineEvaluator(task="classification", window_size=30, step_size=30)
        rng = np.random.default_rng(1)
        y_pred = rng.integers(0, 2, 30)
        y_true = rng.integers(0, 2, 30)
        windows = ev.update(y_pred, y_true)
        assert len(windows) >= 1
        w = windows[0]
        assert isinstance(w, EvalWindow)
        assert 0.0 <= w.accuracy <= 1.0
        assert w.f1_macro is not None
        assert w.mae is None  # classification has no mae
        assert w.n_samples == 30

    def test_window_as_dict(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="classification", window_size=20, step_size=20)
        rng = np.random.default_rng(0)
        ev.update(rng.integers(0, 2, 20), rng.integers(0, 2, 20))
        d = ev.windows[0].as_dict()
        assert "window_id" in d
        assert "accuracy" in d
        assert "f1_macro" in d

    def test_regression_window_fields(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="regression", window_size=30, step_size=30)
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal(30)
        y_true = rng.standard_normal(30)
        windows = ev.update(y_pred, y_true)
        assert len(windows) >= 1
        w = windows[0]
        assert w.mae is not None
        assert w.rmse is not None
        assert w.accuracy is None

    def test_regression_window_as_dict(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="regression", window_size=20, step_size=20)
        rng = np.random.default_rng(0)
        ev.update(rng.standard_normal(20), rng.standard_normal(20))
        d = ev.windows[0].as_dict()
        assert "mae" in d
        assert "rmse" in d
        assert "accuracy" not in d

    def test_multiple_batches_accumulate(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="classification", window_size=50, step_size=25)
        rng = np.random.default_rng(0)
        for _ in range(5):
            ev.update(rng.integers(0, 2, 20), rng.integers(0, 2, 20))
        assert len(ev.windows) >= 2

    def test_n_total_counts(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(window_size=100, step_size=50)
        rng = np.random.default_rng(0)
        ev.update(rng.integers(0, 2, 40), rng.integers(0, 2, 40))
        ev.update(rng.integers(0, 2, 30), rng.integers(0, 2, 30))
        assert ev.n_total == 70

    def test_global_metrics(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="classification", window_size=50, step_size=25)
        rng = np.random.default_rng(0)
        ev.update(rng.integers(0, 2, 60), rng.integers(0, 2, 60))
        metrics = ev.global_metrics()
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_flush_emits_remaining(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="classification", window_size=100, step_size=100)
        rng = np.random.default_rng(0)
        ev.update(rng.integers(0, 2, 40), rng.integers(0, 2, 40))
        # No windows yet (40 < 100)
        assert len(ev.windows) == 0
        w = ev.flush()
        assert w is not None
        assert w.n_samples == 40

    def test_flush_empty_returns_none(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator()
        assert ev.flush() is None

    def test_perfect_accuracy_window(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="classification", window_size=20, step_size=20)
        y = np.arange(20) % 3
        ev.update(y, y)
        assert ev.windows[0].accuracy == 1.0

    def test_perfect_zero_mae_window(self):
        from physml.online_evaluator import OnlineEvaluator
        ev = OnlineEvaluator(task="regression", window_size=20, step_size=20)
        y = np.ones(20)
        ev.update(y, y)
        assert ev.windows[0].mae == pytest.approx(0.0, abs=1e-6)

    def test_public_api_via_init(self):
        from physml import OnlineEvaluator, EvalWindow
        assert OnlineEvaluator
        assert EvalWindow


# ===========================================================================
# Stage 84 — ModelZoo
# ===========================================================================

class TestModelZoo:

    def test_import(self):
        from physml.model_zoo import ModelZoo, ZooEntry
        assert ModelZoo
        assert ZooEntry

    def test_default_zoo_nonempty(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        assert len(zoo) >= 5

    def test_get_existing(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        entry = zoo.get("lr_fast")
        assert entry is not None
        assert entry.name == "lr_fast"

    def test_get_missing_returns_none(self):
        from physml.model_zoo import ModelZoo
        assert ModelZoo().get("nonexistent_xyz") is None

    def test_build_returns_estimator(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        model = zoo.build("lr_fast")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_missing_raises_key_error(self):
        from physml.model_zoo import ModelZoo
        with pytest.raises(KeyError):
            ModelZoo().build("does_not_exist")

    def test_entry_build_fresh_instance(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        m1 = zoo.build("rf_fast")
        m2 = zoo.build("rf_fast")
        assert m1 is not m2

    def test_search_by_task(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        results = zoo.search(task="classification")
        assert all(e.task in ("classification", "any") for e in results)
        assert len(results) >= 3

    def test_search_by_tier(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        fast = zoo.search(tier="fast")
        assert all(e.tier == "fast" for e in fast)
        assert len(fast) >= 1

    def test_search_by_tags(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        results = zoo.search(tags=["linear"])
        assert all("linear" in e.tags for e in results)

    def test_search_combined_filters(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        results = zoo.search(task="classification", tier="fast")
        assert all(e.task in ("classification", "any") and e.tier == "fast" for e in results)

    def test_register_custom_entry(self):
        from physml.model_zoo import ModelZoo, ZooEntry
        zoo = ModelZoo()
        entry = ZooEntry(
            name="my_lr",
            task="classification",
            tier="fast",
            description="Custom LR",
            tags=["custom"],
            factory=lambda: LogisticRegression(max_iter=100),
        )
        zoo.register(entry)
        assert "my_lr" in zoo
        assert zoo.build("my_lr") is not None

    def test_list_names(self):
        from physml.model_zoo import ModelZoo
        names = ModelZoo().list_names()
        assert isinstance(names, list)
        assert "lr_fast" in names

    def test_summary_structure(self):
        from physml.model_zoo import ModelZoo
        summary = ModelZoo().summary()
        assert isinstance(summary, list)
        assert all("name" in d and "task" in d and "tier" in d for d in summary)

    def test_entry_as_dict(self):
        from physml.model_zoo import ModelZoo
        entry = ModelZoo().get("rf_fast")
        d = entry.as_dict()
        assert "name" in d
        assert "task" in d
        assert "tier" in d
        assert "tags" in d

    def test_entry_repr(self):
        from physml.model_zoo import ModelZoo
        entry = ModelZoo().get("lr_fast")
        r = repr(entry)
        assert "ZooEntry" in r
        assert "lr_fast" in r

    def test_contains_operator(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        assert "lr_fast" in zoo
        assert "nonexistent" not in zoo

    @pytest.mark.slow
    def test_compare_method(self):
        from physml.model_zoo import ModelZoo
        X, y = _clf_data(n=200)
        zoo = ModelZoo()
        results = zoo.compare(["lr_fast", "rf_fast"], X, y)
        assert len(results) == 2
        assert all("name" in r and "score" in r for r in results)
        # Results are sorted descending by score
        assert results[0]["score"] >= results[-1]["score"]

    def test_empty_zoo(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo(include_defaults=False)
        assert len(zoo) == 0
        assert zoo.search() == []

    def test_regression_presets_present(self):
        from physml.model_zoo import ModelZoo
        zoo = ModelZoo()
        reg_entries = zoo.search(task="regression")
        assert len(reg_entries) >= 2

    def test_public_api_via_init(self):
        from physml import ModelZoo, ZooEntry
        assert ModelZoo
        assert ZooEntry
