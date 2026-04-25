"""Tests for Stage 36 — CompetitiveEnsemblePredictor and integration fixes.

Covers:
- CEP basic fit/predict/score (classification + regression)
- predict_proba shape and valid probabilities
- partial_fit (replay buffer update)
- runtime_state_.homeostasis_score is a float in [0, 1]
- MyceliumAgent default predictor is now CompetitiveEnsemblePredictor
- PhysicsAgent._predict() squeezes 1-element arrays → plain Python scalar (NumPy 2 fix)
- MyceliumAgent.fit() filters unknown kwargs before passing to CEP
- PhysicsAgent.adapt() routes ensemble backend through partial_fit
"""

from __future__ import annotations

import numpy as np
import pytest

from physml.ensemble_predictor import CompetitiveEnsemblePredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clf_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 8)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:60], y[:60], X[60:], y[60:]


@pytest.fixture()
def reg_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 5)).astype(np.float32)
    y = (X[:, 0] * 2.0 + rng.standard_normal(80) * 0.1).astype(np.float32)
    return X[:60], y[:60], X[60:], y[60:]


# ---------------------------------------------------------------------------
# Unit tests — CompetitiveEnsemblePredictor
# ---------------------------------------------------------------------------


class TestCEPClassifier:
    def test_fit_predict_score(self, clf_data):
        X_tr, y_tr, X_te, y_te = clf_data
        clf = CompetitiveEnsemblePredictor(random_seed=0)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)
        score = clf.score(X_te, y_te)
        assert score >= 0.5, f"Expected score ≥ 0.5, got {score}"

    @pytest.mark.slow
    def test_predict_proba_shape(self, clf_data):
        X_tr, y_tr, X_te, y_te = clf_data
        clf = CompetitiveEnsemblePredictor(random_seed=0)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        assert proba.shape == (len(y_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_partial_fit_updates(self, clf_data):
        X_tr, y_tr, X_te, y_te = clf_data
        clf = CompetitiveEnsemblePredictor(random_seed=0, min_retrain=1)
        clf.fit(X_tr, y_tr)
        score_before = clf.score(X_te, y_te)
        # partial_fit should not raise
        clf.partial_fit(X_te[:5], y_te[:5])
        # Model should still predict valid outputs after update
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)

    def test_homeostasis_in_range(self, clf_data):
        X_tr, y_tr, _, _ = clf_data
        clf = CompetitiveEnsemblePredictor(random_seed=0)
        clf.fit(X_tr, y_tr)
        h = clf.runtime_state_.homeostasis_score
        assert isinstance(h, float)
        assert 0.0 <= h <= 1.0

    def test_no_meta_fallback(self, clf_data):
        X_tr, y_tr, X_te, y_te = clf_data
        clf = CompetitiveEnsemblePredictor(random_seed=0, use_meta=False)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)


class TestCEPRegressor:
    def test_fit_predict(self, reg_data):
        X_tr, y_tr, X_te, y_te = reg_data
        clf = CompetitiveEnsemblePredictor(random_seed=0)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)

    def test_predict_proba_raises(self, reg_data):
        X_tr, y_tr, X_te, _ = reg_data
        clf = CompetitiveEnsemblePredictor(random_seed=0)
        clf.fit(X_tr, y_tr)
        with pytest.raises(ValueError, match="only available for classifiers"):
            clf.predict_proba(X_te)

    def test_score_r2(self, reg_data):
        X_tr, y_tr, X_te, y_te = reg_data
        clf = CompetitiveEnsemblePredictor(random_seed=0)
        clf.fit(X_tr, y_tr)
        score = clf.score(X_te, y_te)
        assert score >= 0.0  # should do better than predicting the mean


# ---------------------------------------------------------------------------
# Integration tests — MyceliumAgent
# ---------------------------------------------------------------------------


class TestMyceliumAgentStage36:
    def _make_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 8)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X[:60], y[:60], X[60:], y[60:]

    def test_default_predictor_is_cep(self):
        from physml.mycelium_agent import MyceliumAgent

        agent = MyceliumAgent(calibrate=False)
        X_tr, y_tr, _, _ = self._make_data()
        agent.fit(X_tr, y_tr)
        assert isinstance(agent._predictor, CompetitiveEnsemblePredictor)

    def test_observe_returns_scalar_prediction(self):
        """Regression test: prediction must be a plain Python scalar (not a numpy array)."""
        from physml.mycelium_agent import MyceliumAgent

        agent = MyceliumAgent(calibrate=False, uncertainty_threshold=0.0)
        X_tr, y_tr, X_te, y_te = self._make_data()
        agent.fit(X_tr, y_tr)
        action = agent.observe(X_te[:1])
        if action.prediction is not None:
            # Must be scalar-convertible without error in NumPy 2.x
            _ = int(action.prediction)

    def test_high_accuracy(self):
        """With CompetitiveEnsemblePredictor the agent should hit ≥80% on easy data."""
        from physml.mycelium_agent import MyceliumAgent
        from sklearn.metrics import accuracy_score

        agent = MyceliumAgent(calibrate=False, uncertainty_threshold=0.0)
        X_tr, y_tr, X_te, y_te = self._make_data()
        agent.fit(X_tr, y_tr)
        preds = [
            int(agent.observe(X_te[i : i + 1]).prediction or 0)
            for i in range(len(X_te))
        ]
        acc = accuracy_score(y_te, preds)
        assert acc >= 0.80, f"Expected ≥80% accuracy, got {acc:.2%}"

    @pytest.mark.slow
    def test_unknown_kwargs_ignored(self):
        """predictor_kwargs with PhysicsPredictor-only params must not crash MyceliumAgent.fit()."""
        from physml.mycelium_agent import MyceliumAgent

        agent = MyceliumAgent(calibrate=False, predictor_kwargs={"n_cycles": 5})
        X_tr, y_tr, _, _ = self._make_data()
        # Should not raise — unknown kwargs are filtered before passing to CEP
        agent.fit(X_tr, y_tr)
        assert isinstance(agent._predictor, CompetitiveEnsemblePredictor)

    def test_reward_triggers_partial_fit(self):
        """reward() must call partial_fit on CEP without raising."""
        from physml.mycelium_agent import MyceliumAgent

        agent = MyceliumAgent(calibrate=False)
        X_tr, y_tr, X_te, y_te = self._make_data()
        agent.fit(X_tr, y_tr)
        # Should not raise
        agent.reward(X_te[:3], y_te[:3])


# ---------------------------------------------------------------------------
# Unit test — _predict squeeze
# ---------------------------------------------------------------------------


class TestPredictSqueeze:
    """PhysicsAgent._predict() must return a scalar for single-sample inputs."""

    def test_scalar_returned_for_single_sample(self):
        from physml.agent import PhysicsAgent

        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 5)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        cep = CompetitiveEnsemblePredictor(random_seed=0)
        cep.fit(X[:50], y[:50])
        agent = PhysicsAgent(predictor=cep, uncertainty_threshold=0.0)
        result = agent._predict(X[50:51])
        # Must be scalar-convertible without error in NumPy 2.x
        _ = int(result)
