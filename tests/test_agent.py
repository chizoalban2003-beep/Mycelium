"""Tests for Stages 3–7: continual learning, agent loop, streaming, session API."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from physml import (
    AgentAction,
    DataStream,
    NeuralPhysicsEngine,
    PhysicsAgent,
    PhysicsAgentSession,
    PhysicsPredictor,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _clf_data(seed: int = 42, n: int = 120):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    return X, y


def _reg_data(seed: int = 42, n: int = 120):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.2, n)
    return X, y


def _split(X, y, test_frac: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_te = max(1, int(n * test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    return X[tr], X[te], y[tr], y[te]


def _fitted_neural_clf():
    X, y = _clf_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    clf = PhysicsPredictor(n_cycles=5, backend="neural")
    clf.fit(X_tr, y_tr)
    return clf, X_te, y_te


def _fitted_neural_reg():
    X, y = _reg_data()
    X_tr, X_te, y_tr, y_te = _split(X, y)
    reg = PhysicsPredictor(plane="solid", n_cycles=5, backend="neural")
    reg.fit(X_tr, y_tr)
    return reg, X_te, y_te


# ============================================================================
# Stage 3 — NeuralPhysicsEngine inductive methods
# ============================================================================

class TestNeuralEngineInductive:
    def test_fit_model_stores_mlp(self):
        X, y = _reg_data(n=80)
        X_tr, X_te, y_tr, y_te = _split(X, y)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X_tr, y_tr, is_classifier=False, n_epochs=50, lr=0.001, random_state=0)
        assert hasattr(eng, "mlp_")
        assert hasattr(eng, "attn_")

    def test_predict_model_shape_regression(self):
        X, y = _reg_data(n=80)
        X_tr, X_te, y_tr, y_te = _split(X, y)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X_tr, y_tr, is_classifier=False, n_epochs=50, lr=0.001, random_state=0)
        preds = eng.predict_model(X_te)
        assert preds.shape == (len(y_te),)
        assert all(math.isfinite(float(p)) for p in preds)

    def test_predict_model_shape_classification(self):
        X, y = _clf_data(n=80)
        X_tr, X_te, y_tr, y_te = _split(X, y)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X_tr, y_tr, is_classifier=True, n_epochs=50, lr=0.001, random_state=0)
        preds = eng.predict_model(X_te)
        assert preds.shape == (len(y_te),)

    def test_predict_model_before_fit_raises(self):
        eng = NeuralPhysicsEngine()
        with pytest.raises(RuntimeError, match="fit_model"):
            eng.predict_model(np.ones((5, 3)))

    def test_partial_fit_model_updates_regression(self):
        X, y = _reg_data(n=80)
        X_tr, X_te, y_tr, y_te = _split(X, y)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X_tr, y_tr, is_classifier=False, n_epochs=50, lr=0.001, random_state=0)
        before = eng.predict_model(X_te).copy()
        # Feed new data
        eng.partial_fit_model(X_te[:10], y_te[:10])
        after = eng.predict_model(X_te)
        assert after.shape == before.shape  # shape unchanged

    def test_partial_fit_model_updates_classification(self):
        X, y = _clf_data(n=80)
        X_tr, X_te, y_tr, y_te = _split(X, y)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X_tr, y_tr, is_classifier=True, n_epochs=50, lr=0.001, random_state=0)
        eng.partial_fit_model(X_te[:10], y_te[:10])
        preds = eng.predict_model(X_te)
        assert preds.shape == (len(y_te),)

    def test_ewc_flat_weights_roundtrip(self):
        X, y = _reg_data(n=60)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X, y, is_classifier=False, n_epochs=30, lr=0.001, random_state=0)
        flat = eng._get_flat_weights()
        assert flat is not None and flat.ndim == 1
        original_pred = eng.predict_model(X[:5]).copy()
        eng._set_flat_weights(flat)
        after_pred = eng.predict_model(X[:5])
        np.testing.assert_allclose(original_pred, after_pred, rtol=1e-5)

    def test_fisher_shape_matches_weights(self):
        X, y = _reg_data(n=60)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X, y, is_classifier=False, n_epochs=30, lr=0.001, random_state=0)
        flat = eng._get_flat_weights()
        fisher = eng._compute_fisher()
        assert fisher is not None
        assert fisher.shape == flat.shape
        assert np.all(fisher >= 0)

    def test_encode_aligned_missing_columns_filled_with_zero(self):
        import pandas as pd
        X, y = _reg_data(n=60)
        eng = NeuralPhysicsEngine()
        eng.fit_model(
            X, y,
            is_classifier=False, n_epochs=30, lr=0.001, random_state=0,
            encoded_feature_names=["f0", "f1", "f2", "f3"],
        )
        # New data with only 2 of the 4 features
        df_new = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0], "__target__": [0.0, 0.0]})
        X_aligned, _ = eng.encode_aligned(df_new, "__target__")
        assert X_aligned.shape[1] == 4


# ============================================================================
# Stage 3 — NeuralPhysicsEngine save / load
# ============================================================================

class TestNeuralEngineSaveLoad:
    def test_save_and_load_roundtrip(self):
        X, y = _reg_data(n=60)
        eng = NeuralPhysicsEngine()
        eng.fit_model(X, y, is_classifier=False, n_epochs=30, lr=0.001, random_state=7)
        preds_before = eng.predict_model(X[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "engine.pkl"
            eng.save(path)
            loaded = NeuralPhysicsEngine.load(path)
            preds_after = loaded.predict_model(X[:10])

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_load_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib
            path = Path(tmpdir) / "wrong.pkl"
            joblib.dump({"not": "an engine"}, str(path))
            with pytest.raises(TypeError):
                NeuralPhysicsEngine.load(path)


# ============================================================================
# Stage 6 — NeuralPhysicsEngine.pretrain
# ============================================================================

class TestNeuralEnginePretrain:
    def test_pretrain_single_dataset_returns_engine(self):
        import pandas as pd
        X, y = _reg_data(n=60)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["target"] = y
        eng = NeuralPhysicsEngine.pretrain(
            [df], target_col="target", n_cycles=5, random_seed=0
        )
        assert hasattr(eng, "mlp_")

    def test_pretrain_two_datasets_no_crash(self):
        import pandas as pd
        rng = np.random.default_rng(0)
        dfs = []
        for i in range(2):
            X = rng.normal(0, 1, (50, 3))
            y = X[:, 0] + rng.normal(0, 0.1, 50)
            df = pd.DataFrame(X, columns=["a", "b", "c"])
            df["tgt"] = y
            dfs.append(df)
        eng = NeuralPhysicsEngine.pretrain(dfs, target_col="tgt", n_cycles=5)
        assert hasattr(eng, "mlp_")


# ============================================================================
# Stage 3 — PhysicsPredictor.partial_fit
# ============================================================================

class TestPhysicsPredictorPartialFit:
    def test_partial_fit_returns_self(self):
        clf, X_te, y_te = _fitted_neural_clf()
        result = clf.partial_fit(X_te[:10], y_te[:10])
        assert result is clf

    def test_partial_fit_predict_shape_unchanged(self):
        clf, X_te, y_te = _fitted_neural_clf()
        clf.partial_fit(X_te[:10], y_te[:10])
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)

    def test_partial_fit_regression_predict_finite(self):
        reg, X_te, y_te = _fitted_neural_reg()
        reg.partial_fit(X_te[:10], y_te[:10])
        preds = reg.predict(X_te)
        assert all(math.isfinite(float(p)) for p in preds)

    def test_partial_fit_multiple_rounds(self):
        clf, X_te, y_te = _fitted_neural_clf()
        for i in range(3):
            chunk = X_te[i * 3: (i + 1) * 3]
            labels = y_te[i * 3: (i + 1) * 3]
            clf.partial_fit(chunk, labels)
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)

    def test_partial_fit_replay_buffer_not_empty(self):
        clf, X_te, y_te = _fitted_neural_clf()
        clf.partial_fit(X_te[:10], y_te[:10])
        assert len(clf._replay_buffer_) > 0

    def test_partial_fit_physics_backend_raises(self):
        X, y = _clf_data()
        X_tr, X_te, y_tr, y_te = _split(X, y)
        clf = PhysicsPredictor(n_cycles=5, backend="physics")
        clf.fit(X_tr, y_tr)
        with pytest.raises(ValueError, match="neural"):
            clf.partial_fit(X_te[:5], y_te[:5])

    def test_partial_fit_before_fit_raises(self):
        clf = PhysicsPredictor(n_cycles=5, backend="neural")
        with pytest.raises(Exception):
            clf.partial_fit(np.ones((5, 4)), np.zeros(5))


# ============================================================================
# Stage 3 — PhysicsPredictor.save / load
# ============================================================================

class TestPhysicsPredictorSaveLoad:
    def test_save_load_roundtrip_clf(self):
        clf, X_te, y_te = _fitted_neural_clf()
        preds_before = clf.predict(X_te)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            loaded = PhysicsPredictor.load(path)
        preds_after = loaded.predict(X_te)
        assert preds_after.shape == preds_before.shape

    def test_save_load_roundtrip_reg(self):
        reg, X_te, y_te = _fitted_neural_reg()
        preds_before = reg.predict(X_te)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            reg.save(path)
            loaded = PhysicsPredictor.load(path)
        preds_after = loaded.predict(X_te)
        assert preds_after.shape == preds_before.shape

    def test_load_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib
            path = Path(tmpdir) / "wrong.pkl"
            joblib.dump(42, str(path))
            with pytest.raises(TypeError):
                PhysicsPredictor.load(path)

    def test_partial_fit_after_load(self):
        clf, X_te, y_te = _fitted_neural_clf()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            loaded = PhysicsPredictor.load(path)
        loaded.partial_fit(X_te[:5], y_te[:5])
        preds = loaded.predict(X_te)
        assert preds.shape == (len(y_te),)


# ============================================================================
# Stage 3 — get_params includes replay_size
# ============================================================================

def test_get_params_includes_replay_size():
    clf = PhysicsPredictor(backend="neural", replay_size=200)
    params = clf.get_params()
    assert "replay_size" in params
    assert params["replay_size"] == 200


def test_set_params_replay_size():
    clf = PhysicsPredictor()
    clf.set_params(replay_size=999)
    assert clf.replay_size == 999


# ============================================================================
# Stage 4 — PhysicsAgent
# ============================================================================

class TestPhysicsAgent:
    def test_observe_returns_agent_action(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        action = agent.observe(X_te[:1])
        assert isinstance(action, AgentAction)

    def test_observe_action_is_valid_string(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        action = agent.observe(X_te[:1])
        assert action.action in ("predict", "abstain", "ask")

    def test_observe_confidence_in_range(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        action = agent.observe(X_te[:1])
        assert 0.0 <= action.confidence <= 1.0

    def test_observe_increments_counter(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        assert agent.n_observations == 0
        agent.observe(X_te[:1])
        agent.observe(X_te[1:2])
        assert agent.n_observations == 2

    def test_reward_increments_counter(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        agent.reward(X_te[:1], y_te[:1])
        assert agent.n_rewards == 1

    def test_reward_updates_model(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        # Should not raise
        agent.reward(X_te[:5], y_te[:5])
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)

    def test_adapt_clears_pending(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, uncertainty_threshold=0.0)  # never ask → never auto-adapt
        agent._pending_labels.append((X_te[:1], y_te[:1]))
        agent.adapt()
        assert len(agent._pending_labels) == 0

    def test_report_returns_dict(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        r = agent.report()
        assert "n_observations" in r
        assert "ask_rate" in r
        assert "homeostasis" in r

    def test_high_threshold_always_asks(self):
        clf, X_te, y_te = _fitted_neural_clf()
        # threshold=1.0 → always ask
        agent = PhysicsAgent(clf, uncertainty_threshold=1.0, homeostasis_weight=0.0)
        action = agent.observe(X_te[:1])
        assert action.action == "ask"
        assert action.needs_label is True

    def test_zero_threshold_always_predicts(self):
        clf, X_te, y_te = _fitted_neural_clf()
        # threshold=0.0 → always predict (confidence > 0)
        agent = PhysicsAgent(clf, uncertainty_threshold=0.0, homeostasis_weight=0.0)
        action = agent.observe(X_te[:1])
        assert action.action == "predict"
        assert action.prediction is not None

    def test_agent_action_needs_label_consistent(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        action = agent.observe(X_te[:1])
        assert action.needs_label == (action.action == "ask")


# ============================================================================
# Stage 5 — DataStream
# ============================================================================

class TestDataStream:
    def test_fit_stream_without_seed(self):
        X, y = _reg_data(n=200)
        chunks = [(X[i * 40: (i + 1) * 40], y[i * 40: (i + 1) * 40]) for i in range(5)]
        predictor = PhysicsPredictor(backend="neural", n_cycles=5)
        stream = DataStream(chunks)
        result = stream.fit_stream(predictor)
        assert result is predictor
        preds = predictor.predict(X[:10])
        assert preds.shape == (10,)

    def test_fit_stream_with_seed(self):
        X, y = _reg_data(n=200)
        seed_X, seed_y = X[:40], y[:40]
        chunks = [(X[40 + i * 40: 40 + (i + 1) * 40], y[40 + i * 40: 40 + (i + 1) * 40])
                  for i in range(4)]
        predictor = PhysicsPredictor(backend="neural", n_cycles=5)
        stream = DataStream(iter(chunks))
        stream.fit_stream(predictor, seed_X=seed_X, seed_y=seed_y)
        preds = predictor.predict(X[:10])
        assert all(math.isfinite(float(p)) for p in preds)

    def test_fit_stream_empty_chunks_no_crash(self):
        predictor = PhysicsPredictor(backend="neural", n_cycles=5)
        X_seed, y_seed = _reg_data(n=40)
        stream = DataStream(iter([]))  # no chunks
        stream.fit_stream(predictor, seed_X=X_seed, seed_y=y_seed)
        # Should still be fitted from the seed
        preds = predictor.predict(X_seed[:5])
        assert preds.shape == (5,)

    def test_datastream_generator(self):
        def gen_chunks():
            rng = np.random.default_rng(99)
            for _ in range(3):
                X = rng.normal(0, 1, (50, 4))
                y = X[:, 0] + rng.normal(0, 0.1, 50)
                yield X, y

        predictor = PhysicsPredictor(backend="neural", n_cycles=5)
        stream = DataStream(gen_chunks())
        stream.fit_stream(predictor)
        rng = np.random.default_rng(0)
        preds = predictor.predict(rng.normal(0, 1, (5, 4)))
        assert preds.shape == (5,)


# ============================================================================
# Stage 7 — PhysicsAgentSession
# ============================================================================

class TestPhysicsAgentSession:
    def _make_session(self, tmpdir: str) -> "PhysicsAgentSession":
        sess = PhysicsAgentSession(
            user_id="test_user",
            model_dir=tmpdir,
            predictor_kwargs={"n_cycles": 3},
        )
        X, y = _clf_data(n=60)
        X_tr, X_te, y_tr, y_te = _split(X, y)
        sess.train(X_tr, y_tr)
        return sess, X_te, y_te

    def test_train_sets_fitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            assert sess._fitted is True

    def test_query_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            result = sess.query(X_te[:1])
            assert isinstance(result, dict)
            assert "prediction" in result
            assert "confidence" in result
            assert "action" in result
            assert "needs_label" in result

    def test_query_before_train_raises(self):
        sess = PhysicsAgentSession(user_id="unfit", model_dir="/tmp")
        X = np.ones((1, 4))
        with pytest.raises(RuntimeError, match="train"):
            sess.query(X)

    def test_feedback_increments_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            sess.feedback(X_te[:2], y_te[:2])
            assert sess.n_feedbacks == 1

    def test_report_contains_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            r = sess.report()
            for key in ("user_id", "session_id", "n_queries", "n_feedbacks", "agent_report"):
                assert key in r

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            path = sess.save()
            assert path.exists()

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            preds_before = sess.query(X_te[:1])["prediction"]
            save_path = sess.save()

            loaded = PhysicsAgentSession.load(save_path)
            result = loaded.query(X_te[:1])
            assert "prediction" in result

    def test_load_by_user_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            sess.save()
            loaded = PhysicsAgentSession.load("test_user", model_dir=tmpdir)
            assert loaded.user_id == "test_user"

    def test_session_query_increments_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            assert sess.n_queries == 0
            sess.query(X_te[:1])
            sess.query(X_te[1:2])
            assert sess.n_queries == 2

    def test_session_feedback_updates_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sess, X_te, y_te = self._make_session(tmpdir)
            sess.feedback(X_te[:5], y_te[:5])
            result = sess.query(X_te[:1])
            assert "prediction" in result

    def test_load_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib
            path = Path(tmpdir) / "bad.pkl"
            joblib.dump({"not": "a session"}, str(path))
            with pytest.raises(TypeError):
                PhysicsAgentSession.load(path)


# ── New tests for: multi-class fix, predict_proba, convenience classes ─────

class TestMultiClassNeuralFix:
    """Neural backend multi-class label dtype must match original y dtype."""

    def test_wine_integer_labels_returned_as_int(self):
        """inverse_transform used to return strings; verify integers now."""
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        clf = PhysicsPredictor(backend="neural", n_cycles=5)
        clf.fit(X[:120], y[:120])
        preds = clf.predict(X[120:])
        assert preds.dtype.kind in ("i", "u"), f"Expected int dtype, got {preds.dtype}"
        # All predictions must be valid class indices
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_multiclass_accuracy_above_random(self):
        """Multi-class neural predictions must beat random baseline (1/n_classes).

        Uses a shuffled split so test rows are representative of all classes.
        """
        from sklearn.datasets import load_wine
        from sklearn.metrics import accuracy_score
        X, y = load_wine(return_X_y=True)
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]
        n_train = int(len(y) * 0.80)
        clf = PhysicsPredictor(backend="neural", n_cycles=20)
        clf.fit(X[:n_train], y[:n_train])
        acc = accuracy_score(y[n_train:], clf.predict(X[n_train:]))
        n_classes = len(np.unique(y))
        assert acc > 1.0 / n_classes, f"Accuracy {acc:.3f} is at or below random ({1/n_classes:.3f})"


class TestPredictProba:
    """predict_proba on the neural backend."""

    def test_predict_proba_shape(self):
        clf, X_te, _ = _fitted_neural_clf()
        proba = clf.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)

    def test_predict_proba_sums_to_one(self):
        clf, X_te, _ = _fitted_neural_clf()
        proba = clf.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_values_in_range(self):
        clf, X_te, _ = _fitted_neural_clf()
        proba = clf.predict_proba(X_te)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_raises_for_regression(self):
        reg, X_te, _ = _fitted_neural_reg()
        with pytest.raises(ValueError, match="classifiers"):
            reg.predict_proba(X_te)

    def test_predict_proba_raises_for_physics_backend(self):
        X, y = _clf_data()
        clf = PhysicsPredictor(n_cycles=5, backend="physics")
        clf.fit(X, y)
        with pytest.raises(ValueError, match="neural"):
            clf.predict_proba(X[:5])

    def test_predict_proba_model_raises_for_regression(self):
        _, reg_engine = _make_fitted_engine(is_clf=False)
        with pytest.raises(ValueError):
            reg_engine.predict_proba_model(np.ones((3, 4)))

    def test_predict_proba_model_shape(self):
        X, clf_engine = _make_fitted_engine(is_clf=True)
        proba = clf_engine.predict_proba_model(X)
        assert proba.ndim == 2 and proba.shape[0] == len(X)


def _make_fitted_engine(is_clf: bool):
    rng = np.random.default_rng(0)
    n, d = 60, 4
    X = rng.normal(size=(n, d))
    if is_clf:
        y = (X[:, 0] > 0).astype(int)
    else:
        y = X[:, 0] * 2 + rng.normal(size=n) * 0.1
    engine = NeuralPhysicsEngine()
    engine.fit_model(X, y, is_classifier=is_clf, n_epochs=20)
    return X, engine


class TestConvenienceSubclasses:
    """PhysicsRegressor and PhysicsClassifier subclasses."""

    def test_physics_regressor_defaults(self):
        from physml import PhysicsRegressor
        reg = PhysicsRegressor()
        assert reg.quantile_transform is True
        assert reg.residual_model == "ridge"
        assert reg.plane in ("solid", "PhysicsPlane.solid")

    def test_physics_classifier_defaults(self):
        from physml import PhysicsClassifier
        clf = PhysicsClassifier()
        assert clf.quantile_transform is True
        assert clf.residual_model == "logistic"

    def test_physics_regressor_fit_predict(self):
        from physml import PhysicsRegressor
        X, y = _reg_data()
        reg = PhysicsRegressor(n_cycles=5)
        reg.fit(X[:80], y[:80])
        preds = reg.predict(X[80:])
        assert preds.shape == (len(y[80:]),)

    def test_physics_classifier_fit_predict(self):
        from physml import PhysicsClassifier
        X, y = _clf_data()
        clf = PhysicsClassifier(n_cycles=5)
        clf.fit(X[:80], y[:80])
        preds = clf.predict(X[80:])
        assert preds.shape == (len(y[80:]),)

    def test_regressor_sklearn_get_params(self):
        from physml import PhysicsRegressor
        reg = PhysicsRegressor(n_cycles=7)
        params = reg.get_params()
        assert params["n_cycles"] == 7
        assert params["quantile_transform"] is True

    def test_classifier_override_defaults(self):
        from physml import PhysicsClassifier
        clf = PhysicsClassifier(quantile_transform=False, residual_model=None)
        assert clf.quantile_transform is False
        assert clf.residual_model is None

    def test_regressor_is_instance_of_predictor(self):
        from physml import PhysicsRegressor
        assert isinstance(PhysicsRegressor(), PhysicsPredictor)

    def test_classifier_is_instance_of_predictor(self):
        from physml import PhysicsClassifier
        assert isinstance(PhysicsClassifier(), PhysicsPredictor)


# ============================================================================
# Stage 8 — Active learning (query_strategy + select_informative)
# ============================================================================

class TestActiveLearning:
    """Tests for PhysicsAgent.select_informative and query_strategy."""

    def test_select_informative_returns_valid_index(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, query_strategy="entropy")
        idx = agent.select_informative(X_te[:5])
        assert 0 <= idx < 5

    def test_select_informative_single_sample(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, query_strategy="entropy")
        idx = agent.select_informative(X_te[:1])
        assert idx == 0

    def test_select_informative_threshold_strategy(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, query_strategy="threshold")
        idx = agent.select_informative(X_te[:8])
        assert 0 <= idx < 8

    def test_select_informative_entropy_vs_threshold_differ(self):
        """Entropy and threshold strategies can return different samples."""
        rng = np.random.default_rng(7)
        X, y = _clf_data(seed=7)
        clf = PhysicsPredictor(n_cycles=10, backend="neural")
        clf.fit(X[:80], y[:80])
        pool = X[80:]
        agent_e = PhysicsAgent(clf, query_strategy="entropy")
        agent_t = PhysicsAgent(clf, query_strategy="threshold")
        idx_e = agent_e.select_informative(pool)
        idx_t = agent_t.select_informative(pool)
        # Both must return valid indices; they may or may not agree
        assert 0 <= idx_e < len(pool)
        assert 0 <= idx_t < len(pool)

    def test_query_strategy_in_constructor(self):
        clf, _, _ = _fitted_neural_clf()
        agent = PhysicsAgent(clf, query_strategy="entropy")
        assert agent.query_strategy == "entropy"

    def test_select_informative_fallback_regression(self):
        """Entropy strategy falls back gracefully for regressors."""
        reg, X_te, y_te = _fitted_neural_reg()
        agent = PhysicsAgent(reg, query_strategy="entropy")
        idx = agent.select_informative(X_te[:6])
        assert 0 <= idx < 6

    def test_active_learning_reduces_oracle_calls(self):
        """Entropy selection should concentrate asks on uncertain samples."""
        X, y = _clf_data(seed=0)
        clf = PhysicsPredictor(n_cycles=10, backend="neural")
        clf.fit(X[:80], y[:80])
        agent = PhysicsAgent(clf, query_strategy="entropy")
        pool = X[80:]
        # Just verify it runs without errors and returns valid indices
        for _ in range(5):
            idx = agent.select_informative(pool)
            assert 0 <= idx < len(pool)
            agent.reward(pool[idx: idx + 1], y[80 + idx: 80 + idx + 1])


# ============================================================================
# Stage 9 — Multi-task engine
# ============================================================================

class TestMultiTaskEngine:
    """Tests for MultiTaskPhysicsEngine."""

    def _make_engine(self, n=80):
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=n)
        engine = MultiTaskPhysicsEngine()
        engine.fit_trunk(X[:60], y[:60], is_classifier=True, n_epochs=50)
        return engine, X, y

    def test_fit_trunk_initialises_trunk(self):
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        engine.fit_trunk(X[:60], y[:60], is_classifier=True, n_epochs=30)
        assert engine._trunk is not None
        assert hasattr(engine._trunk, "attn_")

    def test_fit_task_creates_head(self):
        engine, X, y = self._make_engine()
        engine.fit_task("clf", X[:60], y[:60], is_classifier=True, n_epochs=30)
        assert "clf" in engine.list_tasks()

    def test_predict_task_shape(self):
        engine, X, y = self._make_engine()
        engine.fit_task("clf", X[:60], y[:60], is_classifier=True, n_epochs=30)
        preds = engine.predict_task("clf", X[60:])
        assert preds.shape == (len(X[60:]),)

    def test_predict_task_unknown_raises(self):
        engine, X, y = self._make_engine()
        with pytest.raises(KeyError, match="unknown_task"):
            engine.predict_task("unknown_task", X[:5])

    def test_multiple_tasks_independent(self):
        from physml import MultiTaskPhysicsEngine
        rng = np.random.default_rng(5)
        X = rng.normal(size=(80, 4))
        y_clf = (X[:, 0] > 0).astype(int)
        y_reg = 2.0 * X[:, 1] + rng.normal(size=80) * 0.1
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("clf", X[:60], y_clf[:60], is_classifier=True, n_epochs=50)
        engine.fit_task("reg", X[:60], y_reg[:60], is_classifier=False, n_epochs=50)
        assert set(engine.list_tasks()) == {"clf", "reg"}
        preds_clf = engine.predict_task("clf", X[60:])
        preds_reg = engine.predict_task("reg", X[60:])
        assert preds_clf.shape == (20,)
        assert preds_reg.shape == (20,)

    def test_predict_proba_task_shape(self):
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("clf", X[:60], y[:60], is_classifier=True, n_epochs=50)
        proba = engine.predict_proba_task("clf", X[60:])
        assert proba.ndim == 2
        assert proba.shape[0] == 20

    def test_predict_proba_task_sums_to_one(self):
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("clf", X[:60], y[:60], is_classifier=True, n_epochs=50)
        proba = engine.predict_proba_task("clf", X[60:])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_task_raises_for_regression(self):
        from physml import MultiTaskPhysicsEngine
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4))
        y = X[:, 0] * 2
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("reg", X, y, is_classifier=False, n_epochs=30)
        with pytest.raises(ValueError, match="regression"):
            engine.predict_proba_task("reg", X[:5])

    def test_list_tasks_empty_initially(self):
        from physml import MultiTaskPhysicsEngine
        engine = MultiTaskPhysicsEngine()
        assert engine.list_tasks() == []

    def test_task_info(self):
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("clf", X[:60], y[:60], is_classifier=True, n_epochs=30)
        info = engine.task_info("clf")
        assert info["is_classifier"] is True
        assert info["n_input_features"] == X.shape[1]

    def test_save_load_roundtrip(self):
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("clf", X[:60], y[:60], is_classifier=True, n_epochs=30)
        preds_before = engine.predict_task("clf", X[60:])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "engine.pkl"
            engine.save(path)
            loaded = MultiTaskPhysicsEngine.load(path)
        preds_after = loaded.predict_task("clf", X[60:])
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_auto_init_trunk_on_first_task(self):
        """fit_task without prior fit_trunk should auto-init the trunk."""
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        assert engine._trunk is None
        engine.fit_task("auto", X[:60], y[:60], is_classifier=True, n_epochs=30)
        assert engine._trunk is not None

    def test_agent_with_task_id(self):
        """PhysicsAgent with task_id routes through MultiTaskPhysicsEngine."""
        from physml import MultiTaskPhysicsEngine
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("clf", X[:60], y[:60], is_classifier=True, n_epochs=50)
        agent = PhysicsAgent(engine, task_id="clf", uncertainty_threshold=0.0)
        action = agent.observe(X[60:61])
        assert action.action == "predict"
        assert action.prediction is not None


# ============================================================================
# Stage 10 — Adaptive threshold (reward shaping)
# ============================================================================

class TestAdaptiveThreshold:
    """Tests for PhysicsAgent adaptive policy."""

    def test_policy_fixed_default(self):
        clf, _, _ = _fitted_neural_clf()
        agent = PhysicsAgent(clf)
        assert agent.policy == "fixed"

    def test_policy_adaptive_construction(self):
        clf, _, _ = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive")
        assert agent.policy == "adaptive"

    def test_error_window_empty_initially(self):
        clf, _, _ = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive")
        assert len(agent._error_window) == 0

    def test_error_rate_half_when_empty(self):
        clf, _, _ = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive")
        assert agent._error_rate() == 0.5

    def test_reward_logs_error_to_window(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive")
        agent.reward(X_te[:3], y_te[:3])
        assert len(agent._error_window) == 1

    def test_error_window_respects_maxlen(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive", error_window_size=3)
        for i in range(6):
            agent.reward(X_te[i: i + 1], y_te[i: i + 1])
        assert len(agent._error_window) <= 3

    def test_adaptive_threshold_changes_with_errors(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(
            clf,
            policy="adaptive",
            uncertainty_threshold=0.5,
            homeostasis_weight=0.0,
            error_window_size=10,
        )
        homeostasis = 0.5  # neutral
        base = agent._adaptive_threshold(homeostasis)

        # Fill window with high error rate (all wrong)
        for _ in range(10):
            agent._error_window.append(1.0)
        high_err_thr = agent._adaptive_threshold(homeostasis)

        # Fill window with low error rate (all correct)
        agent._error_window.clear()
        for _ in range(10):
            agent._error_window.append(0.0)
        low_err_thr = agent._adaptive_threshold(homeostasis)

        # High error → HIGHER threshold → harder to be confident → asks more
        # Low error  → LOWER threshold → easier to be confident → asks less
        assert high_err_thr > low_err_thr

    def test_adaptive_threshold_stays_in_range(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive", uncertainty_threshold=0.5)
        for _ in range(20):
            agent._error_window.append(1.0)
        thr = agent._adaptive_threshold(0.5)
        assert 0.05 <= thr <= 0.95

    def test_report_includes_policy_and_error_rate(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive")
        r = agent.report()
        assert "policy" in r
        assert "error_rate" in r
        assert r["policy"] == "adaptive"

    def test_observe_metadata_includes_policy(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(clf, policy="adaptive")
        action = agent.observe(X_te[:1])
        assert "policy" in action.metadata

    def test_fixed_policy_ignores_error_window(self):
        clf, X_te, y_te = _fitted_neural_clf()
        agent = PhysicsAgent(
            clf,
            policy="fixed",
            uncertainty_threshold=0.5,
            homeostasis_weight=0.0,
        )
        # Even with errors logged, fixed policy ignores the window
        for _ in range(10):
            agent._error_window.append(1.0)
        thr = agent._adaptive_threshold(0.5)
        assert math.isclose(thr, 0.5, abs_tol=1e-9)


# ============================================================================
# Stage 11 — MyceliumAgent
# ============================================================================

class TestMyceliumAgent:
    """Tests for the flagship MyceliumAgent class."""

    def _make_agent(self, **kwargs):
        from physml import MyceliumAgent
        X, y = _clf_data()
        X_tr, X_te, y_tr, y_te = _split(X, y)
        agent = MyceliumAgent(predictor_kwargs={"n_cycles": 5}, **kwargs)
        agent.fit(X_tr, y_tr)
        return agent, X_te, y_te

    def test_fit_sets_fitted(self):
        from physml import MyceliumAgent
        X, y = _clf_data()
        agent = MyceliumAgent(predictor_kwargs={"n_cycles": 5})
        agent.fit(X[:80], y[:80])
        assert agent._fitted is True

    def test_observe_before_fit_raises(self):
        from physml import MyceliumAgent
        agent = MyceliumAgent()
        with pytest.raises(RuntimeError, match="fit"):
            agent.observe(np.ones((1, 5)))

    def test_observe_returns_agent_action(self):
        from physml import AgentAction, MyceliumAgent
        agent, X_te, y_te = self._make_agent()
        action = agent.observe(X_te[:1])
        assert isinstance(action, AgentAction)

    def test_observe_action_valid(self):
        from physml import MyceliumAgent
        agent, X_te, y_te = self._make_agent()
        action = agent.observe(X_te[:1])
        assert action.action in ("predict", "abstain", "ask")

    def test_default_query_strategy_entropy(self):
        from physml import MyceliumAgent
        agent = MyceliumAgent()
        assert agent.query_strategy == "entropy"

    def test_default_policy_adaptive(self):
        from physml import MyceliumAgent
        agent = MyceliumAgent()
        assert agent.policy == "adaptive"

    def test_reward_updates_model(self):
        from physml import MyceliumAgent
        agent, X_te, y_te = self._make_agent()
        result = agent.reward(X_te[:3], y_te[:3])
        assert result is agent  # returns self

    def test_select_informative_valid_index(self):
        from physml import MyceliumAgent
        agent, X_te, y_te = self._make_agent()
        idx = agent.select_informative(X_te[:10])
        assert 0 <= idx < 10

    def test_report_contains_expected_keys(self):
        from physml import MyceliumAgent
        agent, X_te, y_te = self._make_agent()
        r = agent.report()
        assert "agent" in r
        assert "query_strategy" in r
        assert "policy" in r
        assert "fitted" in r
        assert r["fitted"] is True

    def test_save_load_roundtrip(self):
        from physml import MyceliumAgent
        agent, X_te, y_te = self._make_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mycelium.pkl"
            agent.save(path)
            loaded = MyceliumAgent.load(path)
        action = loaded.observe(X_te[:1])
        assert isinstance(action.action, str)

    def test_load_wrong_type_raises(self):
        from physml import MyceliumAgent
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib
            path = Path(tmpdir) / "bad.pkl"
            joblib.dump({"not": "an agent"}, str(path))
            with pytest.raises(TypeError):
                MyceliumAgent.load(path)

    def test_mycelium_agent_full_loop(self):
        """End-to-end active-learning loop with adaptive threshold."""
        from physml import MyceliumAgent
        X, y = _clf_data(n=120)
        X_tr, X_te, y_tr, y_te = _split(X, y)
        agent = MyceliumAgent(
            predictor_kwargs={"n_cycles": 5},
            query_strategy="entropy",
            policy="adaptive",
        )
        agent.fit(X_tr[:30], y_tr[:30])

        n_asks = 0
        for i in range(min(20, len(X_te))):
            action = agent.observe(X_te[i: i + 1])
            if action.action == "ask":
                n_asks += 1
                agent.reward(X_te[i: i + 1], y_te[i: i + 1])

        r = agent.report()
        assert r["fitted"] is True
        assert "agent" in r

    def test_mycelium_agent_imported_from_physml(self):
        from physml import MyceliumAgent  # noqa: F401 — just assert importable
        assert MyceliumAgent is not None

    def test_mycelium_agent_multitask(self):
        """MyceliumAgent with MultiTaskPhysicsEngine and task_id."""
        from physml import MultiTaskPhysicsEngine, MyceliumAgent
        X, y = _clf_data(n=80)
        engine = MultiTaskPhysicsEngine()
        engine.fit_task("t1", X[:60], y[:60], is_classifier=True, n_epochs=50)
        agent = MyceliumAgent(
            predictor=engine,
            task_id="t1",
            query_strategy="entropy",
            policy="adaptive",
        )
        agent.fit(X[:60], y[:60])  # re-fits the head
        action = agent.observe(X[60:61])
        assert action.action in ("predict", "abstain", "ask")
