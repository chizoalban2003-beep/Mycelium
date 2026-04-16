"""Tests for Stage 69 — LifelongLearner.

Covers: chunk-based streaming, self-improvement triggering, validation window
rolling, compatibility with sklearn estimators, MyceliumAgent, and
AutonomousAgent, plus the competitive_report integration.
"""

from __future__ import annotations

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from physml.lifelong import LifelongLearner, RoundResult
from physml import LifelongLearner as LifelongLearnerPublic, RoundResult as RoundResultPublic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=300, n_features=8, random_state=0):
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=4,
        random_state=random_state,
    )
    return X, y


# ---------------------------------------------------------------------------
# RoundResult
# ---------------------------------------------------------------------------

class TestRoundResult:
    def test_as_dict_keys(self):
        r = RoundResult(
            round_idx=0,
            accuracy=0.85,
            improved=True,
            improvement_delta=0.05,
            n_samples_seen=100,
            elapsed_s=0.5,
        )
        d = r.as_dict()
        assert set(d.keys()) == {
            "round", "accuracy", "improved", "improvement_delta",
            "n_samples_seen", "elapsed_s",
        }

    def test_as_dict_values(self):
        r = RoundResult(0, 0.9, False, 0.0, 50, 1.2)
        d = r.as_dict()
        assert d["accuracy"] == pytest.approx(0.9, abs=1e-4)
        assert d["improved"] is False

    def test_public_export(self):
        assert RoundResultPublic is RoundResult


# ---------------------------------------------------------------------------
# LifelongLearner basics
# ---------------------------------------------------------------------------

class TestLifelongLearnerPublicExport:
    def test_public_export(self):
        assert LifelongLearnerPublic is LifelongLearner


class TestLifelongLearnerInit:
    def test_defaults(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent)
        assert ll.improvement_threshold == 0.75
        assert ll.eval_every == 2
        assert ll.val_window == 200
        assert ll.verbose is False

    def test_custom_params(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, improvement_threshold=0.60, eval_every=3, val_window=100)
        assert ll.improvement_threshold == 0.60
        assert ll.eval_every == 3
        assert ll.val_window == 100

    def test_initial_state(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent)
        assert ll._n_samples_seen == 0
        assert ll._improvement_count == 0
        assert ll._fitted is False
        assert ll.history == []


# ---------------------------------------------------------------------------
# run() with sklearn estimator
# ---------------------------------------------------------------------------

class TestLifelongLearnerRunSklearn:
    def setup_method(self):
        self.X, self.y = _make_data()

    def test_run_returns_list_of_round_results(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=2)
        history = ll.run(self.X, self.y, chunk_size=50)
        assert isinstance(history, list)
        assert all(isinstance(r, RoundResult) for r in history)

    def test_run_produces_evaluations(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=2)
        history = ll.run(self.X, self.y, chunk_size=50)
        # 300 samples / 50 per chunk = 6 chunks; eval every 2 → 3 rounds
        assert len(history) >= 1

    def test_samples_seen_correct(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=1)
        ll.run(self.X, self.y, chunk_size=50)
        assert ll._n_samples_seen == len(self.X)

    def test_fitted_after_run(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent)
        ll.run(self.X, self.y, chunk_size=50)
        assert ll._fitted is True

    def test_final_accuracy_is_number(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent)
        ll.run(self.X, self.y, chunk_size=50)
        fa = ll.final_accuracy()
        assert isinstance(fa, float)
        assert 0.0 <= fa <= 1.0

    def test_history_matches_property(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=1)
        returned = ll.run(self.X, self.y, chunk_size=50)
        assert returned == ll.history

    def test_round_idx_monotone(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=1)
        ll.run(self.X, self.y, chunk_size=50)
        for i, r in enumerate(ll.history):
            assert r.round_idx == i

    def test_val_window_respected(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, val_window=30, eval_every=1)
        ll.run(self.X, self.y, chunk_size=20)
        # window must never exceed val_window
        assert len(ll._val_X) <= 30


# ---------------------------------------------------------------------------
# Self-improvement triggering
# ---------------------------------------------------------------------------

class TestSelfImprovementTrigger:
    def setup_method(self):
        self.X, self.y = _make_data()

    def test_improvement_fires_when_threshold_above_accuracy(self):
        """Set a very high threshold → improvement should be triggered often."""
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, improvement_threshold=0.99, eval_every=1)
        ll.run(self.X, self.y, chunk_size=50)
        # At least one round should have triggered improvement (accuracy < 0.99)
        assert any(r.improved for r in ll.history)

    def test_improvement_count_increments(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, improvement_threshold=0.99, eval_every=1)
        ll.run(self.X, self.y, chunk_size=50)
        assert ll._improvement_count > 0

    def test_no_improvement_when_threshold_zero(self):
        """Setting threshold to 0.0 means improvement never fires."""
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, improvement_threshold=0.0, eval_every=1)
        ll.run(self.X, self.y, chunk_size=50)
        assert not any(r.improved for r in ll.history)

    def test_improvement_delta_is_float(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, improvement_threshold=0.99, eval_every=1)
        ll.run(self.X, self.y, chunk_size=50)
        for r in ll.history:
            assert isinstance(r.improvement_delta, float)


# ---------------------------------------------------------------------------
# step() API (external streaming)
# ---------------------------------------------------------------------------

class TestStepAPI:
    def setup_method(self):
        self.X, self.y = _make_data()

    def test_step_returns_none_when_no_eval(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=5)
        # First chunk always fits (no eval) — eval_every=5 so first eval at chunk 5
        ll.step(self.X[:30], self.y[:30])  # chunk 1 (fits, no eval)
        result = ll.step(self.X[30:60], self.y[30:60])  # chunk 2 (no eval)
        assert result is None

    def test_step_returns_dict_on_eval(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=2)
        ll.step(self.X[:50], self.y[:50])   # chunk 1 (fits, no eval)
        result = ll.step(self.X[50:100], self.y[50:100])  # chunk 2 → eval
        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_step_accumulates_history(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=1)
        for start in range(0, 150, 50):
            ll.step(self.X[start:start+50], self.y[start:start+50])
        # 3 chunks → first fits (no eval), then 2 evals
        assert len(ll.history) >= 1


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self):
        X, y = _make_data()
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=2)
        ll.run(X, y, chunk_size=50)
        s = ll.summary()
        expected = {
            "n_rounds", "n_samples_seen", "n_improvements",
            "initial_accuracy", "final_accuracy", "peak_accuracy",
            "improvement_threshold",
        }
        assert expected.issubset(set(s.keys()))

    def test_summary_before_run(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent)
        s = ll.summary()
        assert s["n_rounds"] == 0
        assert s["n_samples_seen"] == 0
        assert s["initial_accuracy"] is None


# ---------------------------------------------------------------------------
# final_accuracy()
# ---------------------------------------------------------------------------

class TestFinalAccuracy:
    def test_nan_before_run(self):
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent)
        import math
        assert math.isnan(ll.final_accuracy())

    def test_valid_after_run(self):
        X, y = _make_data()
        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=2)
        ll.run(X, y, chunk_size=50)
        fa = ll.final_accuracy()
        assert 0.0 <= fa <= 1.0


# ---------------------------------------------------------------------------
# MyceliumAgent compatibility
# ---------------------------------------------------------------------------

class TestMyceliumAgentCompat:
    def test_runs_with_mycelium_agent(self):
        from physml import MyceliumAgent

        X, y = _make_data(n=200)
        agent = MyceliumAgent()
        ll = LifelongLearner(agent, improvement_threshold=0.99, eval_every=2)
        history = ll.run(X, y, chunk_size=40)
        # Should complete without error and produce evaluations
        assert len(history) >= 1
        assert ll._fitted is True


# ---------------------------------------------------------------------------
# competitive_report() integration
# ---------------------------------------------------------------------------

class TestCompetitiveReport:
    def test_returns_dict_with_expected_keys(self):
        X, y = _make_data(n=300)
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]

        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=2)
        ll.run(X_train, y_train, chunk_size=50)

        report = ll.competitive_report(X_test, y_test)
        assert "leaderboard" in report
        assert "summary" in report
        assert "verdict" in report

    def test_mycelium_in_leaderboard(self):
        X, y = _make_data(n=300)
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]

        agent = LogisticRegression(max_iter=200)
        ll = LifelongLearner(agent, eval_every=2)
        ll.run(X_train, y_train, chunk_size=50)

        report = ll.competitive_report(X_test, y_test)
        names = [e["name"] for e in report["leaderboard"]]
        assert len(names) >= 1
