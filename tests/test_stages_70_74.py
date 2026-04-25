"""Tests for Stages 70–74: HyperTuner, SelfHealer, EvalScheduler, SelfPlay.

Stage 72 (WebSocket) is also verified at the integration level via the
create_app() factory (route presence check) since full async WebSocket tests
require a running ASGI server.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(n=300, n_features=8, random_state=0):
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=4,
        random_state=random_state,
    )
    return X, y


# ---------------------------------------------------------------------------
# Stage 70 — HyperTuner
# ---------------------------------------------------------------------------

class TestTuneResult:
    def test_as_dict_keys(self):
        from physml.hyper_tuner import TuneResult

        r = TuneResult(
            round_idx=0,
            best_params={"C": 1.0},
            best_score=0.85,
            n_candidates=6,
            elapsed_s=0.5,
        )
        d = r.as_dict()
        assert set(d.keys()) == {
            "round", "best_params", "best_score", "n_candidates",
            "elapsed_s", "stored_in_graph",
        }

    def test_as_dict_values(self):
        from physml.hyper_tuner import TuneResult

        r = TuneResult(0, {"C": 0.1}, 0.9, 4, 1.2, True)
        d = r.as_dict()
        assert d["best_score"] == pytest.approx(0.9, abs=1e-4)
        assert d["stored_in_graph"] is True


class TestHyperTuner:
    def setup_method(self):
        self.X, self.y = _data()

    def test_public_export(self):
        from physml import HyperTuner
        from physml.hyper_tuner import HyperTuner as HT

        assert HyperTuner is HT

    def test_tune_returns_tune_result(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent)
        result = tuner.tune(self.X, self.y)
        from physml.hyper_tuner import TuneResult

        assert isinstance(result, TuneResult)

    def test_tune_result_has_params(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent, n_candidates=3)
        result = tuner.tune(self.X, self.y)
        assert isinstance(result.best_params, dict)

    def test_tune_result_score_is_float(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent, n_candidates=3)
        result = tuner.tune(self.X, self.y)
        assert isinstance(result.best_score, float)

    def test_history_grows(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent, n_candidates=3)
        tuner.tune(self.X, self.y)
        tuner.tune(self.X, self.y)
        assert len(tuner.history) == 2

    def test_best_result_is_highest_score(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent, n_candidates=3)
        tuner.tune(self.X, self.y)
        tuner.tune(self.X, self.y)
        best = tuner.best_result()
        assert best is not None
        assert best.best_score == max(r.best_score for r in tuner.history)

    def test_best_result_none_before_tune(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent)
        assert tuner.best_result() is None

    def test_maybe_tune_fires_every_n(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent, tune_every=3, n_candidates=2)
        results = [tuner.maybe_tune(self.X, self.y) for _ in range(6)]
        # Should fire at call 3 and 6
        not_none = [r for r in results if r is not None]
        assert len(not_none) == 2

    def test_maybe_tune_returns_none_otherwise(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent, tune_every=5, n_candidates=2)
        result = tuner.maybe_tune(self.X, self.y)  # call 1 → no fire
        assert result is None

    def test_summary_keys(self):
        from physml.hyper_tuner import HyperTuner

        agent = LogisticRegression(max_iter=200)
        tuner = HyperTuner(agent, n_candidates=3)
        tuner.tune(self.X, self.y)
        s = tuner.summary()
        assert {"n_rounds", "best_score_ever", "latest_best_params"}.issubset(s.keys())

    def test_knowledge_graph_integration(self):
        from physml.hyper_tuner import HyperTuner
        from physml.knowledge_graph import KnowledgeGraph

        agent = LogisticRegression(max_iter=200)
        kg = KnowledgeGraph()
        tuner = HyperTuner(agent, n_candidates=2, knowledge_graph=kg)
        result = tuner.tune(self.X, self.y)
        assert result.stored_in_graph is True
        nodes = kg.nodes_by_type("hyper_tune")
        assert len(nodes) >= 1

    def test_mycelium_agent_compat(self):
        from physml import MyceliumAgent
        from physml.hyper_tuner import HyperTuner

        X, y = _data(n=200)
        agent = MyceliumAgent()
        tuner = HyperTuner(agent, n_candidates=2)
        result = tuner.tune(X, y)
        assert isinstance(result.best_score, float)


# ---------------------------------------------------------------------------
# Stage 71 — SelfHealer
# ---------------------------------------------------------------------------

class TestHealingIncident:
    def test_as_dict_keys(self):
        from physml.self_healer import HealingIncident

        inc = HealingIncident(
            timestamp=1.0,
            trigger="anomaly",
            anomaly_rate=0.4,
            accuracy_before=0.5,
            checkpoint_path="/tmp/ckpt",
            curriculum_reset=True,
        )
        d = inc.as_dict()
        assert set(d.keys()) == {
            "timestamp", "trigger", "anomaly_rate", "accuracy_before",
            "checkpoint_path", "curriculum_reset",
        }

    def test_nan_accuracy_becomes_none(self):
        from physml.self_healer import HealingIncident

        inc = HealingIncident(1.0, "collapse", 0.1, float("nan"), None, False)
        d = inc.as_dict()
        assert d["accuracy_before"] is None


class TestSelfHealer:
    def setup_method(self):
        self.X, self.y = _data()

    def test_public_export(self):
        from physml import SelfHealer
        from physml.self_healer import SelfHealer as SH

        assert SelfHealer is SH

    def _make_healer(self, tmp_dir):
        from physml.self_healer import SelfHealer

        agent = LogisticRegression(max_iter=200)
        agent.fit(self.X[:150], self.y[:150])
        return SelfHealer(agent, os.path.join(tmp_dir, "agent.ckpt"))

    def test_checkpoint_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            path = healer.checkpoint()
            assert path.exists()

    def test_fit_guard(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            healer.fit_guard(self.X[:100])
            assert healer._guard_fitted is True

    def test_protect_no_anomaly(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            healer.fit_guard(self.X[:200])
            healer.checkpoint()
            result = healer.protect(self.X[200:], self.y[200:])
            assert "healed" in result
            assert "anomaly_rate" in result

    def test_protect_triggers_heal_on_collapse(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            healer.checkpoint()
            # Set very high collapse threshold to ensure trigger
            healer.collapse_threshold = 2.0
            result = healer.protect(self.X[150:], self.y[150:])
            assert result["healed"] is True
            assert healer.n_heals == 1

    def test_protect_triggers_heal_on_anomaly(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            healer.fit_guard(self.X[:100])
            healer.checkpoint()
            # Very low anomaly threshold to ensure trigger
            healer.anomaly_threshold = 0.0
            result = healer.protect(self.X[150:])
            assert result["healed"] is True

    def test_rollback_restores_agent(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            healer.checkpoint()
            ok = healer.rollback()
            assert ok is True

    def test_rollback_fails_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            agent = LogisticRegression(max_iter=200)
            from physml.self_healer import SelfHealer
            healer = SelfHealer(agent, os.path.join(d, "missing.ckpt"))
            assert healer.rollback() is False

    def test_incidents_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            healer.checkpoint()
            healer.collapse_threshold = 2.0
            healer.protect(self.X[150:], self.y[150:])
            assert len(healer.incidents) == 1

    def test_summary_keys(self):
        with tempfile.TemporaryDirectory() as d:
            healer = self._make_healer(d)
            s = healer.summary()
            assert {"n_heals", "guard_fitted", "checkpoint_exists", "incidents"}.issubset(
                s.keys()
            )

    def test_curriculum_reset_on_heal(self):
        """If a CurriculumScheduler is attached, its difficulty is reset."""
        with tempfile.TemporaryDirectory() as d:
            from physml.curriculum import CurriculumScheduler
            from physml.self_healer import SelfHealer

            agent = LogisticRegression(max_iter=200)
            agent.fit(self.X[:150], self.y[:150])
            curriculum = CurriculumScheduler()
            curriculum.difficulty = 0.9
            healer = SelfHealer(
                agent,
                os.path.join(d, "agent.ckpt"),
                curriculum=curriculum,
                reset_difficulty=0.1,
            )
            healer.checkpoint()
            healer.collapse_threshold = 2.0
            healer.protect(self.X[150:], self.y[150:])
            assert curriculum.difficulty == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Stage 72 — WebSocket endpoint (structural check)
# ---------------------------------------------------------------------------

class TestWebSocketEndpoint:
    def test_ws_route_registered(self):
        """Check that /ws/predict is registered in the FastAPI app."""
        try:
            from physml.server import create_app

            app = create_app()
            routes = [getattr(r, "path", None) for r in app.routes]
            assert "/ws/predict" in routes
        except ImportError:
            pytest.skip("fastapi not available")

    def test_create_app_returns_app(self):
        try:
            from physml.server import create_app

            app = create_app()
            assert app is not None
        except ImportError:
            pytest.skip("fastapi not available")


# ---------------------------------------------------------------------------
# Stage 73 — EvalScheduler
# ---------------------------------------------------------------------------

class TestScheduledReport:
    def test_as_dict_keys(self):
        from physml.eval_scheduler import ScheduledReport

        r = ScheduledReport(
            report_idx=0,
            timestamp=1.0,
            mycelium_rank=1,
            mycelium_accuracy=0.9,
            n_competitors=4,
            alert=False,
            winner="MyceliumAgent",
        )
        d = r.as_dict()
        assert set(d.keys()) == {
            "report_idx", "timestamp", "mycelium_rank", "mycelium_accuracy",
            "n_competitors", "alert", "winner", "stored_in_graph",
        }


class TestEvalScheduler:
    def setup_method(self):
        self.X, self.y = _data(n=300)
        self.X_train = self.X[:200]
        self.y_train = self.y[:200]
        self.X_test = self.X[200:]
        self.y_test = self.y[200:]

    def _agent(self):
        """Use LogisticRegression — CompetitiveReport/Arena requires predict()."""
        agent = LogisticRegression(max_iter=200)
        agent.fit(self.X_train, self.y_train)
        return agent

    def test_public_export(self):
        from physml import EvalScheduler
        from physml.eval_scheduler import EvalScheduler as ES

        assert EvalScheduler is ES

    def test_run_returns_scheduled_report(self):
        from physml.eval_scheduler import EvalScheduler, ScheduledReport

        scheduler = EvalScheduler(self._agent())
        report = scheduler.run(self.X_test, self.y_test)
        assert isinstance(report, ScheduledReport)

    def test_report_has_rank(self):
        from physml.eval_scheduler import EvalScheduler

        scheduler = EvalScheduler(self._agent())
        report = scheduler.run(self.X_test, self.y_test)
        assert isinstance(report.mycelium_rank, int)
        assert report.mycelium_rank >= 1

    def test_alert_flag_when_rank_high(self):
        from physml.eval_scheduler import EvalScheduler

        # alert when rank > 0 (always triggers)
        scheduler = EvalScheduler(self._agent(), alert_rank_threshold=0)
        report = scheduler.run(self.X_test, self.y_test)
        assert report.alert is True

    def test_no_alert_when_rank_within_threshold(self):
        from physml.eval_scheduler import EvalScheduler

        # alert only when rank > 100 (impossibly strict)
        scheduler = EvalScheduler(self._agent(), alert_rank_threshold=100)
        report = scheduler.run(self.X_test, self.y_test)
        assert report.alert is False

    def test_history_grows(self):
        from physml.eval_scheduler import EvalScheduler

        scheduler = EvalScheduler(self._agent())
        scheduler.run(self.X_test, self.y_test)
        scheduler.run(self.X_test, self.y_test)
        assert len(scheduler.history) == 2

    def test_maybe_run_fires_every_n(self):
        from physml.eval_scheduler import EvalScheduler

        scheduler = EvalScheduler(self._agent(), eval_every=3)
        results = [scheduler.maybe_run(self.X_test, self.y_test) for _ in range(6)]
        not_none = [r for r in results if r is not None]
        assert len(not_none) == 2

    def test_maybe_run_returns_none_otherwise(self):
        from physml.eval_scheduler import EvalScheduler

        scheduler = EvalScheduler(self._agent(), eval_every=5)
        result = scheduler.maybe_run(self.X_test, self.y_test)
        assert result is None

    def test_knowledge_graph_integration(self):
        from physml.eval_scheduler import EvalScheduler
        from physml.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        scheduler = EvalScheduler(self._agent(), knowledge_graph=kg)
        report = scheduler.run(self.X_test, self.y_test)
        assert report.stored_in_graph is True
        nodes = kg.nodes_by_type("eval_report")
        assert len(nodes) >= 1

    def test_alert_property(self):
        from physml.eval_scheduler import EvalScheduler

        scheduler = EvalScheduler(self._agent(), alert_rank_threshold=0)
        assert scheduler.alert is False  # before any run
        scheduler.run(self.X_test, self.y_test)
        assert scheduler.alert is True

    def test_summary_keys(self):
        from physml.eval_scheduler import EvalScheduler

        scheduler = EvalScheduler(self._agent())
        scheduler.run(self.X_test, self.y_test)
        s = scheduler.summary()
        assert {"n_reports", "n_alerts", "best_rank_ever", "latest_rank"}.issubset(
            s.keys()
        )

    def test_synthetic_data_fallback(self):
        """EvalScheduler runs even without providing test data."""
        from physml.eval_scheduler import EvalScheduler

        scheduler = EvalScheduler(self._agent(), n_samples=200, n_features=8)
        report = scheduler.run()  # no X_test/y_test provided
        assert report.mycelium_rank >= 1


# ---------------------------------------------------------------------------
# Stage 74 — SelfPlay
# ---------------------------------------------------------------------------

class TestPlayRound:
    def test_as_dict_keys(self):
        from physml.self_play import PlayRound

        r = PlayRound(
            round_idx=0,
            winner="agent_a",
            agent_a_accuracy=0.8,
            agent_b_accuracy=0.7,
            federated=True,
            elapsed_s=0.5,
        )
        d = r.as_dict()
        assert set(d.keys()) == {
            "round", "winner", "agent_a_accuracy", "agent_b_accuracy",
            "federated", "elapsed_s",
        }


class TestSelfPlay:
    def setup_method(self):
        self.X, self.y = _data(n=300)

    def test_public_export(self):
        from physml import SelfPlay
        from physml.self_play import SelfPlay as SP

        assert SelfPlay is SP

    @pytest.mark.slow
    def test_run_returns_list_of_play_rounds(self):
        from physml import MyceliumAgent
        from physml.self_play import PlayRound, SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
        history = sp.run(self.X, self.y, n_rounds=3)
        assert len(history) == 3
        assert all(isinstance(r, PlayRound) for r in history)

    @pytest.mark.slow
    def test_winner_is_valid(self):
        from physml import MyceliumAgent
        from physml.self_play import SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
        history = sp.run(self.X, self.y, n_rounds=2)
        for r in history:
            assert r.winner in ("agent_a", "agent_b")

    @pytest.mark.slow
    def test_federated_fires(self):
        from physml import MyceliumAgent
        from physml.self_play import SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent(), federate_every=2)
        history = sp.run(self.X, self.y, n_rounds=4)
        federated_rounds = [r for r in history if r.federated]
        assert len(federated_rounds) >= 1

    @pytest.mark.slow
    def test_leaderboard_structure(self):
        from physml import MyceliumAgent
        from physml.self_play import SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
        sp.run(self.X, self.y, n_rounds=2)
        lb = sp.leaderboard()
        assert "agent_a" in lb
        assert "agent_b" in lb
        assert "n_rounds" in lb

    @pytest.mark.slow
    def test_best_agent_returns_agent(self):
        from physml import MyceliumAgent
        from physml.self_play import SelfPlay

        a = MyceliumAgent()
        b = MyceliumAgent()
        sp = SelfPlay(a, b)
        sp.run(self.X, self.y, n_rounds=3)
        best = sp.best_agent()
        assert best in (a, b)

    @pytest.mark.slow
    def test_compete_single_round(self):
        from physml import MyceliumAgent
        from physml.self_play import PlayRound, SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
        X_train, y_train = self.X[:200], self.y[:200]
        X_test, y_test = self.X[200:], self.y[200:]
        result = sp.compete(X_train, y_train, X_test, y_test)
        assert isinstance(result, PlayRound)

    @pytest.mark.slow
    def test_wins_accumulate(self):
        from physml import MyceliumAgent
        from physml.self_play import SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
        sp.run(self.X, self.y, n_rounds=4)
        total_wins = sp._wins["agent_a"] + sp._wins["agent_b"]
        assert total_wins == 4

    @pytest.mark.slow
    def test_history_matches_property(self):
        from physml import MyceliumAgent
        from physml.self_play import SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
        returned = sp.run(self.X, self.y, n_rounds=3)
        assert returned == sp.history

    @pytest.mark.slow
    def test_federate_method(self):
        from physml import MyceliumAgent
        from physml.self_play import SelfPlay

        sp = SelfPlay(MyceliumAgent(), MyceliumAgent())
        # Train agents first so FedAvg has something to exchange
        sp.agent_a.fit(self.X[:200], self.y[:200])
        sp.agent_b.fit(self.X[:200], self.y[:200])
        result = sp.federate(self.X[:200], self.y[:200])
        # Should return bool (True on success or False on soft failure)
        assert isinstance(result, bool)

    def test_sklearn_compat(self):
        """SelfPlay works with plain sklearn estimators."""
        from physml.self_play import SelfPlay

        sp = SelfPlay(
            LogisticRegression(max_iter=200),
            LogisticRegression(max_iter=200),
        )
        history = sp.run(self.X, self.y, n_rounds=2)
        assert len(history) == 2
