"""Tests for product completion: ActionDispatcher, VoiceInterface,
ExperimentRunner.analyze_with_llm, and MyceliumCompanion LLM wiring.

Run with::

    python3 -m pytest tests/test_product_completion.py -v --timeout=30
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Pre-warm the physml/scipy import chain at collection time (not inside tests)
# so per-test timeouts don't fire on the first slow scipy load.
import physml  # noqa: F401


# ===========================================================================
# TestActionDispatcher
# ===========================================================================


class TestActionDispatcher:
    def test_import(self):
        from physml.llm.action_dispatcher import ActionDispatcher
        assert ActionDispatcher is not None

    def test_import_from_llm_package(self):
        from physml.llm import ActionDispatcher
        assert ActionDispatcher is not None

    def test_import_from_physml(self):
        from physml import ActionDispatcher
        assert ActionDispatcher is not None

    def test_instantiate_no_args(self):
        from physml.llm import ActionDispatcher
        d = ActionDispatcher()
        assert d is not None

    def test_repr(self):
        from physml.llm import ActionDispatcher
        d = ActionDispatcher()
        assert "ActionDispatcher" in repr(d)

    def test_dispatch_help(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher()
        action = PromptAction(intent="help", confidence=1.0, raw_text="help")
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert "train" in reply.lower() or "predict" in reply.lower() or "help" in reply.lower()

    def test_dispatch_report_no_agent(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher(agent=None)
        action = PromptAction(intent="report", confidence=1.0, raw_text="show report")
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert "train" in reply.lower() or "no agent" in reply.lower()

    def test_dispatch_predict_no_agent(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher(agent=None)
        action = PromptAction(intent="predict", confidence=1.0, raw_text="predict 1.5 2.3")
        reply = d.dispatch(action)
        assert "train" in reply.lower() or "no agent" in reply.lower()

    def test_dispatch_save_no_agent(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher(agent=None)
        action = PromptAction(intent="save", confidence=1.0, raw_text="save")
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert "no agent" in reply.lower() or "train" in reply.lower()

    def test_dispatch_train_no_path(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher()
        action = PromptAction(intent="train", confidence=0.9, payload={}, raw_text="train a model")
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        # Should ask for a CSV path
        assert "csv" in reply.lower() or "file" in reply.lower() or "path" in reply.lower()

    def test_dispatch_train_file_not_found(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher()
        action = PromptAction(
            intent="train",
            confidence=0.9,
            payload={"path": "/nonexistent/file.csv"},
            raw_text="train on /nonexistent/file.csv",
        )
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert "not found" in reply.lower() or "error" in reply.lower()

    @pytest.mark.slow
    def test_dispatch_train_and_predict(self, tmp_path):
        """Full train → predict cycle through the dispatcher."""
        import pandas as pd
        from physml.llm import ActionDispatcher, PromptAction

        # Create a small CSV
        csv_path = tmp_path / "data.csv"
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 3))
        y = (X[:, 0] > 0).astype(int)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["target"] = y
        df.to_csv(csv_path, index=False)

        d = ActionDispatcher(agent_path=str(tmp_path / "agent.pkl"))

        # Train
        train_action = PromptAction(
            intent="train",
            confidence=0.95,
            payload={"path": str(csv_path)},
            raw_text=f"train on {csv_path}",
        )
        reply = d.dispatch(train_action)
        assert "train" in reply.lower() or "fitted" in reply.lower() or "samples" in reply.lower()
        assert d.agent is not None

        # Predict
        predict_action = PromptAction(
            intent="predict",
            confidence=0.9,
            payload={"numbers": [0.5, -0.3, 1.2]},
            raw_text="predict 0.5 -0.3 1.2",
        )
        reply2 = d.dispatch(predict_action)
        assert "prediction" in reply2.lower() or "confidence" in reply2.lower()

    @pytest.mark.slow
    def test_dispatch_report_with_agent(self, tmp_path):
        """Report dispatch with a fitted agent."""
        import pandas as pd
        from physml.llm import ActionDispatcher, PromptAction
        from physml.mycelium_agent import MyceliumAgent

        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 2))
        y = (X[:, 0] > 0).astype(int)
        agent = MyceliumAgent()
        agent.fit(X, y)

        d = ActionDispatcher(agent=agent)
        action = PromptAction(intent="report", confidence=1.0, raw_text="show report")
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_dispatch_memory_no_store(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher(store=None)
        action = PromptAction(intent="memory", confidence=1.0, raw_text="what do you remember")
        reply = d.dispatch(action)
        assert isinstance(reply, str)

    def test_dispatch_memory_with_store(self, tmp_path):
        from physml.llm import ActionDispatcher, PromptAction
        from physml.conversation_store import ConversationStore

        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "hello world")
        store.add("assistant", "hi there")

        d = ActionDispatcher(store=store)
        action = PromptAction(intent="memory", confidence=1.0, raw_text="show memory")
        reply = d.dispatch(action)
        assert "2" in reply or "turn" in reply.lower() or "history" in reply.lower()

    def test_dispatch_show_goals(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher()
        action = PromptAction(intent="show_goals", confidence=1.0, raw_text="show goals")
        reply = d.dispatch(action)
        assert isinstance(reply, str)

    def test_dispatch_add_goal(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher()
        action = PromptAction(
            intent="add_goal",
            confidence=0.9,
            payload={"goal_description": "summarise my sales data"},
            raw_text="add goal: summarise my sales data",
        )
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert "goal" in reply.lower() or "noted" in reply.lower()

    def test_dispatch_unknown_no_client(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher(client=None)
        action = PromptAction(
            intent="unknown",
            confidence=0.1,
            raw_text="xyzzy frobnicate the wombat",
        )
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert len(reply) > 0


# ===========================================================================
# TestVoiceInterface
# ===========================================================================


class TestVoiceInterface:
    def test_import(self):
        from physml.voice import VoiceInterface
        assert VoiceInterface is not None

    def test_import_from_physml(self):
        from physml import VoiceInterface
        assert VoiceInterface is not None

    def test_instantiate_default(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface()
        assert vi is not None

    def test_instantiate_tts_disabled(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface(tts=False)
        assert vi._tts_engine is None

    def test_available_property_type(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface()
        assert isinstance(vi.available, bool)

    def test_available_false_without_sr(self, monkeypatch):
        """Patch speech_recognition to simulate absence."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "speech_recognition":
                raise ImportError("mocked absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from physml.voice import VoiceInterface
        vi = VoiceInterface.__new__(VoiceInterface)
        # Directly test _check_sr under mocked imports
        result = vi._check_sr()
        assert result is False

    def test_speak_fallback_prints(self, capsys):
        from physml.voice import VoiceInterface
        vi = VoiceInterface(tts=False)
        vi.speak("Hello world")
        captured = capsys.readouterr()
        assert "Hello world" in captured.out

    def test_repr(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface(tts=False)
        r = repr(vi)
        assert "VoiceInterface" in r
        assert "available" in r

    def test_run_once(self):
        """run_once processes text and returns a string response."""
        from physml.voice import VoiceInterface
        vi = VoiceInterface(tts=False)
        response = vi.run_once("help")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_run_once_predict(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface(tts=False)
        response = vi.run_once("predict 1.5 2.3 3.1")
        assert isinstance(response, str)

    def test_run_once_train_no_path(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface(tts=False)
        response = vi.run_once("train a model")
        # Should ask for a CSV path
        assert isinstance(response, str)

    def test_language_attribute(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface(language="fr-FR", tts=False)
        assert vi.language == "fr-FR"

    def test_timeout_attribute(self):
        from physml.voice import VoiceInterface
        vi = VoiceInterface(timeout=3.0, tts=False)
        assert vi.timeout == 3.0


# ===========================================================================
# TestExperimentAnalysis
# ===========================================================================


class TestExperimentAnalysis:
    def test_analyze_with_llm_method_exists(self):
        from physml.experiment_runner import ExperimentRunner
        assert hasattr(ExperimentRunner, "analyze_with_llm")

    def test_analyze_with_llm_no_client_returns_empty(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner, BenchmarkSummary

        summary = BenchmarkSummary(
            task="regression",
            n_experiments=1,
            best_score=0.85,
            best_config={"plane": "liquid"},
            mean_score=0.80,
            std_score=0.05,
            total_time_s=1.0,
        )
        runner = ExperimentRunner(results_dir=str(tmp_path))
        result = runner.analyze_with_llm(summary, client=None)
        assert result == ""

    def test_analyze_with_llm_unavailable_client_returns_empty(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner, BenchmarkSummary
        from physml.llm import ClaudeClient

        summary = BenchmarkSummary(
            task="classification",
            n_experiments=3,
            best_score=0.92,
            best_config={"plane": "solid"},
            mean_score=0.88,
            std_score=0.03,
            total_time_s=2.5,
        )
        runner = ExperimentRunner(results_dir=str(tmp_path))
        client = ClaudeClient(api_key="")  # no API key → not available
        result = runner.analyze_with_llm(summary, client=client)
        assert result == ""

    def test_analyze_with_llm_returns_str(self, tmp_path):
        """Even without a real client, the return type must be str."""
        from physml.experiment_runner import ExperimentRunner, BenchmarkSummary
        from physml.llm import ClaudeClient

        summary = BenchmarkSummary(
            task="regression",
            n_experiments=2,
            best_score=0.77,
            best_config={"plane": "gas"},
            mean_score=0.70,
            std_score=0.07,
            total_time_s=0.5,
        )
        runner = ExperimentRunner(results_dir=str(tmp_path))
        client = ClaudeClient(api_key="")
        result = runner.analyze_with_llm(summary, client=client)
        assert isinstance(result, str)


# ===========================================================================
# TestCompanionLLM
# ===========================================================================


class TestCompanionLLM:
    def test_companion_has_llm_attribute(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        # llm attribute is set before start() during __init__
        assert hasattr(c, "llm")

    def test_companion_has_claude_client_property(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "claude_client")

    def test_companion_claude_client_type(self):
        from physml.companion import MyceliumCompanion
        from physml.llm import ClaudeClient
        c = MyceliumCompanion()
        cc = c.claude_client
        # May be None if SDK absent, otherwise ClaudeClient
        assert cc is None or isinstance(cc, ClaudeClient)

    def test_companion_has_chat_llm_method(self):
        from physml.companion import MyceliumCompanion
        assert hasattr(MyceliumCompanion, "chat_llm")

    def test_companion_has_voice_interface_method(self):
        from physml.companion import MyceliumCompanion
        assert hasattr(MyceliumCompanion, "start_voice_interface")

    @pytest.mark.slow
    def test_companion_chat_returns_str(self, tmp_path):
        """companion.chat() route must return a string."""
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        try:
            response = c.chat("help")
            assert isinstance(response, str)
        finally:
            c.stop()

    @pytest.mark.slow
    def test_companion_chat_llm_returns_str(self, tmp_path):
        """companion.chat_llm() must return a string even without API key."""
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        try:
            response = c.chat_llm("show me the help")
            assert isinstance(response, str)
            assert len(response) > 0
        finally:
            c.stop()

    @pytest.mark.slow
    def test_companion_chat_llm_train_intent(self, tmp_path):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        try:
            response = c.chat_llm("train a model on data.csv")
            assert isinstance(response, str)
        finally:
            c.stop()

    def test_voice_interface_method_callable(self):
        """start_voice_interface() should exist as a callable method on the class."""
        from physml.companion import MyceliumCompanion
        assert callable(getattr(MyceliumCompanion, "start_voice_interface", None))
