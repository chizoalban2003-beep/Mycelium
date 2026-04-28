"""Tests for product completion: ActionDispatcher, VoiceInterface,
ExperimentRunner.analyze_with_llm, and MyceliumCompanion LLM wiring.

Run with::

    python3 -m pytest tests/test_product_completion.py -v --timeout=30
"""

from __future__ import annotations


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


# ===========================================================================
# TestUserMemory
# ===========================================================================


class TestUserMemory:
    def test_import(self):
        from physml.llm.memory_store import UserMemory
        assert UserMemory is not None

    def test_import_from_llm_package(self):
        from physml.llm import UserMemory
        assert UserMemory is not None

    def test_import_from_physml(self):
        from physml import UserMemory
        assert UserMemory is not None

    def test_instantiate(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        assert mem is not None

    def test_remember_and_recall(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        mem.remember("name", "Alex")
        assert mem.recall("name") == "Alex"

    def test_forget(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        mem.remember("name", "Alex")
        mem.forget("name")
        assert mem.recall("name") is None

    def test_recall_missing_returns_none(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        assert mem.recall("nonexistent_key") is None

    def test_inject_into_prompt_returns_str(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        mem.remember("name", "Bob")
        result = mem.inject_into_prompt()
        assert isinstance(result, str)
        assert "Bob" in result

    def test_inject_into_prompt_empty_when_no_facts(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        assert mem.inject_into_prompt() == ""

    def test_summary_keys(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        mem.remember("name", "Alice")
        mem.remember("lang", "Python")
        s = mem.summary()
        assert isinstance(s, dict)
        assert "name" in s
        assert "lang" in s

    def test_persistence(self, tmp_path):
        """Facts survive a reload from disk."""
        from physml.llm.memory_store import UserMemory
        p = str(tmp_path / "mem.json")
        mem1 = UserMemory(path=p)
        mem1.remember("city", "Lagos")
        # Load fresh instance from same path
        mem2 = UserMemory(path=p)
        assert mem2.recall("city") == "Lagos"

    def test_repr(self, tmp_path):
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        mem.remember("x", "1")
        assert "UserMemory" in repr(mem)
        assert "facts=1" in repr(mem)


# ===========================================================================
# TestHealthCheck
# ===========================================================================


class TestHealthCheck:
    def test_import(self):
        from physml.health import check
        assert callable(check)

    def test_import_from_physml(self):
        from physml import health_check
        assert callable(health_check)

    def test_check_returns_dict(self):
        from physml.health import check
        result = check()
        assert isinstance(result, dict)

    def test_check_has_expected_keys(self):
        from physml.health import check
        result = check()
        expected = {"anthropic", "scipy", "pandas", "speech_recognition", "pyttsx3", "version"}
        assert expected.issubset(result.keys())

    def test_version_present(self):
        from physml.health import check
        result = check()
        assert isinstance(result["version"], str)
        assert len(result["version"]) > 0

    def test_scipy_key_is_bool(self):
        from physml.health import check
        result = check()
        assert isinstance(result["scipy"], bool)

    def test_all_dependency_values_are_bool(self):
        from physml.health import check
        result = check()
        for key in ("anthropic", "scipy", "pandas", "speech_recognition", "pyttsx3"):
            assert isinstance(result[key], bool), f"{key} should be bool"


# ===========================================================================
# TestDispatcherRemember
# ===========================================================================


class TestDispatcherRemember:
    def test_dispatch_remember_stores_fact(self, tmp_path):
        from physml.llm import ActionDispatcher, PromptAction
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        d = ActionDispatcher(user_memory=mem)
        action = PromptAction(
            intent="remember",
            payload={"kv": {"name": "Alex"}},
            confidence=0.95,
            raw_text="remember that name=Alex",
        )
        reply = d.dispatch(action)
        assert isinstance(reply, str)
        assert mem.recall("name") == "Alex"

    def test_dispatch_remember_reply_confirms(self, tmp_path):
        from physml.llm import ActionDispatcher, PromptAction
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        d = ActionDispatcher(user_memory=mem)
        action = PromptAction(
            intent="remember",
            payload={"kv": {"lang": "Python"}},
            confidence=0.9,
            raw_text="remember that lang=Python",
        )
        reply = d.dispatch(action)
        # Reply should confirm storage
        assert "lang" in reply.lower() or "remember" in reply.lower() or "noted" in reply.lower()

    def test_dispatch_remember_raw_text_parsing(self, tmp_path):
        from physml.llm import ActionDispatcher, PromptAction
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        d = ActionDispatcher(user_memory=mem)
        action = PromptAction(
            intent="remember",
            payload={},
            confidence=0.85,
            raw_text="my name is Charlie",
        )
        reply = d.dispatch(action)
        # Should have stored the name
        assert isinstance(reply, str)
        assert mem.recall("name") == "Charlie"

    def test_dispatch_memory_shows_remembered_facts(self, tmp_path):
        from physml.llm import ActionDispatcher, PromptAction
        from physml.llm.memory_store import UserMemory
        mem = UserMemory(path=str(tmp_path / "mem.json"))
        mem.remember("name", "Dana")
        d = ActionDispatcher(user_memory=mem)
        action = PromptAction(
            intent="memory",
            payload={},
            confidence=1.0,
            raw_text="what do you remember about me",
        )
        reply = d.dispatch(action)
        assert "Dana" in reply


# ===========================================================================
# TestVersionAndStatus — CLI commands
# ===========================================================================


class TestVersionAndStatus:
    def test_version_command_prints_version(self, capsys):
        from physml.cli import _cmd_version
        import argparse
        _cmd_version(argparse.Namespace())
        captured = capsys.readouterr()
        assert "physml" in captured.out
        assert "1.0" in captured.out

    def test_status_command_prints_table(self, capsys):
        from physml.cli import _cmd_status
        import argparse
        _cmd_status(argparse.Namespace())
        captured = capsys.readouterr()
        assert "physml" in captured.out
        assert "scipy" in captured.out.lower() or "pandas" in captured.out.lower()


# ===========================================================================
# TestTranscribeText — VoiceInterface.transcribe_text
# ===========================================================================


class TestTranscribeText:
    def test_transcribe_text_exists(self):
        from physml.voice import VoiceInterface
        assert hasattr(VoiceInterface, "transcribe_text")

    def test_transcribe_text_returns_str(self):
        from physml.voice import VoiceInterface
        v = VoiceInterface(tts=False)
        result = v.transcribe_text("help")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transcribe_text_same_as_run_once(self):
        from physml.voice import VoiceInterface
        v = VoiceInterface(tts=False)
        r1 = v.run_once("help")
        r2 = v.transcribe_text("help")
        assert r1 == r2


# ===========================================================================
# TestGoalEngineWiring — ActionDispatcher goal_engine parameter
# ===========================================================================


class TestGoalEngineWiring:
    def test_dispatcher_accepts_goal_engine_param(self):
        from physml.llm import ActionDispatcher
        d = ActionDispatcher(goal_engine=None)
        assert d.goal_engine is None

    def test_dispatcher_show_goals_no_engine(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher(goal_engine=None)
        action = PromptAction(intent="show_goals", payload={}, confidence=1.0, raw_text="show goals")
        reply = d.dispatch(action)
        assert isinstance(reply, str)

    def test_dispatcher_add_goal_no_engine(self):
        from physml.llm import ActionDispatcher, PromptAction
        d = ActionDispatcher(goal_engine=None)
        action = PromptAction(
            intent="add_goal",
            payload={"goal_description": "Summarise sales.csv"},
            confidence=1.0,
            raw_text="add goal: summarise sales.csv",
        )
        reply = d.dispatch(action)
        assert "Summarise sales.csv" in reply or "summarise" in reply.lower() or "noted" in reply.lower()

    def test_dispatcher_add_goal_with_mock_engine(self):
        from physml.llm import ActionDispatcher, PromptAction

        class _FakeGoalEngine:
            def add_goal(self, desc, **kwargs):
                return "abcd1234-fake-id"

        d = ActionDispatcher(goal_engine=_FakeGoalEngine())
        action = PromptAction(
            intent="add_goal",
            payload={"goal_description": "Run nightly report"},
            confidence=1.0,
            raw_text="add goal: run nightly report",
        )
        reply = d.dispatch(action)
        assert "abcd1234" in reply

    def test_dispatcher_show_goals_with_mock_engine(self):
        from physml.llm import ActionDispatcher, PromptAction
        from dataclasses import dataclass

        @dataclass
        class _FakeGoalRecord:
            id: str
            description: str
            status: str

        class _FakeGoalEngine:
            def goals(self, status=None):
                return [_FakeGoalRecord(id="aaa-bbb-ccc", description="Do X", status="pending")]

        d = ActionDispatcher(goal_engine=_FakeGoalEngine())
        action = PromptAction(intent="show_goals", payload={}, confidence=1.0, raw_text="show goals")
        reply = d.dispatch(action)
        assert "Do X" in reply
        assert "pending" in reply


# ===========================================================================
# TestWhisperFlag — VoiceInterface whisper_available property
# ===========================================================================


class TestWhisperFlag:
    def test_whisper_available_is_bool(self):
        from physml.voice import VoiceInterface
        v = VoiceInterface(tts=False)
        assert isinstance(v.whisper_available, bool)

    def test_available_true_when_sr_or_whisper(self):
        from physml.voice import VoiceInterface
        v = VoiceInterface(tts=False)
        # available == sr_available OR whisper_available
        assert v.available == (v._sr_available or v._whisper_available)

    def test_check_whisper_method_exists(self):
        from physml.voice import VoiceInterface
        v = VoiceInterface(tts=False)
        assert hasattr(v, "_check_whisper")

    def test_listen_sr_method_exists(self):
        from physml.voice import VoiceInterface
        v = VoiceInterface(tts=False)
        assert hasattr(v, "_listen_sr")

    def test_listen_whisper_method_exists(self):
        from physml.voice import VoiceInterface
        v = VoiceInterface(tts=False)
        assert hasattr(v, "_listen_whisper")


# ===========================================================================
# TestHealthCheckExtended — whisper + sounddevice keys
# ===========================================================================


class TestHealthCheckExtended:
    def test_health_has_whisper_key(self):
        from physml.health import check
        result = check()
        assert "whisper" in result

    def test_health_has_sounddevice_key(self):
        from physml.health import check
        result = check()
        assert "sounddevice" in result

    def test_whisper_value_is_bool(self):
        from physml.health import check
        result = check()
        assert isinstance(result["whisper"], bool)

    def test_sounddevice_value_is_bool(self):
        from physml.health import check
        result = check()
        assert isinstance(result["sounddevice"], bool)


# ===========================================================================
# TestCLIIntegration — end-to-end subprocess CLI tests
# ===========================================================================


class TestCLIIntegration:
    @pytest.mark.slow
    def test_help_flag(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "physml", "--help"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        assert "physml" in result.stdout.lower() or "usage" in result.stdout.lower()

    @pytest.mark.slow
    def test_version_subcommand(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "physml", "version"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        assert "physml" in result.stdout.lower()

    @pytest.mark.slow
    def test_status_subcommand(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "physml", "status"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        output = result.stdout.lower()
        assert "scipy" in output or "pandas" in output or "physml" in output
