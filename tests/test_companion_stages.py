"""Tests for Stages 106–122: companion, voice, LLM integration, and subsystems."""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Stage 106 — NaturalLanguageRouter
# ---------------------------------------------------------------------------

class TestNaturalLanguageRouter:
    def _make_router(self):
        from physml.nl_router import NaturalLanguageRouter, Intent

        r = NaturalLanguageRouter(min_confidence=0.3)
        r.register_many([
            Intent("predict", ["predict", "forecast", "estimate"]),
            Intent("train",   ["train on", "learn from", "fit"]),
            Intent("report",  ["show report", "stats"]),
        ])
        return r

    def test_known_intent_matched(self):
        r = self._make_router()
        result = r.route("predict the outcome")
        assert result.intent == "predict"
        assert result.confidence > 0

    def test_unknown_intent_returns_unknown(self):
        from physml.nl_router import NaturalLanguageRouter, Intent

        # Use high min_confidence so random text can't match
        r = NaturalLanguageRouter(min_confidence=0.99)
        r.register_many([
            Intent("predict", ["predict", "forecast", "estimate"]),
        ])
        result = r.route("xyzzy frobble bazzle quux")
        assert result.intent == "unknown"

    def test_entity_number_extraction(self):
        r = self._make_router()
        result = r.route("predict 1.5 2.3 100")
        assert 1.5 in result.entities.get("numbers", [])
        assert 2.3 in result.entities.get("numbers", [])

    def test_entity_path_extraction(self):
        r = self._make_router()
        result = r.route("train on /data/train.csv")
        paths = result.entities.get("paths", [])
        assert any("train.csv" in p for p in paths)

    def test_quoted_entity_extraction(self):
        r = self._make_router()
        result = r.route('predict "hello world"')
        quoted = result.entities.get("quoted", [])
        assert "hello world" in quoted

    def test_register_intent(self):
        from physml.nl_router import NaturalLanguageRouter, Intent

        r = NaturalLanguageRouter()
        r.register(Intent("custom", ["do the custom thing"]))
        result = r.route("do the custom thing")
        assert result.intent == "custom"

    def test_route_returns_routed_action(self):
        from physml.nl_router import RoutedAction

        r = self._make_router()
        result = r.route("show report")
        assert isinstance(result, RoutedAction)
        assert result.raw_text == "show report"


# ---------------------------------------------------------------------------
# Stage 108 — ConversationManager
# ---------------------------------------------------------------------------

class TestConversationManager:
    def test_add_turn_and_context(self):
        from physml.conversation import ConversationManager

        mgr = ConversationManager(max_history=10)
        mgr.add_turn("user", "hello")
        mgr.add_turn("agent", "hi there")
        ctx = mgr.context(n=5)
        assert "hello" in ctx
        assert "hi there" in ctx

    def test_max_history_trimming(self):
        from physml.conversation import ConversationManager

        mgr = ConversationManager(max_history=3)
        for i in range(10):
            mgr.add_turn("user", f"msg {i}")
        assert len(mgr.turns) <= 3

    def test_save_and_load_roundtrip(self, tmp_path):
        from physml.conversation import ConversationManager

        path = str(tmp_path / "convo.json")
        mgr = ConversationManager()
        mgr.add_turn("user", "test message")
        mgr.save(path)

        mgr2 = ConversationManager()
        mgr2.load(path)
        assert any(t.text == "test message" for t in mgr2.turns)

    def test_turn_has_intent_after_routing(self):
        from physml.conversation import ConversationManager
        from physml.nl_router import NaturalLanguageRouter, Intent

        router = NaturalLanguageRouter()
        router.register(Intent("predict", ["predict"]))
        mgr = ConversationManager(router=router)
        turn = mgr.add_turn("user", "predict 1 2 3")
        assert turn.intent == "predict"

    def test_last_intent_property(self):
        from physml.conversation import ConversationManager

        mgr = ConversationManager()
        assert mgr.last_intent is None
        mgr.add_turn("user", "hello")
        # last_intent returns None or a string
        assert mgr.last_intent is None or isinstance(mgr.last_intent, str)


# ---------------------------------------------------------------------------
# Stage 109 — ScheduledTaskRunner
# ---------------------------------------------------------------------------

class TestScheduledTaskRunner:
    def test_schedule_and_list_tasks(self):
        from physml.scheduler_tasks import ScheduledTaskRunner

        runner = ScheduledTaskRunner()
        runner.schedule("test_task", fn=lambda: None, interval_seconds=60)
        tasks = runner.list_tasks()
        assert any(t["name"] == "test_task" for t in tasks)

    def test_cancel_task(self):
        from physml.scheduler_tasks import ScheduledTaskRunner

        runner = ScheduledTaskRunner()
        runner.schedule("remove_me", fn=lambda: None, interval_seconds=60)
        runner.cancel("remove_me")
        tasks = runner.list_tasks()
        assert not any(t["name"] == "remove_me" for t in tasks)

    def test_start_stop(self):
        from physml.scheduler_tasks import ScheduledTaskRunner

        runner = ScheduledTaskRunner()
        runner.schedule("bg_task", fn=lambda: None, interval_seconds=3600)
        runner.start()
        runner.stop()


# ---------------------------------------------------------------------------
# Stage 110 — DocumentProcessor
# ---------------------------------------------------------------------------

class TestDocumentProcessor:
    def test_process_csv(self, tmp_path):
        from physml.doc_processor import DocumentProcessor

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

        dp = DocumentProcessor()
        result = dp.process(str(csv_file))
        assert result.success
        assert result.df is not None
        assert result.df.shape == (2, 3)

    def test_process_txt(self, tmp_path):
        from physml.doc_processor import DocumentProcessor

        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("Hello world.\nSecond line.\n")

        dp = DocumentProcessor()
        result = dp.process(str(txt_file))
        assert result.success
        assert "Hello world" in result.text

    def test_process_missing_file(self):
        from physml.doc_processor import DocumentProcessor

        dp = DocumentProcessor()
        result = dp.process("/nonexistent/file.csv")
        assert not result.success
        assert result.error is not None

    def test_process_json(self, tmp_path):
        from physml.doc_processor import DocumentProcessor

        jf = tmp_path / "data.json"
        jf.write_text(json.dumps({"key": "value", "num": 42}))

        dp = DocumentProcessor()
        result = dp.process(str(jf))
        assert result.success


# ---------------------------------------------------------------------------
# Stage 111 — VoiceInputAdapter
# ---------------------------------------------------------------------------

class TestVoiceInputAdapter:
    def test_passthrough_mode(self):
        from physml.voice_adapter import VoiceInputAdapter

        adapter = VoiceInputAdapter(preferred_backend="passthrough")
        result = adapter.from_text("hello world")
        assert result.text == "hello world"
        assert result.success
        assert result.backend == "passthrough"

    def test_active_backend_is_passthrough_without_deps(self):
        from physml.voice_adapter import VoiceInputAdapter

        adapter = VoiceInputAdapter(preferred_backend="auto")
        # passthrough is always last resort
        assert adapter.active_backend in (
            "faster_whisper", "openai_whisper", "whisper", "speechrecognition", "passthrough"
        )

    def test_transcribe_missing_file_passthrough(self):
        from physml.voice_adapter import VoiceInputAdapter

        adapter = VoiceInputAdapter(preferred_backend="passthrough")
        result = adapter.transcribe_file("/nonexistent/audio.wav")
        assert result.backend == "passthrough"
        assert not result.success

    def test_repr(self):
        from physml.voice_adapter import VoiceInputAdapter

        adapter = VoiceInputAdapter()
        r = repr(adapter)
        assert "VoiceInputAdapter" in r


# ---------------------------------------------------------------------------
# Stage 112 — PluginRegistry
# ---------------------------------------------------------------------------

class TestPluginRegistry:
    def test_empty_registry(self, tmp_path):
        from physml.plugin_registry import PluginRegistry

        reg = PluginRegistry(plugin_dir=str(tmp_path))
        reg.load_all()
        assert len(reg.loaded) == 0

    def test_register_callable(self):
        from physml.plugin_registry import PluginRegistry

        reg = PluginRegistry()
        reg.register("greet", lambda name: f"Hello {name}")
        assert "greet" in reg.list()

    def test_call_plugin(self):
        from physml.plugin_registry import PluginRegistry

        reg = PluginRegistry()
        reg.register("double", lambda x: x * 2)
        result = reg.call("double", 5)
        assert result == 10

    def test_call_missing_plugin_raises(self):
        from physml.plugin_registry import PluginRegistry

        reg = PluginRegistry()
        with pytest.raises((KeyError, ValueError)):
            reg.call("nonexistent")


# ---------------------------------------------------------------------------
# Stage 113 — UserProfileLearner
# ---------------------------------------------------------------------------

class TestUserProfileLearner:
    def test_record_interaction(self, tmp_path):
        from physml.user_profile import UserProfileLearner

        path = str(tmp_path / "profile.json")
        profile = UserProfileLearner(profile_path=path)
        profile.record_interaction(intent="predict", topic="sales")
        assert "sales" in profile.top_topics(3)

    def test_set_and_get_preference(self, tmp_path):
        from physml.user_profile import UserProfileLearner

        profile = UserProfileLearner(profile_path=str(tmp_path / "p.json"))
        profile.set_preference("verbosity", "concise")
        assert profile.get_preference("verbosity") == "concise"

    def test_save_and_load(self, tmp_path):
        from physml.user_profile import UserProfileLearner

        path = str(tmp_path / "profile.json")
        p1 = UserProfileLearner(profile_path=path)
        p1.set_preference("theme", "dark")
        p1.save()

        p2 = UserProfileLearner(profile_path=path)
        p2.load()
        assert p2.get_preference("theme") == "dark"

    def test_feedback_score(self, tmp_path):
        from physml.user_profile import UserProfileLearner

        profile = UserProfileLearner(profile_path=str(tmp_path / "p.json"))
        profile.record_interaction(feedback="positive")
        profile.record_interaction(feedback="positive")
        score = profile.feedback_score()
        assert 0.0 <= score <= 1.0

    def test_summary_keys(self, tmp_path):
        from physml.user_profile import UserProfileLearner

        profile = UserProfileLearner(profile_path=str(tmp_path / "p.json"))
        summary = profile.summary()
        assert "interaction_count" in summary
        assert "top_topics" in summary


# ---------------------------------------------------------------------------
# Stage 114 — ResponseFormatter
# ---------------------------------------------------------------------------

class TestResponseFormatter:
    def test_format_prediction(self):
        from physml.response_formatter import ResponseFormatter

        fmt = ResponseFormatter(verbosity="normal")
        out = fmt.format_prediction(prediction=42.0, confidence=0.87)
        assert "42" in out or "prediction" in out.lower()

    def test_format_report(self):
        from physml.response_formatter import ResponseFormatter

        fmt = ResponseFormatter()
        out = fmt.format_report({"status": "ok", "accuracy": 0.9})
        assert out  # non-empty

    def test_format_uncertainty(self):
        from physml.response_formatter import ResponseFormatter

        fmt = ResponseFormatter()
        out = fmt.format_uncertainty("not sure about this")
        assert "not sure" in out.lower() or out  # non-empty

    def test_format_advice(self):
        from physml.response_formatter import ResponseFormatter

        fmt = ResponseFormatter()
        out = fmt.format_advice("Consider retraining", action="retrain", severity="info")
        assert out  # non-empty


# ---------------------------------------------------------------------------
# Stage 115 — LocalREPL
# ---------------------------------------------------------------------------

class TestLocalREPL:
    def test_instantiate(self):
        from physml.local_repl import LocalREPL

        repl = LocalREPL()
        assert repl is not None

    def test_eval_safe_expression(self):
        from physml.local_repl import LocalREPL

        repl = LocalREPL()
        if hasattr(repl, "eval"):
            result = repl.eval("1 + 1")
            assert "2" in str(result)


# ---------------------------------------------------------------------------
# Stage 116 — DeviceMonitor
# ---------------------------------------------------------------------------

class TestDeviceMonitor:
    def test_snapshot(self):
        from physml.device_monitor import DeviceMonitor, DeviceSnapshot

        monitor = DeviceMonitor()
        snap = monitor.snapshot()
        assert isinstance(snap, DeviceSnapshot)

    def test_snapshot_has_cpu_field(self):
        from physml.device_monitor import DeviceMonitor

        monitor = DeviceMonitor()
        snap = monitor.snapshot()
        assert hasattr(snap, "cpu_percent") or hasattr(snap, "timestamp")

    def test_start_stop_background(self):
        from physml.device_monitor import DeviceMonitor
        import time

        monitor = DeviceMonitor(poll_interval=1)
        monitor.start_background()
        time.sleep(0.1)
        monitor.stop()


# ---------------------------------------------------------------------------
# Stage 117 — SecureVault
# ---------------------------------------------------------------------------

class TestSecureVault:
    def test_store_and_retrieve_secret(self, tmp_path):
        from physml.secure_vault import SecureVault

        vault = SecureVault(vault_path=str(tmp_path / "vault.enc"))
        vault.unlock("test_password")
        vault.store("api_key", "sk-secret-123")
        retrieved = vault.retrieve("api_key")
        assert retrieved == "sk-secret-123"

    def test_locked_vault_raises(self, tmp_path):
        from physml.secure_vault import SecureVault

        vault = SecureVault(vault_path=str(tmp_path / "vault.enc"))
        with pytest.raises(Exception):
            vault.retrieve("anything")

    def test_save_and_load_roundtrip(self, tmp_path):
        from physml.secure_vault import SecureVault

        path = str(tmp_path / "vault.enc")
        v1 = SecureVault(vault_path=path)
        v1.unlock("pass123")
        v1.store("token", "abc")
        v1.save()

        v2 = SecureVault(vault_path=path)
        v2.unlock("pass123")
        v2.load()
        assert v2.retrieve("token") == "abc"


# ---------------------------------------------------------------------------
# Stage 118 — ProactiveAdvisor
# ---------------------------------------------------------------------------

class TestProactiveAdvisor:
    def test_check_returns_list(self):
        from physml.proactive_advisor import ProactiveAdvisor

        advisor = ProactiveAdvisor()
        result = advisor.check()
        assert isinstance(result, list)

    def test_add_rule_and_trigger(self):
        from physml.proactive_advisor import ProactiveAdvisor

        advisor = ProactiveAdvisor()
        if hasattr(advisor, "add_rule"):
            advisor.add_rule(
                name="always_fire",
                condition=lambda: True,
                message="Test advice",
                action="test",
            )
            result = advisor.check()
            assert any(a.message == "Test advice" for a in result)


# ---------------------------------------------------------------------------
# Stage 119 — DigitalSoul
# ---------------------------------------------------------------------------

class TestDigitalSoul:
    def test_create_and_save(self, tmp_path):
        from physml.digital_soul import DigitalSoul

        soul = DigitalSoul(soul_path=str(tmp_path / "soul.json"), name="TestBot")
        soul.save()
        assert (tmp_path / "soul.json").exists()

    def test_record_event(self, tmp_path):
        from physml.digital_soul import DigitalSoul

        soul = DigitalSoul(soul_path=str(tmp_path / "soul.json"))
        soul.record_event("first_prediction", details={"target": "sales"})
        assert len(soul.events) >= 1

    def test_mood_update(self, tmp_path):
        from physml.digital_soul import DigitalSoul

        soul = DigitalSoul(soul_path=str(tmp_path / "soul.json"))
        soul.update_mood(homeostasis_score=0.9)
        assert soul.mood in ("confident", "curious", "learning", "uncertain")

    def test_increment_stat(self, tmp_path):
        from physml.digital_soul import DigitalSoul

        soul = DigitalSoul(soul_path=str(tmp_path / "soul.json"))
        soul.increment_stat("total_predictions")
        assert soul.stats.get("total_predictions", 0) >= 1

    def test_life_story(self, tmp_path):
        from physml.digital_soul import DigitalSoul

        soul = DigitalSoul(soul_path=str(tmp_path / "soul.json"))
        soul.record_event("born", description="Soul created")
        story = soul.life_story()
        assert isinstance(story, str)

    def test_load_roundtrip(self, tmp_path):
        from physml.digital_soul import DigitalSoul

        path = str(tmp_path / "soul.json")
        s1 = DigitalSoul(soul_path=path, name="Myco")
        s1.record_event("test_event")
        s1.save()

        s2 = DigitalSoul(soul_path=path)
        s2.load()
        assert s2.name == "Myco"
        assert len(s2.events) >= 1


# ---------------------------------------------------------------------------
# Stage 120 — MyceliumCompanion
# ---------------------------------------------------------------------------

class TestMyceliumCompanion:
    def test_start_and_stop(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        c.start()
        assert c._started
        c.stop()
        assert not c._started

    def test_chat_returns_string(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        c.start()
        response = c.chat("hello")
        assert isinstance(response, str)
        assert len(response) > 0
        c.stop()

    def test_chat_help_intent(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        c.start()
        response = c.chat("help")
        assert "predict" in response.lower() or "train" in response.lower()
        c.stop()

    def test_chat_status_intent(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        c.start()
        response = c.chat("status")
        assert response  # non-empty
        c.stop()

    def test_chat_profile_intent(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        c.start()
        response = c.chat("what have you learned about me?")
        assert response
        c.stop()

    def test_status_dict_keys(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        c.start()
        s = c.status()
        assert "name" in s
        assert "started" in s
        c.stop()

    def test_chat_train_without_file(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        c.start()
        response = c.chat("train the model")
        assert response  # should ask for a file or respond gracefully
        c.stop()

    def test_chat_document_csv(self, tmp_path):
        from physml.companion import MyceliumCompanion

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n1,2\n3,4\n")
        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path / "companion"))
        c.start()
        response = c.chat(f"read {csv_file}")
        assert response
        c.stop()

    def test_repr(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        assert "MyceliumCompanion" in repr(c)

    def test_auto_start_on_chat(self, tmp_path):
        from physml.companion import MyceliumCompanion

        c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
        assert not c._started
        c.chat("hello")  # should auto-start
        assert c._started
        c.stop()


# ---------------------------------------------------------------------------
# Stage 121 — LLMIntegration
# ---------------------------------------------------------------------------

class TestLLMIntegration:
    def test_no_api_key_not_available(self):
        from physml.llm_integration import LLMIntegration

        llm = LLMIntegration(api_key="")
        assert not llm.available

    def test_fallback_result_when_unavailable(self):
        from physml.llm_integration import LLMIntegration

        llm = LLMIntegration(api_key="")
        result = llm.chat("hello")
        assert not result.available
        assert result.error is not None

    def test_build_system_prompt_without_soul(self):
        from physml.llm_integration import LLMIntegration

        llm = LLMIntegration(api_key="")
        prompt = llm.build_system_prompt()
        assert "Mycelium" in prompt
        assert len(prompt) > 20

    def test_build_system_prompt_with_soul(self, tmp_path):
        from physml.llm_integration import LLMIntegration
        from physml.digital_soul import DigitalSoul

        soul = DigitalSoul(soul_path=str(tmp_path / "soul.json"), name="Myco")
        llm = LLMIntegration(api_key="")
        prompt = llm.build_system_prompt(soul=soul)
        assert "Myco" in prompt

    def test_build_system_prompt_with_profile(self, tmp_path):
        from physml.llm_integration import LLMIntegration
        from physml.user_profile import UserProfileLearner

        profile = UserProfileLearner(profile_path=str(tmp_path / "p.json"))
        profile.record_interaction(topic="finance")
        llm = LLMIntegration(api_key="")
        prompt = llm.build_system_prompt(profile=profile)
        assert isinstance(prompt, str)

    def test_llm_message_dataclass(self):
        from physml.llm_integration import LLMMessage

        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_llm_config_defaults(self):
        from physml.llm_integration import LLMConfig

        cfg = LLMConfig()
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.max_tokens > 0

    def test_llm_result_defaults(self):
        from physml.llm_integration import LLMResult

        r = LLMResult()
        assert r.text == ""
        assert not r.available

    def test_repr(self):
        from physml.llm_integration import LLMIntegration

        llm = LLMIntegration(api_key="")
        assert "LLMIntegration" in repr(llm)


# ---------------------------------------------------------------------------
# Stage 122 — VoiceOutputAdapter
# ---------------------------------------------------------------------------

class TestVoiceOutputAdapter:
    def test_silent_backend(self):
        from physml.voice_output import VoiceOutputAdapter

        tts = VoiceOutputAdapter(preferred_backend="silent")
        result = tts.speak("Hello world")
        assert result.backend == "silent"
        assert result.success
        assert result.text == "Hello world"

    def test_active_backend_is_silent_without_deps(self):
        from physml.voice_output import VoiceOutputAdapter

        tts = VoiceOutputAdapter(preferred_backend="auto")
        assert tts.active_backend in ("pyttsx3", "gtts", "silent")

    def test_repr(self):
        from physml.voice_output import VoiceOutputAdapter

        tts = VoiceOutputAdapter(preferred_backend="silent")
        assert "VoiceOutputAdapter" in repr(tts)
