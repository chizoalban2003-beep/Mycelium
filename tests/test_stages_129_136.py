"""Tests for Stages 129–136.

Stage 129 — ScreenAgent
Stage 130 — BrowserAgent
Stage 131 — PermissionManager
Stage 132 — FileWatcher
Stage 133 — Notifier
Stage 134 — KnowledgeExtractor
Stage 135 — FeedbackLoop
Stage 136 — PersonalisationManager
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest


# ===========================================================================
# Stage 129 — ScreenAgent
# ===========================================================================


class TestScreenAgent:
    def test_import(self):
        from physml.screen_agent import ScreenAgent

        assert ScreenAgent is not None

    def test_instantiation(self, tmp_path):
        from physml.screen_agent import ScreenAgent

        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        assert sa is not None

    def test_available_is_bool(self, tmp_path):
        from physml.screen_agent import ScreenAgent

        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        assert isinstance(sa.available, bool)

    def test_screenshot_returns_path_or_none(self, tmp_path):
        from physml.screen_agent import ScreenAgent

        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        result = sa.screenshot()
        assert result is None or isinstance(result, str)

    def test_click_returns_bool_when_unavailable(self, tmp_path):
        from physml.screen_agent import ScreenAgent, _PAG_OK

        if _PAG_OK:
            pytest.skip("pyautogui available — skip no-op test")
        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        assert sa.click(0, 0) is False

    def test_type_text_returns_bool(self, tmp_path):
        from physml.screen_agent import ScreenAgent, _PAG_OK

        if _PAG_OK:
            pytest.skip("pyautogui available")
        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        assert sa.type_text("hello") is False

    def test_hotkey_returns_bool(self, tmp_path):
        from physml.screen_agent import ScreenAgent, _PAG_OK

        if _PAG_OK:
            pytest.skip("pyautogui available")
        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        assert sa.hotkey("ctrl", "c") is False

    def test_scroll_returns_bool(self, tmp_path):
        from physml.screen_agent import ScreenAgent, _PAG_OK

        if _PAG_OK:
            pytest.skip("pyautogui available")
        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        assert sa.scroll() is False

    def test_find_text_returns_none_when_no_screen(self, tmp_path):
        from physml.screen_agent import ScreenAgent

        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        result = sa.find_text_on_screen("hello")
        assert result is None

    def test_status_dict(self, tmp_path):
        from physml.screen_agent import ScreenAgent

        sa = ScreenAgent(screenshot_dir=str(tmp_path))
        s = sa.status()
        assert "available" in s
        assert "pyautogui" in s
        assert "mss" in s
        assert "screenshot_dir" in s


# ===========================================================================
# Stage 130 — BrowserAgent
# ===========================================================================


class TestBrowserAgent:
    def test_import(self):
        from physml.browser_agent import BrowserAgent

        assert BrowserAgent is not None

    def test_instantiation(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert ba is not None

    def test_available_is_bool(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert isinstance(ba.available, bool)

    def test_open_returns_bool_when_unavailable(self):
        from physml.browser_agent import BrowserAgent, _PW_OK

        if _PW_OK:
            pytest.skip("playwright available")
        ba = BrowserAgent()
        assert ba.open() is False

    def test_goto_when_not_open(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert ba.goto("http://example.com") is False

    def test_get_text_when_not_open(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert ba.get_text() == ""

    def test_extract_links_when_not_open(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert ba.extract_links() == []

    def test_current_url_when_not_open(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert ba.current_url() == ""

    def test_title_when_not_open(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert ba.title() == ""

    def test_run_js_when_not_open(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        assert ba.run_js("1+1") is None

    def test_status_dict(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        s = ba.status()
        assert "available" in s
        assert "open" in s

    def test_close_safe_when_not_open(self):
        from physml.browser_agent import BrowserAgent

        ba = BrowserAgent()
        ba.close()  # should not raise


# ===========================================================================
# Stage 131 — PermissionManager
# ===========================================================================


class TestPermissionManager:
    def test_import(self):
        from physml.permission_manager import PermissionManager, PermissionLevel

        assert PermissionManager is not None
        assert PermissionLevel is not None

    def test_allow_default_actions(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        assert pm.check("file.read") is True
        assert pm.check("screen.screenshot") is True
        assert pm.check("predict") is True

    def test_deny_default_dangerous_actions(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        assert pm.check("process.kill") is False
        assert pm.check("system.reboot") is False

    def test_ask_actions_denied_without_callback(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        assert pm.check("file.write") is False

    def test_grant_and_check(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        pm.grant("file.write")
        assert pm.check("file.write") is True

    def test_deny_overrides_grant(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        pm.grant("file.write")
        pm.deny("file.write")
        assert pm.check("file.write") is False

    def test_reset_reverts_to_policy(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        pm.grant("file.write")
        pm.reset("file.write")
        assert pm.check("file.write") is False  # ASK → deny without callback

    def test_set_policy(self, tmp_path):
        from physml.permission_manager import PermissionManager, PermissionLevel

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        pm.set_policy("file.write", PermissionLevel.ALLOW)
        assert pm.check("file.write") is True

    def test_auto_ask_callback(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(
            config_path=str(tmp_path / "perms.json"),
            auto_ask_callback=lambda action, desc: True,
        )
        assert pm.check("file.write") is True

    def test_summary_keys(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        s = pm.summary()
        assert "granted" in s
        assert "denied" in s
        assert "policy" in s

    def test_allowed_actions_list(self, tmp_path):
        from physml.permission_manager import PermissionManager

        pm = PermissionManager(config_path=str(tmp_path / "perms.json"))
        actions = pm.allowed_actions()
        assert isinstance(actions, list)
        assert "file.read" in actions

    def test_persistence_roundtrip(self, tmp_path):
        from physml.permission_manager import PermissionManager

        path = str(tmp_path / "perms.json")
        pm1 = PermissionManager(config_path=path)
        pm1.grant("file.write")

        pm2 = PermissionManager(config_path=path)
        assert pm2.check("file.write") is True


# ===========================================================================
# Stage 132 — FileWatcher
# ===========================================================================


class TestFileWatcher:
    def test_import(self):
        from physml.file_watcher import FileWatcher

        assert FileWatcher is not None

    def test_instantiation(self, tmp_path):
        from physml.file_watcher import FileWatcher

        fw = FileWatcher(watch_dirs=[str(tmp_path)])
        assert fw is not None

    def test_available_is_bool(self, tmp_path):
        from physml.file_watcher import FileWatcher

        fw = FileWatcher(watch_dirs=[str(tmp_path)])
        assert isinstance(fw.available, bool)

    def test_start_and_stop(self, tmp_path):
        from physml.file_watcher import FileWatcher

        fw = FileWatcher(watch_dirs=[str(tmp_path)], poll_interval=0.1)
        fw.start()
        assert fw._running is True
        fw.stop()
        assert fw._running is False

    def test_callback_fires_on_new_file(self, tmp_path):
        from physml.file_watcher import FileWatcher, _WD_OK

        if _WD_OK:
            pytest.skip("watchdog active — timing-sensitive; skip in CI")
        seen = []
        fw = FileWatcher(
            watch_dirs=[str(tmp_path)],
            callback=seen.append,
            extensions={".csv"},
            poll_interval=0.1,
        )
        fw.start()
        (tmp_path / "test.csv").write_text("a,b\n1,2\n")
        time.sleep(0.5)
        fw.stop()
        assert len(seen) >= 1

    def test_add_directory(self, tmp_path):
        from physml.file_watcher import FileWatcher

        fw = FileWatcher()
        fw.add_directory(str(tmp_path))
        assert str(tmp_path.expanduser()) in fw.watch_dirs or str(tmp_path) in fw.watch_dirs

    def test_status_dict(self, tmp_path):
        from physml.file_watcher import FileWatcher

        fw = FileWatcher(watch_dirs=[str(tmp_path)])
        s = fw.status()
        assert "running" in s
        assert "backend" in s
        assert "watching" in s
        assert "extensions" in s

    def test_double_start_safe(self, tmp_path):
        from physml.file_watcher import FileWatcher

        fw = FileWatcher(watch_dirs=[str(tmp_path)], poll_interval=0.1)
        fw.start()
        fw.start()  # should not raise or duplicate threads
        fw.stop()


# ===========================================================================
# Stage 133 — Notifier
# ===========================================================================


class TestNotifier:
    def test_import(self):
        from physml.notifier import Notifier

        assert Notifier is not None

    def test_instantiation(self):
        from physml.notifier import Notifier

        n = Notifier(app_name="Test")
        assert n.app_name == "Test"

    def test_available_is_bool(self):
        from physml.notifier import Notifier

        n = Notifier()
        assert isinstance(n.available, bool)

    def test_send_returns_bool(self):
        from physml.notifier import Notifier

        n = Notifier()
        result = n.send("Test Title", "Test message")
        assert isinstance(result, bool)

    def test_send_logs_to_history(self):
        from physml.notifier import Notifier

        n = Notifier()
        n.send("T1", "M1")
        n.send("T2", "M2")
        h = n.history()
        assert len(h) == 2
        assert h[0]["title"] == "T1"
        assert h[1]["title"] == "T2"

    def test_send_alert(self):
        from physml.notifier import Notifier

        n = Notifier()
        result = n.send_alert("Something went wrong")
        assert isinstance(result, bool)

    def test_send_success(self):
        from physml.notifier import Notifier

        n = Notifier()
        result = n.send_success("Model trained!")
        assert isinstance(result, bool)

    def test_history_limit(self):
        from physml.notifier import Notifier

        n = Notifier()
        for i in range(25):
            n.send(f"T{i}", "msg")
        h = n.history(n=10)
        assert len(h) == 10

    def test_remind_schedules_without_error(self):
        from physml.notifier import Notifier

        n = Notifier()
        n.remind("Check results", delay_seconds=9999)  # won't fire in test

    def test_status_dict(self):
        from physml.notifier import Notifier

        n = Notifier()
        n.send("x", "y")
        s = n.status()
        assert "available" in s
        assert "app_name" in s
        assert "total_logged" in s
        assert s["total_logged"] == 1


# ===========================================================================
# Stage 134 — KnowledgeExtractor
# ===========================================================================


class TestKnowledgeExtractor:
    def test_import(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        assert KnowledgeExtractor is not None

    def test_extract_name(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("My name is Alice.")
        names = [f for f in facts if f["predicate"] == "name"]
        assert len(names) >= 1
        assert names[0]["object"] == "Alice"

    def test_extract_name_no_overrun(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("My name is Alex and I work as a data scientist.")
        names = [f for f in facts if f["predicate"] == "name"]
        assert names[0]["object"] == "Alex"

    def test_extract_occupation(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("I work as a software engineer.")
        occupations = [f for f in facts if f["predicate"] == "occupation"]
        assert len(occupations) >= 1
        assert "software engineer" in occupations[0]["object"]

    def test_extract_location(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("I live in London.")
        locations = [f for f in facts if f["predicate"] == "location"]
        assert len(locations) >= 1
        assert "London" in locations[0]["object"]

    def test_extract_likes(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("I love hiking.")
        likes = [f for f in facts if f["predicate"] == "likes"]
        assert len(likes) >= 1

    def test_extract_dislikes(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("I hate traffic.")
        dislikes = [f for f in facts if f["predicate"] == "dislikes"]
        assert len(dislikes) >= 1

    def test_extract_employer(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("I work at Acme Corp.")
        employers = [f for f in facts if f["predicate"] == "employer"]
        assert len(employers) >= 1

    def test_extract_goal(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("I want to learn machine learning.")
        goals = [f for f in facts if f["predicate"] == "goal"]
        assert len(goals) >= 1

    def test_deduplication(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("My name is Bob. My name is Bob.")
        names = [f for f in facts if f["predicate"] == "name"]
        assert len(names) == 1

    def test_empty_text(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("")
        assert facts == []

    def test_all_facts_have_subject(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("My name is Carol. I live in Paris.")
        for f in facts:
            assert f["subject"] == "user"

    def test_store_without_backends(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract("My name is Dave.")
        stored = ke.store(facts)
        assert stored == len(facts)
        assert ke.status()["facts_stored"] == stored

    def test_extract_and_store(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        facts = ke.extract_and_store("I work as a doctor.")
        assert len(facts) >= 1
        assert ke.status()["facts_stored"] >= 1

    def test_llm_extract_falls_back(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor(llm=None)
        facts = ke.llm_extract("My name is Eve.")
        names = [f for f in facts if f["predicate"] == "name"]
        assert names[0]["object"] == "Eve"

    def test_status_keys(self):
        from physml.knowledge_extractor import KnowledgeExtractor

        ke = KnowledgeExtractor()
        s = ke.status()
        assert "facts_stored" in s
        assert "knowledge_graph_connected" in s
        assert "vector_memory_connected" in s


# ===========================================================================
# Stage 135 — FeedbackLoop
# ===========================================================================


class TestFeedbackLoop:
    def test_import(self):
        from physml.feedback_loop import FeedbackLoop, CorrectionRecord

        assert FeedbackLoop is not None
        assert CorrectionRecord is not None

    def test_record_correction_no_model(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        result = fl.record_correction([1.0, 2.0, 3.0], correct_label=1)
        assert isinstance(result, bool)

    def test_correction_stored(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        fl.record_correction([1.0], correct_label=5)
        assert len(fl._corrections) == 1
        assert fl._corrections[0].correct_label == 5

    def test_parse_and_record_detects_correction(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        response = fl.parse_and_record("That's wrong, it should be 42.")
        assert response is not None
        assert "42" in response or "correct" in response.lower()

    def test_parse_and_record_no_correction(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        response = fl.parse_and_record("Thanks, that looks great!")
        assert response is None

    def test_parse_and_record_with_features(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        response = fl.parse_and_record("Incorrect, the answer is 7.", last_features=[1.0, 2.0])
        assert response is not None

    def test_extract_value_numeric(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        v = fl._extract_value("It should be 3.14.")
        assert v == pytest.approx(3.14)

    def test_extract_value_integer(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        v = fl._extract_value("The answer is 42.")
        assert v == 42

    def test_extract_value_string(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        v = fl._extract_value("It should be positive.")
        assert v == "positive"

    def test_max_buffer_trim(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop(max_buffer=5)
        for i in range(10):
            fl.record_correction([float(i)], correct_label=i)
        assert len(fl._corrections) <= 5

    def test_status_dict(self):
        from physml.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        fl.record_correction([1.0], correct_label=0)
        s = fl.status()
        assert "total_corrections" in s
        assert "pending" in s
        assert "total_applied" in s
        assert "model_connected" in s
        assert s["total_corrections"] == 1

    def test_correction_record_as_dict(self):
        from physml.feedback_loop import CorrectionRecord

        cr = CorrectionRecord(
            timestamp=1.0,
            features=[1.0, 2.0],
            correct_label=3,
            predicted_label=0,
        )
        d = cr.as_dict()
        assert d["correct_label"] == 3
        assert d["n_features"] == 2


# ===========================================================================
# Stage 136 — PersonalisationManager
# ===========================================================================


class TestPersonalisationManager:
    def test_import(self):
        from physml.personalisation import PersonalisationManager

        assert PersonalisationManager is not None

    def test_instantiation(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        assert pm is not None

    def test_default_values(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        assert pm.get("language") == "en"
        assert pm.get("verbosity") == "normal"
        assert pm.get("agent_name") == "Myco"

    def test_set_and_get(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.set("name", "Alex")
        assert pm.get("name") == "Alex"

    def test_invalid_verbosity_raises(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        with pytest.raises(ValueError):
            pm.set("verbosity", "ultra")

    def test_valid_verbosity_values(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        for v in ("concise", "normal", "verbose"):
            pm.set("verbosity", v)
            assert pm.get("verbosity") == v

    def test_update_batch(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.update({"name": "Bob", "language": "fr"})
        assert pm.get("name") == "Bob"
        assert pm.get("language") == "fr"

    def test_reset_single_key(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.set("name", "Carol")
        pm.reset("name")
        assert pm.get("name") is None

    def test_reset_all(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.set("name", "Dave")
        pm.set("language", "de")
        pm.reset()
        assert pm.get("name") is None
        assert pm.get("language") == "en"

    def test_persistence_roundtrip(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        path = str(tmp_path / "config.json")
        pm1 = PersonalisationManager(config_path=path)
        pm1.set("name", "Eve")

        pm2 = PersonalisationManager(config_path=path)
        assert pm2.get("name") == "Eve"

    def test_profile_returns_dict(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.set("name", "Frank")
        p = pm.profile()
        assert isinstance(p, dict)
        assert p["name"] == "Frank"

    def test_greeting_with_name(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.set("name", "Grace")
        g = pm.greeting()
        assert "Grace" in g

    def test_greeting_without_name(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        g = pm.greeting()
        assert "Hello" in g

    def test_system_prompt_additions_empty_defaults(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        s = pm.system_prompt_additions()
        assert isinstance(s, str)

    def test_system_prompt_includes_name(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.set("name", "Hal")
        s = pm.system_prompt_additions()
        assert "Hal" in s

    def test_system_prompt_includes_verbosity(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        pm.set("verbosity", "concise")
        s = pm.system_prompt_additions()
        assert "concise" in s.lower() or "2-3 sentences" in s

    def test_keys_list(self, tmp_path):
        from physml.personalisation import PersonalisationManager

        pm = PersonalisationManager(config_path=str(tmp_path / "config.json"))
        keys = pm.keys()
        assert "name" in keys
        assert "language" in keys
        assert "verbosity" in keys
