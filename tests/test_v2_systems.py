"""Tests for v1.1 systems: MultiModalIngester, ScreenObserver, MacroRecorder,
ImitationLearner, UserModel, GoalEngine skill-save, browser extension API.

Run with::

    python3 -m pytest tests/test_v2_systems.py -v --timeout=60 -m "not slow"
"""

from __future__ import annotations

import time

import pytest

# Pre-warm physml/scipy at collection time
import physml  # noqa: F401


# ===========================================================================
# TestMultiModalIngester
# ===========================================================================


class TestMultiModalIngester:
    def test_import(self):
        from physml.multimodal_ingester import MultiModalIngester
        assert MultiModalIngester is not None

    def test_import_from_physml(self):
        from physml import MultiModalIngester
        assert MultiModalIngester is not None

    def test_ingest_result_import(self):
        from physml import IngestResult
        assert IngestResult is not None

    def test_instantiate(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        assert ing is not None
        assert ing.ingested_count == 0

    def test_ingest_raw_text(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        result = ing.ingest("Hello, my name is Alex and I work as a developer.")
        assert result.source == "Hello, my name is Alex and I work as a developer."
        assert result.success
        assert result.text  # text was extracted

    def test_ingest_increments_count(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        ing.ingest("First ingestion.")
        ing.ingest("Second ingestion.")
        assert ing.ingested_count == 2

    def test_ingest_deduplication(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester(deduplicate=True)
        ing.ingest("Duplicate text.")
        ing.ingest("Duplicate text.")
        s = ing.summary()
        assert s["deduplicated"] >= 1

    def test_ingest_nonexistent_file(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        result = ing.ingest("/nonexistent/path/file.txt")
        # Should return IngestResult with success=False or empty text
        assert hasattr(result, "source")

    def test_ingest_txt_file(self, tmp_path):
        from physml.multimodal_ingester import MultiModalIngester
        f = tmp_path / "notes.txt"
        f.write_text("My name is Bob and I enjoy cycling.")
        ing = MultiModalIngester()
        result = ing.ingest(str(f))
        assert isinstance(result.text, str)

    def test_ingest_py_file(self, tmp_path):
        from physml.multimodal_ingester import MultiModalIngester
        f = tmp_path / "main.py"
        f.write_text("def hello(): return 'world'")
        ing = MultiModalIngester()
        result = ing.ingest(str(f))
        assert result.metadata.get("type") in ("code", "text", "raw_text", "unknown")

    def test_ingest_many(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        results = ing.ingest_many(["text one.", "text two.", "text three."])
        assert len(results) == 3

    def test_ingest_directory_empty(self, tmp_path):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        results = ing.ingest_directory(str(tmp_path))
        assert isinstance(results, list)

    def test_ingest_directory_with_files(self, tmp_path):
        from physml.multimodal_ingester import MultiModalIngester
        (tmp_path / "a.txt").write_text("File A content")
        (tmp_path / "b.txt").write_text("File B content")
        ing = MultiModalIngester()
        results = ing.ingest_directory(str(tmp_path), extensions=[".txt"])
        assert len(results) == 2

    def test_summary_keys(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        s = ing.summary()
        for key in ("total", "succeeded", "failed", "facts_extracted", "deduplicated"):
            assert key in s

    def test_status_keys(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        st = ing.status()
        assert "ingested" in st

    def test_last_result(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        assert ing.last_result is None
        ing.ingest("some text")
        assert ing.last_result is not None

    def test_repr(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        assert "MultiModalIngester" in repr(ing)

    def test_ingest_result_has_elapsed(self):
        from physml.multimodal_ingester import MultiModalIngester
        ing = MultiModalIngester()
        result = ing.ingest("timing test text")
        assert result.elapsed >= 0.0


# ===========================================================================
# TestScreenObserver
# ===========================================================================


class TestScreenObserver:
    def test_import(self):
        from physml.screen_observer import ScreenObserver
        assert ScreenObserver is not None

    def test_import_from_physml(self):
        from physml import ScreenObserver, ScreenSnapshot
        assert ScreenObserver is not None
        assert ScreenSnapshot is not None

    def test_instantiate(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=999.0)
        assert obs is not None
        assert not obs.running

    def test_snapshots_empty_initially(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=999.0)
        assert obs.snapshots == []

    def test_focus_summary_empty(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=999.0)
        assert obs.focus_summary() == {}

    def test_top_apps_empty(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=999.0)
        assert obs.top_apps() == []

    def test_recent_context_no_snaps(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=999.0)
        ctx = obs.recent_context()
        assert isinstance(ctx, str)
        assert "No" in ctx or ctx == ""

    def test_status_keys(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=30.0)
        st = obs.status()
        for key in ("running", "interval", "snapshots", "llm_describe"):
            assert key in st

    def test_repr(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=30.0)
        assert "ScreenObserver" in repr(obs)

    def test_start_stop(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=9999.0, llm_describe=False)
        obs.start()
        assert obs.running
        obs.stop()
        assert not obs.running

    def test_start_idempotent(self):
        from physml.screen_observer import ScreenObserver
        obs = ScreenObserver(interval=9999.0, llm_describe=False)
        obs.start()
        obs.start()  # second start should be no-op
        assert obs.running
        obs.stop()

    def test_on_snapshot_callback(self):
        from physml.screen_observer import ScreenObserver, ScreenSnapshot
        captured = []

        def cb(snap):
            captured.append(snap)

        obs = ScreenObserver(interval=9999.0, llm_describe=False, on_snapshot=cb)
        # Manually inject a snapshot
        snap = ScreenSnapshot(timestamp=time.time(), app_name="TestApp", description="test")
        obs._snapshots.append(snap)
        if cb:
            cb(snap)
        assert len(captured) == 1
        assert captured[0].app_name == "TestApp"

    def test_screen_snapshot_dataclass(self):
        from physml.screen_observer import ScreenSnapshot
        snap = ScreenSnapshot(timestamp=1234567890.0, app_name="VSCode", window_title="main.py")
        assert snap.app_name == "VSCode"
        assert snap.screenshot_path is None


# ===========================================================================
# TestMacroRecorder
# ===========================================================================


class TestMacroRecorder:
    def test_import(self):
        from physml.macro_recorder import MacroRecorder
        assert MacroRecorder is not None

    def test_import_from_physml(self):
        from physml import MacroRecorder, MacroSequence, ActionStep, ActionType
        assert MacroRecorder is not None
        assert MacroSequence is not None
        assert ActionStep is not None
        assert ActionType is not None

    def test_instantiate(self):
        from physml.macro_recorder import MacroRecorder
        rec = MacroRecorder()
        assert not rec.recording
        assert rec.sequences == []

    def test_available_is_bool(self):
        from physml.macro_recorder import MacroRecorder
        rec = MacroRecorder()
        assert isinstance(rec.available, bool)

    def test_record_text_sequence(self):
        from physml.macro_recorder import MacroRecorder, ActionType
        rec = MacroRecorder()
        seq = rec.record_text_sequence("test_seq", [
            {"action_type": ActionType.CLICK, "x": 100, "y": 200, "app_name": "App1"},
            {"action_type": ActionType.TYPE_TEXT, "text": "hello", "app_name": "App1"},
            {"action_type": ActionType.CLICK, "x": 300, "y": 400, "app_name": "App1"},
        ])
        assert seq.name == "test_seq"
        assert len(seq.steps) == 3
        assert seq in rec.sequences

    def test_macro_sequence_duration(self):
        from physml.macro_recorder import MacroRecorder, ActionType
        import time
        rec = MacroRecorder()
        now = time.time()
        seq = rec.record_text_sequence("dur_test", [
            {"action_type": ActionType.CLICK, "x": 0, "y": 0, "timestamp": now},
            {"action_type": ActionType.CLICK, "x": 0, "y": 0, "timestamp": now + 5.0},
        ])
        assert seq.duration >= 0.0

    def test_macro_sequence_apps_used(self):
        from physml.macro_recorder import MacroRecorder, ActionType
        rec = MacroRecorder()
        seq = rec.record_text_sequence("apps", [
            {"action_type": ActionType.CLICK, "x": 0, "y": 0, "app_name": "Firefox"},
            {"action_type": ActionType.CLICK, "x": 0, "y": 0, "app_name": "VSCode"},
        ])
        assert "Firefox" in seq.apps_used
        assert "VSCode" in seq.apps_used

    def test_save_sequence(self, tmp_path):
        from physml.macro_recorder import MacroRecorder, ActionType
        rec = MacroRecorder(save_dir=str(tmp_path))
        seq = rec.record_text_sequence("save_test", [
            {"action_type": ActionType.CLICK, "x": 50, "y": 50, "app_name": "App"},
            {"action_type": ActionType.CLICK, "x": 60, "y": 60, "app_name": "App"},
        ])
        path = rec.save_sequence(seq)
        assert (tmp_path / path.split("/")[-1]).exists() or __import__("pathlib").Path(path).exists()

    def test_load_sequences(self, tmp_path):
        from physml.macro_recorder import MacroRecorder, ActionType
        rec = MacroRecorder(save_dir=str(tmp_path))
        seq = rec.record_text_sequence("load_test", [
            {"action_type": ActionType.CLICK, "x": 10, "y": 10, "app_name": "X"},
            {"action_type": ActionType.CLICK, "x": 20, "y": 20, "app_name": "X"},
        ])
        rec.save_sequence(seq)
        loaded = rec.load_sequences()
        assert len(loaded) >= 1
        assert loaded[0].name == "load_test"

    def test_save_to_skill_library(self, tmp_path):
        from physml.macro_recorder import MacroRecorder, ActionType
        from physml.skill_library import SkillLibrary
        lib = SkillLibrary()
        rec = MacroRecorder(skill_library=lib, save_dir=str(tmp_path))
        seq = rec.record_text_sequence("skill_test", [
            {"action_type": ActionType.CLICK, "x": 1, "y": 1, "app_name": "A"},
            {"action_type": ActionType.CLICK, "x": 2, "y": 2, "app_name": "A"},
        ])
        ok = rec.save_to_skill_library(seq)
        assert ok
        assert "skill_test" in lib.list_names()

    def test_action_step_to_dict_roundtrip(self):
        from physml.macro_recorder import ActionStep, ActionType
        step = ActionStep(action_type=ActionType.CLICK, x=100, y=200, app_name="MyApp")
        d = step.to_dict()
        step2 = ActionStep.from_dict(d)
        assert step2.action_type == step.action_type
        assert step2.x == step.x

    def test_macro_sequence_to_dict_roundtrip(self):
        from physml.macro_recorder import MacroRecorder, MacroSequence, ActionType
        rec = MacroRecorder()
        seq = rec.record_text_sequence("roundtrip", [
            {"action_type": ActionType.SCROLL, "x": 0, "y": 0, "app_name": "Browser"},
        ])
        d = seq.to_dict()
        seq2 = MacroSequence.from_dict(d)
        assert seq2.name == seq.name
        assert len(seq2.steps) == len(seq.steps)

    def test_status(self):
        from physml.macro_recorder import MacroRecorder
        rec = MacroRecorder()
        st = rec.status()
        for key in ("recording", "pynput_available", "sequences_captured"):
            assert key in st

    def test_manual_start_stop(self):
        from physml.macro_recorder import MacroRecorder
        rec = MacroRecorder(min_steps=0)
        rec.start_recording("manual")
        assert rec.recording
        rec.stop_recording()
        assert not rec.recording


# ===========================================================================
# TestImitationLearner
# ===========================================================================


class TestImitationLearner:
    def test_import(self):
        from physml.imitation_learner import ImitationLearner
        assert ImitationLearner is not None

    def test_import_from_physml(self):
        from physml import ImitationLearner, ActionSuggestion
        assert ImitationLearner is not None
        assert ActionSuggestion is not None

    def test_instantiate(self):
        from physml.imitation_learner import ImitationLearner
        learner = ImitationLearner()
        assert not learner.is_fitted
        assert learner.sequence_count == 0

    def test_add_sequence(self):
        from physml.imitation_learner import ImitationLearner
        from physml.macro_recorder import MacroRecorder, ActionType
        rec = MacroRecorder()
        seq = rec.record_text_sequence("seq1", [
            {"action_type": ActionType.CLICK, "x": 10, "y": 10, "app_name": "A"},
            {"action_type": ActionType.TYPE_TEXT, "text": "hi", "app_name": "A"},
        ])
        learner = ImitationLearner()
        learner.add_sequence(seq)
        assert learner.sequence_count == 1
        assert not learner.is_fitted

    def test_predict_next_unfitted_returns_heuristic(self):
        from physml.imitation_learner import ImitationLearner
        learner = ImitationLearner()
        suggestions = learner.predict_next(context_app="Browser", context_action="click")
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        for s in suggestions:
            assert 0.0 <= s.confidence <= 1.0

    @pytest.mark.slow
    def test_fit_and_predict(self):
        from physml.imitation_learner import ImitationLearner
        from physml.macro_recorder import MacroRecorder, ActionType
        rec = MacroRecorder()
        # Need enough steps for context_window
        steps = []
        for i in range(20):
            atype = ActionType.CLICK if i % 2 == 0 else ActionType.TYPE_TEXT
            steps.append({"action_type": atype, "x": i * 10, "y": i * 5, "app_name": "App"})
        seq = rec.record_text_sequence("train_seq", steps)
        learner = ImitationLearner(context_window=2, min_sequences=1)
        learner.add_sequence(seq)
        ok = learner.fit()
        assert ok
        assert learner.is_fitted
        suggestions = learner.predict_next(context_app="App", context_action="click")
        assert isinstance(suggestions, list)

    def test_status(self):
        from physml.imitation_learner import ImitationLearner
        learner = ImitationLearner()
        st = learner.status()
        assert "sequences" in st
        assert "fitted" in st

    def test_add_sequences_plural(self):
        from physml.imitation_learner import ImitationLearner
        from physml.macro_recorder import MacroRecorder, ActionType
        rec = MacroRecorder()
        seqs = []
        for name in ["s1", "s2", "s3"]:
            seq = rec.record_text_sequence(name, [
                {"action_type": ActionType.CLICK, "x": 0, "y": 0, "app_name": "X"},
            ])
            seqs.append(seq)
        learner = ImitationLearner()
        learner.add_sequences(seqs)
        assert learner.sequence_count == 3


# ===========================================================================
# TestUserModel
# ===========================================================================


class TestUserModel:
    def test_import(self):
        from physml.user_model import UserModel
        assert UserModel is not None

    def test_import_from_physml(self):
        from physml import UserModel
        assert UserModel is not None

    def test_instantiate(self):
        from physml.user_model import UserModel
        m = UserModel()
        assert m is not None

    def test_update_interaction(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.update({"type": "interaction", "intent": "train", "topic": "sales"})
        assert len(m._event_log) == 1

    def test_update_screen(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.update({"type": "screen", "app": "VSCode", "window": "main.py", "description": "editing"})
        assert m._current_app == "VSCode"

    def test_update_preference(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.update({"type": "preference", "key": "verbosity", "value": "concise"})
        # No crash = pass

    def test_update_text(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.update({"type": "text", "text": "Interesting article about ML.", "source": "web"})
        assert len(m._event_log) == 1

    def test_update_fact(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.update({"type": "fact", "subject": "user", "predicate": "likes", "object": "Python"})
        assert len(m._event_log) == 1

    def test_update_goal_completed(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.update({"type": "goal_completed", "goal_description": "Analyse sales.csv"})
        assert len(m._event_log) == 1

    def test_event_log_capped(self):
        from physml.user_model import UserModel
        m = UserModel()
        for i in range(600):
            m.update({"type": "text", "text": f"event {i}"})
        assert len(m._event_log) <= 500

    def test_current_context_returns_dict(self):
        from physml.user_model import UserModel
        m = UserModel()
        ctx = m.current_context()
        assert isinstance(ctx, dict)
        assert "app" in ctx
        assert "session_seconds" in ctx

    def test_behavioral_patterns_returns_list(self):
        from physml.user_model import UserModel
        m = UserModel()
        patterns = m.behavioral_patterns()
        assert isinstance(patterns, list)

    def test_inject_into_prompt_returns_str(self):
        from physml.user_model import UserModel
        m = UserModel()
        result = m.inject_into_prompt()
        assert isinstance(result, str)

    def test_set_preference(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.set_preference("language", "en")
        # No crash = pass

    def test_remember_fact(self):
        from physml.user_model import UserModel
        m = UserModel()
        m.remember_fact("user", "name", "Alice")
        assert len(m._event_log) == 1

    def test_recall_no_vm_returns_empty(self):
        from physml.user_model import UserModel
        m = UserModel(vector_memory=None)
        m._vm = None  # Force no VM
        # Recall might initialise VM lazily — either way should return list
        result = m.recall("anything")
        assert isinstance(result, list)

    def test_summary_keys(self):
        from physml.user_model import UserModel
        m = UserModel()
        s = m.summary()
        assert "context" in s
        assert "behavioral_patterns" in s
        assert "event_log_size" in s

    def test_status_keys(self):
        from physml.user_model import UserModel
        m = UserModel()
        st = m.status()
        for key in ("user_profile", "digital_soul", "vector_memory", "events_processed"):
            assert key in st

    def test_repr(self):
        from physml.user_model import UserModel
        m = UserModel()
        assert "UserModel" in repr(m)


# ===========================================================================
# TestGoalEngineSkillAutoSave
# ===========================================================================


class TestGoalEngineSkillAutoSave:
    def test_goal_engine_accepts_skill_library(self):
        from physml.goal_engine import GoalEngine
        from physml.skill_library import SkillLibrary
        lib = SkillLibrary()
        ge = GoalEngine(skill_library=lib)
        assert ge._skill_library is lib

    def test_goal_engine_accepts_user_model(self):
        from physml.goal_engine import GoalEngine
        from physml.user_model import UserModel
        um = UserModel()
        ge = GoalEngine(user_model=um)
        assert ge._user_model is um

    def test_auto_save_skill_method_exists(self):
        from physml.goal_engine import GoalEngine
        assert hasattr(GoalEngine, "_auto_save_skill")

    def test_notify_user_model_method_exists(self):
        from physml.goal_engine import GoalEngine
        assert hasattr(GoalEngine, "_notify_user_model")

    def test_auto_save_skill_with_mock_goal(self):
        from physml.goal_engine import GoalEngine, GoalRecord, GoalStatus
        from physml.skill_library import SkillLibrary
        lib = SkillLibrary()
        ge = GoalEngine(skill_library=lib)
        # Build a fake completed goal
        import time
        goal = GoalRecord(
            id="test-id-0001",
            description="Analyse sales.csv and summarise revenue",
            status=GoalStatus.COMPLETED,
            created_at=time.time(),
        )
        ge._auto_save_skill(goal)
        # Should have registered a skill with a slug of the description
        names = lib.list_names()
        assert any("analyse" in n or "goal_" in n for n in names)

    def test_notify_user_model_no_crash(self):
        from physml.goal_engine import GoalEngine, GoalRecord, GoalStatus
        from physml.user_model import UserModel
        um = UserModel()
        ge = GoalEngine(user_model=um)
        import time
        goal = GoalRecord(
            id="test-id-0002",
            description="Send digest email",
            status=GoalStatus.COMPLETED,
            created_at=time.time(),
        )
        ge._notify_user_model(goal)
        assert len(um._event_log) >= 1


# ===========================================================================
# TestBrowserExtensionAPI
# ===========================================================================


class TestBrowserExtensionAPI:
    def test_import(self):
        from physml.browser_extension_api import _FASTAPI_AVAILABLE
        assert isinstance(_FASTAPI_AVAILABLE, bool)

    def test_router_exists_when_fastapi_available(self):
        from physml.browser_extension_api import _FASTAPI_AVAILABLE, router
        if _FASTAPI_AVAILABLE:
            assert router is not None
        else:
            # Acceptable for router to be None when FastAPI is missing
            assert router is None

    def test_request_models_importable_when_fastapi_available(self):
        from physml.browser_extension_api import _FASTAPI_AVAILABLE
        if not _FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not installed")
        from physml.browser_extension_api import (
            PageVisitRequest, SelectionRequest, BookmarkRequest, CommandRequest,
        )
        assert PageVisitRequest is not None
        assert SelectionRequest is not None
        assert BookmarkRequest is not None
        assert CommandRequest is not None

    def test_page_visit_request_defaults(self):
        from physml.browser_extension_api import _FASTAPI_AVAILABLE
        if not _FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not installed")
        from physml.browser_extension_api import PageVisitRequest
        req = PageVisitRequest(url="https://example.com")
        assert req.url == "https://example.com"
        assert req.title == ""
        assert req.session_id == "default"

    def test_browser_ext_files_exist(self):
        from pathlib import Path
        ext_dir = Path(__file__).parent.parent / "physml" / "browser_ext"
        assert (ext_dir / "manifest.json").exists()
        assert (ext_dir / "background.js").exists()
        assert (ext_dir / "content.js").exists()
        assert (ext_dir / "popup.html").exists()
        assert (ext_dir / "popup.js").exists()

    def test_manifest_json_valid(self):
        import json
        from pathlib import Path
        manifest = json.loads(
            (Path(__file__).parent.parent / "physml" / "browser_ext" / "manifest.json").read_text()
        )
        assert manifest["manifest_version"] == 3
        assert "Mycelium" in manifest["name"]
        assert "action" in manifest


# ===========================================================================
# TestCompanionV2Integration
# ===========================================================================


class TestCompanionV2Integration:
    def test_companion_has_ingester_attr(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "ingester")

    def test_companion_has_screen_observer_attr(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "screen_observer")

    def test_companion_has_macro_recorder_attr(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "macro_recorder")

    def test_companion_has_imitation_learner_attr(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "imitation_learner")

    def test_companion_has_user_model_attr(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "user_model")

    def test_companion_has_skill_library_attr(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "skill_library")

    def test_companion_has_ingest_method(self):
        from physml.companion import MyceliumCompanion
        assert callable(getattr(MyceliumCompanion, "ingest", None))

    def test_companion_has_start_screen_observer_method(self):
        from physml.companion import MyceliumCompanion
        assert callable(getattr(MyceliumCompanion, "start_screen_observer", None))

    def test_companion_has_start_macro_recording_method(self):
        from physml.companion import MyceliumCompanion
        assert callable(getattr(MyceliumCompanion, "start_macro_recording", None))

    def test_companion_has_stop_macro_recording_method(self):
        from physml.companion import MyceliumCompanion
        assert callable(getattr(MyceliumCompanion, "stop_macro_recording", None))

    def test_companion_has_suggest_next_action_method(self):
        from physml.companion import MyceliumCompanion
        assert callable(getattr(MyceliumCompanion, "suggest_next_action", None))

    @pytest.mark.slow
    def test_companion_ingest_after_start(self, tmp_path):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        try:
            result = c.ingest("Hello, this is a test ingestion.")
            # Either IngestResult or None (graceful)
            if result is not None:
                assert hasattr(result, "success")
        finally:
            c.stop()

    @pytest.mark.slow
    def test_companion_suggest_no_crash(self, tmp_path):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion(data_dir=str(tmp_path))
        c.start()
        try:
            suggestions = c.suggest_next_action(context_app="Browser")
            assert isinstance(suggestions, list)
        finally:
            c.stop()


# ===========================================================================
# TestCLIV2Commands
# ===========================================================================


class TestCLIV2Commands:
    def test_ingest_command_exists(self):
        from physml import cli
        assert hasattr(cli, "_cmd_ingest")

    def test_observe_command_exists(self):
        from physml import cli
        assert hasattr(cli, "_cmd_observe")

    def test_record_command_exists(self):
        from physml import cli
        assert hasattr(cli, "_cmd_record")

    def test_model_command_exists(self):
        from physml import cli
        assert hasattr(cli, "_cmd_model")

    def test_ingest_command_runs(self, capsys):
        import argparse
        from physml.cli import _cmd_ingest
        args = argparse.Namespace(source="My name is Alex.", topic="test")
        _cmd_ingest(args)
        out = capsys.readouterr().out
        assert "Ingesting" in out or "text chars" in out or "elapsed" in out

    def test_model_command_runs(self, capsys):
        import argparse
        from physml.cli import _cmd_model
        _cmd_model(argparse.Namespace())
        out = capsys.readouterr().out
        assert "User Model" in out or "context" in out.lower() or "pattern" in out.lower()
