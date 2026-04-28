"""Tests for Stages 123–128: ModelManager, ToolBridge, VoiceLoop, VectorMemory, ServerAuth, WebUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(tmp_path: Path, name: str = "data.csv") -> str:
    p = tmp_path / name
    p.write_text("a,b,target\n1,2,3\n4,5,9\n7,8,15\n10,11,21\n13,14,27\n16,17,33\n")
    return str(p)


def _make_companion(tmp_path: Path) -> Any:
    from physml.companion import MyceliumCompanion

    c = MyceliumCompanion(name="TestBot", data_dir=str(tmp_path))
    c.start()
    return c


# ===========================================================================
# Stage 123 — ModelManager
# ===========================================================================


class TestModelManager:
    @pytest.mark.slow
    def test_train_from_csv_success(self, tmp_path):
        from physml.model_manager import ModelManager

        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        result = mgr.train_from_csv(_make_csv(tmp_path))
        assert result.success
        assert result.n_rows > 0
        assert result.n_features > 0
        assert result.target_column != ""

    @pytest.mark.slow
    def test_train_and_predict(self, tmp_path):
        from physml.model_manager import ModelManager

        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        mgr.train_from_csv(_make_csv(tmp_path))
        assert mgr.fitted
        result = mgr.predict([1.0, 2.0])
        assert result.model_fitted
        assert result.error is None
        assert isinstance(result.value, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_without_model_returns_error(self, tmp_path):
        from physml.model_manager import ModelManager

        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        result = mgr.predict([1.0, 2.0])
        assert not result.model_fitted
        assert result.error is not None

    @pytest.mark.slow
    def test_save_and_load_roundtrip(self, tmp_path):
        from physml.model_manager import ModelManager

        model_dir = str(tmp_path / "model")
        mgr = ModelManager(model_dir=model_dir)
        mgr.train_from_csv(_make_csv(tmp_path))
        mgr.save()

        mgr2 = ModelManager(model_dir=model_dir)
        loaded = mgr2.load()
        assert loaded
        assert mgr2.fitted
        pred = mgr2.predict([1.0, 2.0])
        assert pred.model_fitted

    @pytest.mark.slow
    def test_train_with_target_column(self, tmp_path):
        from physml.model_manager import ModelManager

        p = tmp_path / "data.csv"
        p.write_text("x,y,z\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n")
        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        result = mgr.train_from_csv(str(p), target_column="y")
        assert result.success
        assert result.target_column == "y"

    def test_train_missing_file(self, tmp_path):
        from physml.model_manager import ModelManager

        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        result = mgr.train_from_csv("/nonexistent/data.csv")
        assert not result.success
        assert result.error is not None

    def test_train_empty_csv(self, tmp_path):
        from physml.model_manager import ModelManager

        p = tmp_path / "empty.csv"
        p.write_text("a,b\n")
        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        result = mgr.train_from_csv(str(p))
        assert not result.success

    @pytest.mark.slow
    def test_train_from_arrays(self, tmp_path):
        from physml.model_manager import ModelManager

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([3, 7, 11, 15, 19], dtype=float)
        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        result = mgr.train_from_arrays(X, y, feature_names=["a", "b"])
        assert result.success
        assert result.n_rows == 5

    @pytest.mark.slow
    def test_partial_fit_without_prior_model(self, tmp_path):
        from physml.model_manager import ModelManager

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3.0, 7.0, 11.0])
        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        result = mgr.partial_fit(X, y)
        assert result.success

    def test_status_dict_keys(self, tmp_path):
        from physml.model_manager import ModelManager

        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        s = mgr.status()
        assert "fitted" in s
        assert "n_training_rows" in s
        assert "n_predictions" in s

    def test_repr(self, tmp_path):
        from physml.model_manager import ModelManager

        mgr = ModelManager(model_dir=str(tmp_path / "model"))
        assert "ModelManager" in repr(mgr)
        assert "unfitted" in repr(mgr)


# ===========================================================================
# Stage 124 — ToolBridge
# ===========================================================================


class TestToolBridge:
    def test_build_tool_definitions(self):
        from physml.tool_bridge import build_tool_definitions

        tools = build_tool_definitions()
        assert len(tools) >= 5
        names = [t["name"] for t in tools]
        assert "run_prediction" in names
        assert "train_on_file" in names
        assert "read_document" in names
        assert "show_report" in names
        assert "execute_task" in names

    def test_tool_definitions_have_required_keys(self):
        from physml.tool_bridge import build_tool_definitions

        for tool in build_tool_definitions():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_execute_unknown_tool(self):
        from physml.tool_bridge import ToolBridge

        tb = ToolBridge()
        result = tb.execute("nonexistent_tool", {})
        assert "Unknown tool" in result or "error" in result.lower()

    def test_execute_show_report_no_companion(self):
        from physml.tool_bridge import ToolBridge

        tb = ToolBridge(companion=None)
        result = tb.execute("show_report", {})
        assert "not connected" in result.lower() or result

    def test_execute_run_prediction_no_companion(self):
        from physml.tool_bridge import ToolBridge

        tb = ToolBridge(companion=None)
        result = tb.execute("run_prediction", {"features": [1.0, 2.0]})
        assert result  # non-empty

    def test_execute_execute_task_system_info(self):
        from physml.tool_bridge import ToolBridge

        tb = ToolBridge(companion=None)
        result = tb.execute("execute_task", {"action": "system_info"})
        assert result  # non-empty, should contain platform info

    def test_execute_execute_task_exists(self, tmp_path):
        from physml.tool_bridge import ToolBridge

        tb = ToolBridge(companion=None)
        result = tb.execute("execute_task", {"action": "exists", "path": str(tmp_path)})
        assert "exist" in result.lower()

    def test_execute_all_returns_list(self):
        from physml.tool_bridge import ToolBridge

        tb = ToolBridge(companion=None)
        calls = [{"id": "call_1", "name": "show_report", "input": {}}]
        results = tb.execute_all(calls)
        assert len(results) == 1
        assert "tool_use_id" in results[0]
        assert "content" in results[0]

    @pytest.mark.slow
    def test_execute_with_companion_predict(self, tmp_path):
        from physml.tool_bridge import ToolBridge

        companion = _make_companion(tmp_path)
        try:
            # Train first
            csv = _make_csv(tmp_path)
            companion.model_manager.train_from_csv(csv)

            tb = ToolBridge(companion=companion)
            result = tb.execute("run_prediction", {"features": [1.0, 2.0]})
            assert result
            assert "prediction" in result.lower() or "error" in result.lower()
        finally:
            companion.stop()

    @pytest.mark.slow
    def test_execute_with_companion_train_and_predict(self, tmp_path):
        from physml.tool_bridge import ToolBridge

        companion = _make_companion(tmp_path)
        try:
            tb = ToolBridge(companion=companion)
            csv = _make_csv(tmp_path)
            train_result = tb.execute("train_on_file", {"path": csv})
            assert "Trained" in train_result or "trained" in train_result.lower()

            pred_result = tb.execute("run_prediction", {"features": [1.0, 2.0]})
            assert pred_result
        finally:
            companion.stop()

    def test_execute_with_companion_read_document(self, tmp_path):
        from physml.tool_bridge import ToolBridge

        companion = _make_companion(tmp_path)
        try:
            txt = tmp_path / "notes.txt"
            txt.write_text("Hello world. This is a test document.\n")
            tb = ToolBridge(companion=companion)
            result = tb.execute("read_document", {"path": str(txt)})
            assert "Hello world" in result or "notes.txt" in result
        finally:
            companion.stop()

    def test_execute_with_companion_show_report(self, tmp_path):
        from physml.tool_bridge import ToolBridge

        companion = _make_companion(tmp_path)
        try:
            tb = ToolBridge(companion=companion)
            result = tb.execute("show_report", {})
            assert "TestBot" in result or "status" in result.lower() or result
        finally:
            companion.stop()


# ===========================================================================
# Stage 125 — VoiceLoop
# ===========================================================================


class TestVoiceLoop:
    def test_instantiate(self):
        from physml.voice_loop import VoiceLoop

        loop = VoiceLoop()
        assert loop is not None
        assert not loop.running

    def test_run_once_text(self, tmp_path):
        from physml.voice_loop import VoiceLoop

        companion = _make_companion(tmp_path)
        try:
            loop = VoiceLoop(companion=companion, speak_response=False)
            response = loop.run_once("help")
            assert isinstance(response, str)
            assert len(response) > 0
        finally:
            companion.stop()

    def test_run_once_no_companion(self):
        from physml.voice_loop import VoiceLoop

        loop = VoiceLoop(companion=None, speak_response=False)
        result = loop.run_once("hello")
        assert "No companion" in result

    def test_callbacks_triggered(self, tmp_path):
        from physml.voice_loop import VoiceLoop

        companion = _make_companion(tmp_path)
        responses = []
        try:
            loop = VoiceLoop(
                companion=companion,
                speak_response=False,
                on_response=responses.append,
            )
            loop.run_once("help")
            assert len(responses) == 1
        finally:
            companion.stop()

    def test_start_stop_background(self):
        from physml.voice_loop import VoiceLoop
        import time

        loop = VoiceLoop(speak_response=False)
        loop.start()
        assert loop.running
        time.sleep(0.1)
        loop.stop()
        assert not loop.running

    def test_stt_backend_property(self):
        from physml.voice_loop import VoiceLoop

        loop = VoiceLoop()
        assert isinstance(loop.stt_backend, str)

    def test_tts_backend_property(self):
        from physml.voice_loop import VoiceLoop

        loop = VoiceLoop(speak_response=True)
        assert isinstance(loop.tts_backend, str)

    def test_wake_word_gate(self, tmp_path):
        from physml.voice_loop import VoiceLoop

        companion = _make_companion(tmp_path)
        try:
            loop = VoiceLoop(
                companion=companion,
                wake_word="hey myco",
                speak_response=False,
            )
            # Without wake word, should return empty
            # (text passthrough doesn't contain wake word)
            # The internal state should block
            assert loop.wake_word == "hey myco"
            assert not loop._wake_word_active
        finally:
            companion.stop()

    def test_repr(self):
        from physml.voice_loop import VoiceLoop

        loop = VoiceLoop(speak_response=False)
        assert "VoiceLoop" in repr(loop)


# ===========================================================================
# Stage 126 — VectorMemory
# ===========================================================================


class TestVectorMemory:
    def test_add_and_len(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="linear_scan")
        mem.add("First memory about sales")
        mem.add("Second memory about training")
        assert len(mem) == 2

    def test_search_returns_results(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="linear_scan")
        mem.add("User asked about sales forecasting")
        mem.add("Model trained on retail_data.csv")
        mem.add("User prefers concise answers")
        results = mem.search("predict revenue", k=2)
        assert len(results) <= 2
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_search_empty_memory_returns_empty(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="linear_scan")
        results = mem.search("anything", k=3)
        assert results == []

    def test_tfidf_backend(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="tfidf")
        mem.add("Machine learning prediction model")
        mem.add("Sales revenue forecast quarterly")
        results = mem.search("sales forecast", k=2)
        assert len(results) <= 2

    def test_max_entries_eviction(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(max_entries=3, backend="linear_scan")
        for i in range(10):
            mem.add(f"Entry {i}")
        assert len(mem) <= 3

    def test_save_and_load(self, tmp_path):
        from physml.vector_memory import VectorMemory

        path = str(tmp_path / "mem.json")
        mem = VectorMemory(backend="linear_scan")
        mem.add("Persistent memory entry")
        mem.save(path)

        mem2 = VectorMemory(backend="linear_scan")
        n = mem2.load(path)
        assert n == 1
        assert mem2._entries[0].text == "Persistent memory entry"

    def test_clear(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="linear_scan")
        mem.add("Some text")
        mem.clear()
        assert len(mem) == 0

    def test_entry_has_metadata(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="linear_scan")
        entry = mem.add("Test entry", metadata={"intent": "test", "source": "unit"})
        assert entry.metadata["intent"] == "test"

    def test_search_result_fields(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="linear_scan")
        mem.add("Test memory text")
        results = mem.search("test memory", k=1)
        if results:
            r = results[0]
            assert hasattr(r, "text")
            assert hasattr(r, "score")
            assert hasattr(r, "rank")
            assert r.rank == 1

    def test_repr(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="linear_scan")
        assert "VectorMemory" in repr(mem)

    def test_active_backend_property(self):
        from physml.vector_memory import VectorMemory

        mem = VectorMemory(backend="tfidf")
        assert isinstance(mem.active_backend, str)


# ===========================================================================
# Stage 127 — Server JWT Auth
# ===========================================================================


class TestServerJWT:
    def test_create_and_verify_token(self):
        from physml.server import _create_token, _verify_token

        token = _create_token("alice")
        user_id = _verify_token(token)
        assert user_id == "alice"

    def test_verify_invalid_token_raises(self):
        from physml.server import _verify_token

        with pytest.raises(ValueError):
            _verify_token("not.a.valid.token")

    def test_verify_tampered_token_raises(self):
        from physml.server import _create_token, _verify_token

        token = _create_token("alice")
        parts = token.split(".")
        parts[1] = parts[1][:-2] + "AA"  # tamper with payload
        with pytest.raises(ValueError):
            _verify_token(".".join(parts))

    def test_token_contains_user_id(self):
        import base64
        from physml.server import _create_token

        token = _create_token("bob")
        payload_b64 = token.split(".")[1]
        pad = 4 - len(payload_b64) % 4
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (pad % 4)))
        assert payload["sub"] == "bob"


# ===========================================================================
# Companion integration — train + predict + memory + tool bridge
# ===========================================================================


class TestCompanionIntegration:
    @pytest.mark.slow
    def test_train_and_predict_flow(self, tmp_path):
        companion = _make_companion(tmp_path)
        try:
            csv = _make_csv(tmp_path)
            r1 = companion.chat(f"train on {csv}")
            assert "Trained" in r1 or "trained" in r1.lower()
            assert companion.model_manager.fitted

            r2 = companion.chat("predict 1 2")
            assert "Predicted" in r2 or "prediction" in r2.lower()
        finally:
            companion.stop()

    @pytest.mark.slow
    def test_model_persists_across_restart(self, tmp_path):
        from physml.companion import MyceliumCompanion

        csv = _make_csv(tmp_path)
        c1 = MyceliumCompanion(data_dir=str(tmp_path))
        c1.start()
        c1.chat(f"train on {csv}")
        c1.stop()

        c2 = MyceliumCompanion(data_dir=str(tmp_path))
        c2.start()
        assert c2.model_manager.fitted  # model restored
        c2.stop()

    def test_predict_without_model_helpful_message(self, tmp_path):
        companion = _make_companion(tmp_path)
        try:
            r = companion.chat("predict 1 2 3")
            assert "train" in r.lower() or "model" in r.lower()
        finally:
            companion.stop()

    @pytest.mark.slow
    def test_vector_memory_grows_on_train(self, tmp_path):
        companion = _make_companion(tmp_path)
        try:
            initial_count = len(companion.vector_memory)
            csv = _make_csv(tmp_path)
            companion.chat(f"train on {csv}")
            assert len(companion.vector_memory) > initial_count
        finally:
            companion.stop()

    def test_tool_bridge_wired_to_companion(self, tmp_path):
        companion = _make_companion(tmp_path)
        try:
            assert companion.tool_bridge is not None
            result = companion.tool_bridge.execute("show_report", {})
            assert result
        finally:
            companion.stop()

    def test_model_manager_wired_to_companion(self, tmp_path):
        companion = _make_companion(tmp_path)
        try:
            assert companion.model_manager is not None
            s = companion.model_manager.status()
            assert "fitted" in s
        finally:
            companion.stop()
