"""Tests for Stage 106 (NaturalLanguageRouter) and Stage 107 (LocalTaskExecutor)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Stage 106 — NaturalLanguageRouter
# ---------------------------------------------------------------------------

class TestNaturalLanguageRouter:
    def _router(self):
        from physml.nl_router import NaturalLanguageRouter, Intent
        router = NaturalLanguageRouter(min_confidence=0.1)
        router.register(Intent("predict", ["predict this", "run inference on", "what is the result", "classify"]))
        router.register(Intent("train",   ["train on", "learn from", "fit the model", "teach"]))
        router.register(Intent("report",  ["show me stats", "give a report", "how is the model doing", "status"]))
        router.register(Intent("save",    ["save the model", "persist", "write to disk"]))
        return router

    def test_registers_intents(self):
        from physml.nl_router import NaturalLanguageRouter, Intent
        router = NaturalLanguageRouter()
        router.register(Intent("foo", ["do foo", "make foo"]))
        assert "foo" in router.list_intents()

    def test_routes_known_intent(self):
        router = self._router()
        result = router.route("please train on this new data")
        assert result.intent == "train"
        assert result.confidence > 0.0

    def test_routes_predict_intent(self):
        router = self._router()
        result = router.route("classify the input features")
        assert result.intent == "predict"

    def test_unknown_intent_low_confidence(self):
        router = self._router()
        result = router.route("buy me a coffee please xyzzy")
        # Either "unknown" or low confidence — not a hard intent match
        assert result.intent == "unknown" or result.confidence < 0.5

    def test_entity_extraction_numbers(self):
        from physml.nl_router import _extract_entities
        entities = _extract_entities("predict for values 1.5 3.2 -0.7")
        assert "numbers" in entities
        assert 1.5 in entities["numbers"]
        assert -0.7 in entities["numbers"]

    def test_entity_extraction_paths(self):
        from physml.nl_router import _extract_entities
        entities = _extract_entities("train on data.csv and save to model.pkl")
        assert "paths" in entities
        path_names = entities["paths"]
        assert any("data.csv" in p for p in path_names)

    def test_entity_extraction_quoted(self):
        from physml.nl_router import _extract_entities
        entities = _extract_entities('set target to "price" in the dataset')
        assert "quoted" in entities
        assert "price" in entities["quoted"]

    def test_entity_extraction_kv(self):
        from physml.nl_router import _extract_entities
        entities = _extract_entities("fit with backend=neural n_cycles=10")
        assert "kv" in entities
        assert entities["kv"].get("backend") == "neural"

    def test_route_extracts_entities(self):
        router = self._router()
        result = router.route("train on data.csv with 50 iterations")
        assert result.raw_text == "train on data.csv with 50 iterations"
        assert "numbers" in result.entities

    def test_empty_router_returns_unknown(self):
        from physml.nl_router import NaturalLanguageRouter
        router = NaturalLanguageRouter()
        result = router.route("hello")
        assert result.intent == "unknown"
        assert result.confidence == 0.0

    def test_register_many(self):
        from physml.nl_router import NaturalLanguageRouter, Intent
        router = NaturalLanguageRouter()
        router.register_many([
            Intent("a", ["do a"]),
            Intent("b", ["do b"]),
        ])
        assert len(router.list_intents()) == 2

    def test_repr(self):
        router = self._router()
        assert "NaturalLanguageRouter" in repr(router)


# ---------------------------------------------------------------------------
# Stage 107 — LocalTaskExecutor
# ---------------------------------------------------------------------------

class TestLocalTaskExecutor:
    def _executor(self, **kwargs):
        from physml.local_executor import LocalTaskExecutor
        return LocalTaskExecutor(**kwargs)

    def test_default_policy_is_read_only(self):
        from physml.local_executor import LocalTaskExecutor
        ex = LocalTaskExecutor()
        assert ex.policy.read_only is True

    def test_list_dir_works(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.csv").write_text("x,y")
        ex = self._executor(base_dir=tmp_path)
        result = ex.list_dir(".")
        assert result.success
        assert "a.txt" in result.output
        assert "b.csv" in result.output

    def test_read_file_works(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello mycelium")
        ex = self._executor(base_dir=tmp_path)
        result = ex.read_file("test.txt")
        assert result.success
        assert "hello mycelium" in result.output

    def test_read_file_missing(self, tmp_path):
        ex = self._executor(base_dir=tmp_path)
        result = ex.read_file("nonexistent.txt")
        assert not result.success
        assert result.error is not None

    def test_file_exists_true(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("x")
        ex = self._executor(base_dir=tmp_path)
        result = ex.file_exists("x.txt")
        assert result.success
        assert result.output is True

    def test_file_exists_false(self, tmp_path):
        ex = self._executor(base_dir=tmp_path)
        result = ex.file_exists("ghost.txt")
        assert result.success
        assert result.output is False

    def test_file_info_returns_metadata(self, tmp_path):
        f = tmp_path / "info.txt"
        f.write_text("test data")
        ex = self._executor(base_dir=tmp_path)
        result = ex.file_info("info.txt")
        assert result.success
        assert result.output["exists"] is True
        assert result.output["is_file"] is True
        assert result.output["size_bytes"] > 0

    def test_write_blocked_by_read_only(self, tmp_path):
        ex = self._executor(base_dir=tmp_path)
        result = ex.write_file("out.txt", "content")
        assert not result.success
        assert "read_only" in (result.error or "").lower()

    def test_write_works_when_allowed(self, tmp_path):
        from physml.local_executor import ExecutionPolicy, LocalTaskExecutor
        policy = ExecutionPolicy(read_only=False)
        ex = LocalTaskExecutor(policy=policy, base_dir=tmp_path)
        result = ex.write_file("out.txt", "mycelium rocks")
        assert result.success
        assert (tmp_path / "out.txt").read_text() == "mycelium rocks"

    def test_copy_file_works(self, tmp_path):
        from physml.local_executor import ExecutionPolicy, LocalTaskExecutor
        src = tmp_path / "src.txt"
        src.write_text("original")
        policy = ExecutionPolicy(read_only=False)
        ex = LocalTaskExecutor(policy=policy, base_dir=tmp_path)
        result = ex.copy_file("src.txt", "dst.txt")
        assert result.success
        assert (tmp_path / "dst.txt").read_text() == "original"

    def test_shell_blocked_by_read_only(self):
        ex = self._executor()
        result = ex.run_shell("echo hello")
        assert not result.success
        assert "read_only" in (result.error or "").lower()

    def test_shell_blocked_by_safe_policy(self, tmp_path):
        from physml.local_executor import ExecutionPolicy, LocalTaskExecutor
        policy = ExecutionPolicy(read_only=False, safe_shell_only=True)
        ex = LocalTaskExecutor(policy=policy, base_dir=tmp_path)
        result = ex.run_shell("sudo rm -rf /")
        assert not result.success
        assert "blocked" in (result.error or "").lower()

    def test_shell_runs_safe_command(self, tmp_path):
        from physml.local_executor import ExecutionPolicy, LocalTaskExecutor
        policy = ExecutionPolicy(read_only=False, safe_shell_only=True)
        ex = LocalTaskExecutor(policy=policy, base_dir=tmp_path)
        result = ex.run_shell("echo hello_mycelium")
        assert result.success
        assert "hello_mycelium" in result.output["stdout"]

    def test_dry_run_does_not_write(self, tmp_path):
        from physml.local_executor import ExecutionPolicy, LocalTaskExecutor
        policy = ExecutionPolicy(read_only=False, dry_run=True)
        ex = LocalTaskExecutor(policy=policy, base_dir=tmp_path)
        result = ex.write_file("dry.txt", "should not exist")
        assert result.success
        assert result.dry_run is True
        assert not (tmp_path / "dry.txt").exists()

    def test_is_command_available(self):
        ex = self._executor()
        result = ex.is_command_available("python3")
        assert result.success
        assert isinstance(result.output, bool)

    def test_list_processes_returns_list(self):
        ex = self._executor()
        result = ex.list_processes()
        assert result.success
        assert isinstance(result.output, list)

    def test_list_processes_with_filter(self):
        ex = self._executor()
        result = ex.list_processes(name_filter="python")
        assert result.success
        # Should find at least the current python process
        assert any("python" in p.get("name", "").lower() for p in result.output)

    def test_allowed_dirs_blocks_outside_path(self, tmp_path):
        from physml.local_executor import ExecutionPolicy, LocalTaskExecutor
        policy = ExecutionPolicy(
            read_only=False,
            allowed_dirs=[str(tmp_path / "safe")],
        )
        (tmp_path / "safe").mkdir()
        ex = LocalTaskExecutor(policy=policy, base_dir=tmp_path)
        result = ex.read_file(str(tmp_path / "outside.txt"))
        assert not result.success

    def test_repr(self):
        ex = self._executor()
        assert "LocalTaskExecutor" in repr(ex)
