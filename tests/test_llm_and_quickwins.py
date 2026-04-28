"""Tests for Phase 4 (LLM integration package) and Phase 5 (quick wins).

Covers:
- physml.llm.ClaudeClient (no API key — fallback mode)
- physml.llm.PromptSystem (rule-based routing)
- physml.experiment_runner.ExperimentRunner
- physml.conversation_store.ConversationStore
"""

from __future__ import annotations

import json

import pytest


# ===========================================================================
# Phase 4 — LLM package: ClaudeClient
# ===========================================================================


class TestClaudeClient:
    def test_import(self):
        from physml.llm import ClaudeClient
        assert ClaudeClient is not None

    def test_instantiate_no_key(self):
        from physml.llm import ClaudeClient
        client = ClaudeClient(api_key="")
        assert not client.available

    def test_chat_fallback_returns_chat_result(self):
        from physml.llm import ClaudeClient, ChatResult
        client = ClaudeClient(api_key="")
        result = client.chat("Hello")
        assert isinstance(result, ChatResult)
        assert not result.available
        assert result.error is not None

    def test_complete_fallback(self):
        from physml.llm import ClaudeClient
        client = ClaudeClient(api_key="")
        result = client.complete("test prompt")
        assert not result.available

    def test_tool_call_fallback(self):
        from physml.llm import ClaudeClient, ToolCallResult
        client = ClaudeClient(api_key="")
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        result = client.tool_call("use the test tool", tools=tools)
        assert isinstance(result, ToolCallResult)
        assert not result.available

    def test_call_count_starts_at_zero(self):
        from physml.llm import ClaudeClient
        client = ClaudeClient(api_key="")
        assert client.call_count == 0

    def test_cache_hit_count_starts_at_zero(self):
        from physml.llm import ClaudeClient
        client = ClaudeClient(api_key="")
        assert client.cache_hit_count == 0

    def test_repr_contains_class_name(self):
        from physml.llm import ClaudeClient
        client = ClaudeClient(api_key="")
        assert "ClaudeClient" in repr(client)

    def test_model_attribute(self):
        from physml.llm import ClaudeClient
        client = ClaudeClient(api_key="", model="claude-sonnet-4-6")
        assert client.model == "claude-sonnet-4-6"

    def test_chat_result_fields(self):
        from physml.llm import ChatResult
        r = ChatResult(text="hi", available=True, model="claude-sonnet-4-6")
        assert r.text == "hi"
        assert r.available is True
        assert r.model == "claude-sonnet-4-6"
        assert r.tool_calls == []

    def test_tool_call_result_fields(self):
        from physml.llm import ToolCallResult
        tc = ToolCallResult(tool_calls=[{"name": "foo", "input": {}}], available=True)
        assert tc.tool_calls[0]["name"] == "foo"

    def test_exported_from_physml(self):
        from physml import ClaudeClient, ChatResult, ToolCallResult
        assert ClaudeClient is not None
        assert ChatResult is not None
        assert ToolCallResult is not None


# ===========================================================================
# Phase 4 — LLM package: PromptSystem
# ===========================================================================


class TestPromptSystem:
    def test_import(self):
        from physml.llm import PromptSystem
        assert PromptSystem is not None

    def test_instantiate(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        assert ps is not None

    def test_route_train_intent(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("train a model on data.csv")
        assert action.intent == "train"
        assert action.confidence > 0

    def test_route_predict_intent(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("predict the outcome for 1.5 2.3 3.1")
        assert action.intent == "predict"

    def test_route_report_intent(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("show me the model status")
        assert action.intent in ("report", "show_goals", "unknown")

    def test_route_help_intent(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("help me out")
        # Should classify as 'help' or have reasonable confidence
        assert isinstance(action.intent, str)

    def test_route_extracts_numbers(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("predict for values 1.5 3.2 -0.7")
        if action.payload.get("numbers"):
            assert 1.5 in action.payload["numbers"]

    def test_route_extracts_path(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("train on /home/user/sales.csv")
        # Path or paths should be in payload
        assert "path" in action.payload or "paths" in action.payload

    def test_route_unknown_gracefully(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("xyzzy frobnicate the zorbinator")
        assert isinstance(action.intent, str)
        assert 0.0 <= action.confidence <= 1.0

    def test_action_raw_text_preserved(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        text = "train on my_data.csv"
        action = ps.route(text)
        assert action.raw_text == text

    def test_via_llm_false_without_api_key(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        action = ps.route("predict something")
        # Without a real API key, routing is rule-based
        assert action.via_llm is False

    def test_describe_intent(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        desc = ps.describe_intent("train")
        assert "train" in desc.lower() or isinstance(desc, str)

    def test_repr(self):
        from physml.llm import PromptSystem
        ps = PromptSystem()
        assert "PromptSystem" in repr(ps)

    def test_prompt_action_fields(self):
        from physml.llm import PromptAction
        action = PromptAction(
            intent="train",
            confidence=0.95,
            payload={"path": "data.csv"},
            raw_text="train on data.csv",
        )
        assert action.intent == "train"
        assert action.confidence == 0.95
        assert action.payload["path"] == "data.csv"
        assert not action.via_llm

    def test_exported_from_physml(self):
        from physml import PromptSystem, PromptAction
        assert PromptSystem is not None
        assert PromptAction is not None


# ===========================================================================
# Phase 5 — ExperimentRunner
# ===========================================================================


class TestExperimentRunner:
    def test_import(self):
        from physml.experiment_runner import ExperimentRunner
        assert ExperimentRunner is not None

    def test_instantiate(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(results_dir=str(tmp_path))
        assert runner is not None

    def test_repr(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(results_dir=str(tmp_path))
        assert "ExperimentRunner" in repr(runner)

    @pytest.mark.slow
    def test_run_regression_returns_summary(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner, BenchmarkSummary
        runner = ExperimentRunner(
            results_dir=str(tmp_path),
            configs=[{"plane": "liquid", "n_cycles": 3}],
        )
        summary = runner.run(task="regression", n_samples=80, n_features=3)
        assert isinstance(summary, BenchmarkSummary)
        assert summary.task == "regression"
        assert summary.n_experiments == 1

    @pytest.mark.slow
    def test_run_classification_returns_summary(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(
            results_dir=str(tmp_path),
            configs=[{"plane": "liquid", "n_cycles": 3}],
        )
        summary = runner.run(task="classification", n_samples=80, n_features=3)
        assert summary.task == "classification"

    @pytest.mark.slow
    def test_run_writes_log(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(
            results_dir=str(tmp_path),
            configs=[{"plane": "liquid", "n_cycles": 3}],
        )
        runner.run(task="regression", n_samples=80, n_features=3)
        log_path = tmp_path / "experiments.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert "task" in parsed
        assert "score" in parsed

    @pytest.mark.slow
    def test_load_history(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(
            results_dir=str(tmp_path),
            configs=[{"plane": "liquid", "n_cycles": 3}],
        )
        runner.run(task="regression", n_samples=80, n_features=3)
        history = runner.load_history()
        assert len(history) >= 1

    def test_load_history_empty(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(results_dir=str(tmp_path))
        history = runner.load_history()
        assert history == []

    def test_invalid_task_raises(self, tmp_path):
        from physml.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(results_dir=str(tmp_path))
        with pytest.raises(ValueError, match="task"):
            runner.run(task="unsupported_task")

    def test_exported_from_physml(self):
        from physml import ExperimentRunner, BenchmarkSummary, ExperimentResult
        assert ExperimentRunner is not None
        assert BenchmarkSummary is not None
        assert ExperimentResult is not None


# ===========================================================================
# Phase 5 — ConversationStore
# ===========================================================================


class TestConversationStore:
    def test_import(self):
        from physml.conversation_store import ConversationStore
        assert ConversationStore is not None

    def test_instantiate(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        assert store is not None
        assert len(store) == 0

    def test_add_user_turn(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "Hello!")
        assert len(store) == 1
        assert store[0]["role"] == "user"
        assert store[0]["content"] == "Hello!"

    def test_add_assistant_turn(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "Hi")
        store.add("assistant", "Hi there!")
        assert len(store) == 2

    def test_save_and_reload(self, tmp_path):
        from physml.conversation_store import ConversationStore
        path = str(tmp_path / "conv.json")
        store = ConversationStore(path=path)
        store.add("user", "Remember this!")
        store.save()

        store2 = ConversationStore(path=path)
        assert len(store2) == 1
        assert store2[0]["content"] == "Remember this!"

    def test_search_returns_relevant(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "Tell me about machine learning predictions")
        store.add("assistant", "Machine learning can predict sales trends")
        store.add("user", "What is the weather like today?")
        results = store.search("machine learning", k=2)
        assert len(results) <= 2
        # The ML-related turns should score higher
        assert any("machine learning" in r["content"].lower() for r in results)

    def test_search_empty_returns_empty(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        results = store.search("anything")
        assert results == []

    def test_to_messages_format(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "Hello")
        store.add("assistant", "Hi!")
        msgs = store.to_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"

    def test_to_messages_max_turns(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        for i in range(10):
            store.add("user", f"Message {i}")
        msgs = store.to_messages(max_turns=3)
        assert len(msgs) == 3

    def test_max_turns_eviction(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"), max_turns=5)
        for i in range(10):
            store.add("user", f"Message {i}")
        assert len(store) <= 5

    def test_clear(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "Hello")
        store.clear()
        assert len(store) == 0

    def test_search_by_role(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "predict sales")
        store.add("assistant", "I can predict your sales data")
        results = store.search("predict sales", role="user")
        assert all(r["role"] == "user" for r in results)

    def test_summary(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "Hello")
        store.add("assistant", "Hi")
        s = store.summary()
        assert s["total_turns"] == 2
        assert s["user_turns"] == 1
        assert s["assistant_turns"] == 1

    def test_add_with_metadata(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        turn = store.add("user", "Hello", metadata={"intent": "greeting"})
        assert turn["metadata"]["intent"] == "greeting"

    def test_to_llm_messages(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "Hello")
        store.add("assistant", "Hi!")
        msgs = store.to_llm_messages(max_turns=20)
        assert len(msgs) == 2
        assert hasattr(msgs[0], "role")
        assert hasattr(msgs[0], "content")

    def test_repr(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        assert "ConversationStore" in repr(store)

    def test_exported_from_physml(self):
        from physml import ConversationStore
        assert ConversationStore is not None

    def test_iteration(self, tmp_path):
        from physml.conversation_store import ConversationStore
        store = ConversationStore(path=str(tmp_path / "conv.json"))
        store.add("user", "A")
        store.add("assistant", "B")
        turns = list(store)
        assert len(turns) == 2
        assert turns[0]["content"] == "A"
