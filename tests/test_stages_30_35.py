"""Tests for Stages 30–35.

Stage 30 — Featurizer
Stage 31 — ToolRegistry + AutonomousLoop
Stage 32 — GoalPlanner + SubTask
Stage 33 — EpisodicMemory
Stage 34 — pretrain_neural_engine / pretrain_mycelium
Stage 35 — ParallelDataStream
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_agent(n: int = 40, d: int = 8):
    """Return a fitted MyceliumAgent on small synthetic data."""
    from physml.mycelium_agent import MyceliumAgent

    X = RNG.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    agent = MyceliumAgent(calibrate=False)
    agent.fit(X, y)
    return agent, d


def _make_featurizer(output_dim: int = 16):
    """Return a fitted text Featurizer."""
    from physml.featurizer import Featurizer

    texts = ["hello world", "foo bar baz", "physics engine", "machine learning"]
    f = Featurizer(output_dim=output_dim)
    f.fit(texts)
    return f


# ===========================================================================
# Stage 30 — Featurizer
# ===========================================================================


class TestStage30Featurizer:
    def test_text_fit_transform_shape(self):
        from physml.featurizer import Featurizer

        texts = ["hello world", "foo bar", "physics ml", "autonomous agent", "data science"]
        f = Featurizer(output_dim=16)
        X = f.fit_transform(texts)
        assert X.shape == (5, 16), f"Expected (5, 16), got {X.shape}"
        assert X.dtype == np.float32

    def test_text_transform_separate_shape(self):
        from physml.featurizer import Featurizer

        train = ["hello world", "foo bar baz"] * 5
        test = ["new string", "another one"]
        f = Featurizer(output_dim=8)
        f.fit(train)
        X = f.transform(test)
        assert X.shape == (2, 8)
        assert X.dtype == np.float32

    def test_numeric_fit_transform_shape(self):
        from physml.featurizer import Featurizer

        X_in = RNG.standard_normal((30, 20)).tolist()
        f = Featurizer(output_dim=10)
        X_out = f.fit_transform(X_in)
        assert X_out.shape == (30, 10)
        assert X_out.dtype == np.float32

    def test_numeric_passthrough_small_dim(self):
        """When n_features <= output_dim, output should still be output_dim."""
        from physml.featurizer import Featurizer

        X_in = RNG.standard_normal((20, 4)).tolist()
        f = Featurizer(output_dim=10)
        X_out = f.fit_transform(X_in)
        assert X_out.shape == (20, 10)

    def test_dict_fit_transform_shape(self):
        from physml.featurizer import Featurizer

        dicts = [{"a": 1, "b": "hello"}, {"a": 2, "c": "world"}, {"x": 99}] * 5
        f = Featurizer(output_dim=12)
        X_out = f.fit_transform(dicts)
        assert X_out.shape == (15, 12)
        assert X_out.dtype == np.float32

    def test_fit_transform_consistent(self):
        """fit_transform should equal fit().transform()."""
        from physml.featurizer import Featurizer

        texts = ["alpha", "beta", "gamma", "delta"] * 4
        f1 = Featurizer(output_dim=8, hash_features=256)
        X1 = f1.fit_transform(texts)

        f2 = Featurizer(output_dim=8, hash_features=256)
        f2.fit(texts)
        X2 = f2.transform(texts)

        np.testing.assert_allclose(X1, X2, atol=1e-5)

    def test_not_fitted_raises(self):
        from physml.featurizer import Featurizer

        f = Featurizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            f.transform(["test"])


# ===========================================================================
# Stage 31 — ToolRegistry + AutonomousLoop
# ===========================================================================


class TestStage31ToolRegistry:
    def test_register_and_call(self):
        from physml.tools import Tool, ToolRegistry

        reg = ToolRegistry()
        tool = Tool(name="echo", description="Echoes input", fn=lambda x: f"echo:{x}")
        reg.register(tool)
        assert reg.call("echo", "hello") == "echo:hello"

    def test_list_tools(self):
        from physml.tools import Tool, ToolRegistry

        reg = ToolRegistry()
        reg.register(Tool(name="t1", description="Tool one", fn=lambda x: x))
        reg.register(Tool(name="t2", description="Tool two", fn=lambda x: x[::-1]))
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"t1", "t2"}

    def test_call_unknown_raises(self):
        from physml.tools import ToolRegistry

        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.call("nonexistent", "input")

    @pytest.mark.slow
    def test_autonomous_loop_returns_dict(self):
        from physml.featurizer import Featurizer
        from physml.tools import AutonomousLoop, Tool, ToolRegistry

        agent, d = _make_agent(d=16)
        texts = ["compute the sum", "find maximum value", "search results"] * 5
        f = Featurizer(output_dim=16)
        f.fit(texts)

        reg = ToolRegistry()
        reg.register(Tool(name="search", description="search web", fn=lambda x: "result: 42"))

        loop = AutonomousLoop(agent=agent, registry=reg, featurizer=f, max_steps=5)
        result = loop.run("compute the sum of values")

        assert isinstance(result, dict)
        assert "steps" in result
        assert "result" in result
        assert "n_tool_calls" in result

    @pytest.mark.slow
    def test_autonomous_loop_no_tools(self):
        from physml.featurizer import Featurizer
        from physml.tools import AutonomousLoop, ToolRegistry

        agent, d = _make_agent(d=16)
        texts = ["hello world", "machine learning", "data science"] * 5
        f = Featurizer(output_dim=16)
        f.fit(texts)

        reg = ToolRegistry()  # empty
        loop = AutonomousLoop(agent=agent, registry=reg, featurizer=f, max_steps=3)
        result = loop.run("some goal")
        assert isinstance(result, dict)
        assert result["n_tool_calls"] == 0


# ===========================================================================
# Stage 32 — GoalPlanner
# ===========================================================================


class TestStage32GoalPlanner:
    def _make_planner(self, n_subtasks: int = 3):
        from physml.featurizer import Featurizer
        from physml.planner import GoalPlanner

        agent, _ = _make_agent(d=16)
        texts = ["step one", "step two", "step three", "gather data", "process results"] * 4
        f = Featurizer(output_dim=16)
        f.fit(texts)
        return GoalPlanner(featurizer=f, agent=agent, n_subtasks=n_subtasks)

    @pytest.mark.slow
    def test_plan_returns_n_subtasks(self):
        planner = self._make_planner(n_subtasks=3)
        subtasks = planner.plan("gather data and process results then summarise")
        assert len(subtasks) == 3

    @pytest.mark.slow
    def test_plan_subtask_types(self):
        from physml.planner import SubTask

        planner = self._make_planner(n_subtasks=2)
        subtasks = planner.plan("do something useful")
        for st in subtasks:
            assert isinstance(st, SubTask)
            assert isinstance(st.task_id, str)
            assert isinstance(st.description, str)
            assert isinstance(st.feature_vec, np.ndarray)
            assert isinstance(st.depends_on, list)

    @pytest.mark.slow
    def test_plan_linear_dependency_chain(self):
        planner = self._make_planner(n_subtasks=3)
        subtasks = planner.plan("step one then step two then step three")
        # First task has no deps, subsequent tasks depend on previous
        assert subtasks[0].depends_on == []
        assert len(subtasks[1].depends_on) == 1
        assert len(subtasks[2].depends_on) == 1

    @pytest.mark.slow
    def test_execute_returns_dict(self):
        planner = self._make_planner(n_subtasks=3)
        result = planner.execute("gather data and process results then summarise")
        assert isinstance(result, dict)
        assert "plan" in result
        assert "results" in result
        assert "n_steps" in result

    @pytest.mark.slow
    def test_execute_n_steps(self):
        planner = self._make_planner(n_subtasks=3)
        result = planner.execute("alpha and beta and gamma")
        assert result["n_steps"] == 3
        assert len(result["plan"]) == 3


# ===========================================================================
# Stage 33 — EpisodicMemory
# ===========================================================================


class TestStage33EpisodicMemory:
    def test_store_and_len(self):
        from physml.memory import EpisodicMemory

        mem = EpisodicMemory(capacity=100, n_neighbors=3)
        assert len(mem) == 0
        ctx = RNG.standard_normal(8).astype(np.float32)
        mem.store(ctx, "predict", 1.0)
        assert len(mem) == 1

    def test_retrieve_returns_k_nearest(self):
        from physml.memory import EpisodicMemory

        mem = EpisodicMemory(n_neighbors=3)
        rng = np.random.default_rng(42)
        for i in range(10):
            ctx = rng.standard_normal(8).astype(np.float32)
            mem.store(ctx, "predict" if i % 2 == 0 else "ask", float(i) / 10)

        query = rng.standard_normal(8).astype(np.float32)
        neighbors = mem.retrieve(query, k=3)
        assert len(neighbors) == 3
        for nb in neighbors:
            assert "context" in nb
            assert "action" in nb
            assert "outcome" in nb
            assert "similarity" in nb

    def test_retrieve_empty_returns_empty(self):
        from physml.memory import EpisodicMemory

        mem = EpisodicMemory()
        result = mem.retrieve(np.zeros(4), k=3)
        assert result == []

    def test_augment_features_shape(self):
        from physml.memory import EpisodicMemory

        n_neighbors = 3
        mem = EpisodicMemory(n_neighbors=n_neighbors)
        rng = np.random.default_rng(7)
        d = 8
        for i in range(20):
            ctx = rng.standard_normal(d).astype(np.float32)
            mem.store(ctx, "predict" if i % 2 == 0 else "ask", float(i % 5) / 5)

        X = rng.standard_normal((5, d)).astype(np.float32)
        X_aug = mem.augment_features(X)
        expected_cols = d + n_neighbors * 2
        assert X_aug.shape == (5, expected_cols), f"Expected (5, {expected_cols}), got {X_aug.shape}"
        assert X_aug.dtype == np.float32

    def test_augment_features_empty_returns_unchanged(self):
        from physml.memory import EpisodicMemory

        mem = EpisodicMemory()
        X = RNG.standard_normal((3, 6)).astype(np.float32)
        X_out = mem.augment_features(X)
        np.testing.assert_array_equal(X, X_out)

    def test_capacity_eviction(self):
        from physml.memory import EpisodicMemory

        mem = EpisodicMemory(capacity=5)
        for i in range(10):
            ctx = RNG.standard_normal(4).astype(np.float32)
            mem.store(ctx, "act", float(i))
        assert len(mem) == 5

    @pytest.mark.slow
    def test_mycelium_augment_with_memory(self):
        """Test MyceliumAgent.augment_with_memory integration."""
        from physml.memory import EpisodicMemory

        agent, d = _make_agent(d=8)
        n_neighbors = 3
        mem = EpisodicMemory(n_neighbors=n_neighbors)
        for i in range(15):
            ctx = RNG.standard_normal(d).astype(np.float32)
            mem.store(ctx, "predict", float(i % 2))

        X = RNG.standard_normal((4, d)).astype(np.float32)
        X_aug = agent.augment_with_memory(X, mem)
        assert X_aug.shape == (4, d + n_neighbors * 2)


# ===========================================================================
# Stage 34 — Pretraining
# ===========================================================================


class TestStage34Pretrain:
    @pytest.mark.slow
    def test_pretrain_neural_engine_sets_attribute(self):
        from physml.neural_engine import NeuralPhysicsEngine
        from physml.pretrain import pretrain_neural_engine

        engine = NeuralPhysicsEngine()
        X = RNG.standard_normal((50, 8)).astype(np.float32)
        engine = pretrain_neural_engine(engine, X, n_epochs=2, random_state=0)
        assert hasattr(engine, "pretrained_coefs_")
        assert isinstance(engine.pretrained_coefs_, list)
        assert len(engine.pretrained_coefs_) > 0

    @pytest.mark.slow
    def test_pretrain_neural_engine_no_error(self):
        from physml.neural_engine import NeuralPhysicsEngine
        from physml.pretrain import pretrain_neural_engine

        engine = NeuralPhysicsEngine()
        X = RNG.standard_normal((30, 4)).astype(np.float32)
        result = pretrain_neural_engine(engine, X, mask_fraction=0.2, n_epochs=3, batch_size=16)
        assert result is engine

    @pytest.mark.slow
    def test_pretrain_mycelium_sets_pretrained(self):
        from physml.pretrain import pretrain_mycelium

        agent, d = _make_agent(d=8)
        X_unlabelled = RNG.standard_normal((40, d)).astype(np.float32)
        result = pretrain_mycelium(agent, X_unlabelled, n_epochs=2, random_state=1)
        assert result is agent
        assert getattr(agent, "_pretrained", False) is True
        assert hasattr(agent, "_pretrain_coefs_")

    @pytest.mark.slow
    def test_pretrain_mycelium_unfitted(self):
        """pretrain_mycelium should work even before agent.fit()."""
        from physml.mycelium_agent import MyceliumAgent
        from physml.pretrain import pretrain_mycelium

        agent = MyceliumAgent(calibrate=False)
        X_unlabelled = RNG.standard_normal((30, 6)).astype(np.float32)
        result = pretrain_mycelium(agent, X_unlabelled, n_epochs=2)
        assert result is agent
        assert getattr(agent, "_pretrained", False) is True

    @pytest.mark.slow
    def test_use_tool_integration(self):
        """MyceliumAgent.use_tool should call tool and return string."""
        from physml.tools import Tool, ToolRegistry

        agent, _ = _make_agent()
        reg = ToolRegistry()
        reg.register(Tool(name="upper", description="uppercase", fn=str.upper))
        result = agent.use_tool("upper", "hello", reg)
        assert result == "HELLO"


# ===========================================================================
# Stage 35 — ParallelDataStream
# ===========================================================================


class TestStage35ParallelDataStream:
    def _make_chunks(self, n_chunks: int = 4, n: int = 20, d: int = 5):
        rng = np.random.default_rng(99)
        return [
            (rng.standard_normal((n, d)).astype(np.float32), rng.integers(0, 2, n))
            for _ in range(n_chunks)
        ]

    def test_map_returns_correct_count(self):
        from physml.stream_worker import ParallelDataStream

        chunks = self._make_chunks(n_chunks=5)
        stream = ParallelDataStream(chunks, n_workers=2, backend="thread")
        results = stream.map(lambda X, y: X.shape[0], use_parallel=True)
        assert len(results) == 5
        assert all(r == 20 for r in results)

    def test_map_sequential(self):
        from physml.stream_worker import ParallelDataStream

        chunks = self._make_chunks(n_chunks=3)
        stream = ParallelDataStream(chunks, n_workers=1)
        results = stream.map(lambda X, y: int(y.sum()), use_parallel=False)
        assert len(results) == 3

    def test_fit_stream_sequential(self):
        from sklearn.linear_model import SGDClassifier

        from physml.stream_worker import ParallelDataStream

        chunks = self._make_chunks(n_chunks=4, n=30, d=5)
        clf = SGDClassifier(max_iter=5, random_state=0)
        # Need to initialise with classes
        X0, y0 = chunks[0]
        clf.partial_fit(X0, y0, classes=np.array([0, 1]))

        stream = ParallelDataStream(chunks[1:], n_workers=2, backend="thread")
        result = stream.fit_stream(clf, sequential=True)
        assert result is clf

    def test_fit_stream_parallel_thread(self):
        from sklearn.linear_model import SGDClassifier

        from physml.stream_worker import ParallelDataStream

        chunks = self._make_chunks(n_chunks=4, n=30, d=5)
        # Pre-fit so coef_ exists
        clf = SGDClassifier(max_iter=10, random_state=0)
        X_all = np.vstack([X for X, _ in chunks])
        y_all = np.hstack([y for _, y in chunks])
        clf.fit(X_all, y_all)

        stream = ParallelDataStream(chunks, n_workers=2, backend="thread")
        result = stream.fit_stream(clf, sequential=False)
        assert result is clf

    def test_fit_stream_no_partial_fit(self):
        """Estimator without partial_fit falls back to fit()."""
        from sklearn.ensemble import RandomForestClassifier

        from physml.stream_worker import ParallelDataStream

        chunks = self._make_chunks(n_chunks=2, n=25, d=4)
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        stream = ParallelDataStream(chunks, n_workers=2)
        result = stream.fit_stream(clf, sequential=True)
        assert result is clf

    def test_empty_chunks(self):
        from physml.stream_worker import ParallelDataStream

        stream = ParallelDataStream([], n_workers=2)
        from sklearn.linear_model import SGDClassifier

        clf = SGDClassifier()
        result = stream.fit_stream(clf, sequential=True)
        assert result is clf
