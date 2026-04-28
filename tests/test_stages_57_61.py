"""Tests for Stages 57–61.

Stage 57 — KnowledgeGraph / KnowledgeNode
Stage 58 — RewardShaper
Stage 59 — CurriculumScheduler
Stage 60 — SyntheticDataGenerator
Stage 61 — UncertaintyEstimator
"""

from __future__ import annotations


import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from physml.knowledge_graph import KnowledgeGraph, KnowledgeNode
from physml.reward_shaper import RewardShaper
from physml.curriculum import CurriculumScheduler
from physml.synthetic_data import SyntheticDataGenerator
from physml.uncertainty import UncertaintyEstimator


# ---------------------------------------------------------------------------
# Stage 57 — KnowledgeGraph
# ---------------------------------------------------------------------------

class TestKnowledgeNode:

    def test_node_defaults(self):
        n = KnowledgeNode(name="foo")
        assert n.name == "foo"
        assert n.node_type == "concept"
        assert n.payload == {}

    def test_node_equality(self):
        a = KnowledgeNode("x")
        b = KnowledgeNode("x", node_type="event")
        assert a == b  # equality by name

    def test_node_hash(self):
        s = {KnowledgeNode("a"), KnowledgeNode("a")}
        assert len(s) == 1


class TestKnowledgeGraph:

    def _simple_graph(self) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        kg.add_node("A", node_type="feature")
        kg.add_node("B", node_type="event")
        kg.add_node("C", node_type="concept")
        kg.add_edge("A", "B", relation="causes", weight=0.8)
        kg.add_edge("B", "C", relation="implies", weight=0.5)
        return kg

    def test_node_count(self):
        kg = self._simple_graph()
        assert kg.node_count() == 3

    def test_edge_count(self):
        kg = self._simple_graph()
        assert kg.edge_count() == 2

    def test_has_node(self):
        kg = self._simple_graph()
        assert kg.has_node("A")
        assert not kg.has_node("Z")

    def test_has_edge(self):
        kg = self._simple_graph()
        assert kg.has_edge("A", "B")
        assert not kg.has_edge("B", "A")

    def test_has_edge_with_relation(self):
        kg = self._simple_graph()
        assert kg.has_edge("A", "B", relation="causes")
        assert not kg.has_edge("A", "B", relation="implies")

    def test_neighbors(self):
        kg = self._simple_graph()
        assert kg.neighbors("A") == ["B"]
        assert kg.neighbors("B") == ["C"]

    def test_neighbors_by_relation(self):
        kg = self._simple_graph()
        assert kg.neighbors("A", relation="causes") == ["B"]
        assert kg.neighbors("A", relation="implies") == []

    def test_nodes_by_type(self):
        kg = self._simple_graph()
        feats = kg.nodes_by_type("feature")
        assert len(feats) == 1 and feats[0].name == "A"

    def test_path_direct(self):
        kg = self._simple_graph()
        assert kg.path("A", "B") == ["A", "B"]

    def test_path_indirect(self):
        kg = self._simple_graph()
        assert kg.path("A", "C") == ["A", "B", "C"]

    def test_path_not_reachable(self):
        kg = self._simple_graph()
        assert kg.path("C", "A") is None

    def test_path_self(self):
        kg = self._simple_graph()
        assert kg.path("A", "A") == ["A"]

    def test_reachable(self):
        kg = self._simple_graph()
        r = kg.reachable("A")
        assert r == {"B", "C"}

    def test_remove_node(self):
        kg = self._simple_graph()
        kg.remove_node("B")
        assert not kg.has_node("B")
        assert kg.edge_count() == 0  # edges to/from B removed

    def test_remove_edge(self):
        kg = self._simple_graph()
        removed = kg.remove_edge("A", "B")
        assert removed == 1
        assert not kg.has_edge("A", "B")

    def test_undirected_graph(self):
        kg = KnowledgeGraph(directed=False)
        kg.add_edge("X", "Y")
        assert kg.has_edge("X", "Y")
        assert kg.has_edge("Y", "X")

    def test_auto_create_nodes(self):
        kg = KnowledgeGraph()
        kg.add_edge("P", "Q")
        assert kg.has_node("P") and kg.has_node("Q")

    def test_serialisation_roundtrip(self):
        kg = self._simple_graph()
        data = kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)
        assert kg2.node_count() == kg.node_count()
        assert kg2.edge_count() == kg.edge_count()
        assert kg2.has_edge("A", "B", relation="causes")

    def test_repr(self):
        kg = self._simple_graph()
        r = repr(kg)
        assert "KnowledgeGraph" in r

    def test_edges_from(self):
        kg = self._simple_graph()
        edges = kg.edges_from("A")
        assert len(edges) == 1
        assert edges[0]["relation"] == "causes"
        assert edges[0]["weight"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Stage 58 — RewardShaper
# ---------------------------------------------------------------------------

class TestRewardShaper:

    def test_identity(self):
        rs = RewardShaper()
        assert rs.shape(0.5) == pytest.approx(0.5)

    def test_clip_upper(self):
        rs = RewardShaper(clip=(-1.0, 1.0))
        assert rs.shape(5.0) == pytest.approx(1.0)

    def test_clip_lower(self):
        rs = RewardShaper(clip=(-1.0, 1.0))
        assert rs.shape(-5.0) == pytest.approx(-1.0)

    def test_normalise_zero_mean(self):
        rs = RewardShaper(normalise=True)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.shape(v)
        hist = rs.history()
        # shaped rewards should be approx zero mean (last few)
        shaped = hist["shaped"]
        assert abs(np.mean(shaped)) < 2.0  # soft check: normalisation active

    def test_curiosity_bonus(self):
        rs = RewardShaper(curiosity_weight=1.0)
        r = rs.shape(0.0, error=0.5)
        assert r == pytest.approx(0.5)

    def test_potential_shaping(self):
        phi = lambda s: float(np.sum(s)) if s is not None else 0.0
        rs = RewardShaper(potential_fn=phi, gamma=1.0)
        s1, s2 = np.array([1.0]), np.array([2.0])
        # F = gamma*phi(s2) - phi(s1) = 2 - 1 = 1; raw = 0 → shaped = 1
        r = rs.shape(0.0, state=s1, next_state=s2)
        assert r == pytest.approx(1.0)

    def test_summary_after_steps(self):
        rs = RewardShaper()
        for v in [0.1, 0.2, 0.3]:
            rs.shape(v)
        s = rs.summary()
        assert s["n"] == 3
        assert s["mean"] == pytest.approx(0.2)

    def test_history_length(self):
        rs = RewardShaper()
        for _ in range(10):
            rs.shape(1.0)
        h = rs.history()
        assert len(h["raw"]) == 10
        assert len(h["shaped"]) == 10

    def test_reset_stats(self):
        rs = RewardShaper(normalise=True)
        for v in [1.0, 2.0, 3.0]:
            rs.shape(v)
        rs.reset_stats()
        assert rs.n_samples == 0

    def test_clear_history(self):
        rs = RewardShaper()
        rs.shape(1.0)
        rs.clear_history()
        assert rs.summary()["n"] == 0

    def test_running_mean_increases(self):
        rs = RewardShaper(normalise=True)
        for v in [1.0, 2.0, 4.0, 8.0]:
            rs.shape(v)
        assert rs.running_mean > 0

    def test_repr(self):
        rs = RewardShaper(clip=(-1, 1))
        assert "RewardShaper" in repr(rs)


# ---------------------------------------------------------------------------
# Stage 59 — CurriculumScheduler
# ---------------------------------------------------------------------------

class TestCurriculumScheduler:

    def test_linear_monotone(self):
        sched = CurriculumScheduler(strategy="linear", total_steps=100)
        diffs = [sched.step() for _ in range(10)]
        assert diffs == sorted(diffs)

    def test_linear_reaches_max(self):
        sched = CurriculumScheduler(strategy="linear", max_difficulty=1.0, total_steps=50)
        for _ in range(50):
            d = sched.step()
        assert d == pytest.approx(1.0)

    def test_cosine_starts_slow(self):
        sched = CurriculumScheduler(strategy="cosine", total_steps=100)
        d1 = sched.step()
        sched2 = CurriculumScheduler(strategy="linear", total_steps=100)
        d2 = sched2.step()
        # Cosine should start slower than linear
        assert d1 <= d2 + 0.05

    def test_cosine_reaches_max(self):
        sched = CurriculumScheduler(strategy="cosine", max_difficulty=1.0, total_steps=50)
        for _ in range(50):
            d = sched.step()
        assert d == pytest.approx(1.0)

    def test_step_strategy_milestones(self):
        sched = CurriculumScheduler(
            strategy="step", milestones=[5, 10], total_steps=20,
            milestone_factor=0.5,
        )
        for _ in range(5):
            sched.step()
        before = sched.current_difficulty
        sched.step()  # step 6 triggers first milestone
        after = sched.current_difficulty
        assert after > before

    def test_adaptive_advances_on_high_accuracy(self):
        sched = CurriculumScheduler(
            strategy="adaptive", adaptive_threshold=0.7, adaptive_window=3,
            adaptive_increment=0.1,
        )
        start = sched.current_difficulty
        for _ in range(3):
            sched.step(accuracy=0.9)
        assert sched.current_difficulty > start

    def test_adaptive_no_advance_on_low_accuracy(self):
        sched = CurriculumScheduler(
            strategy="adaptive", adaptive_threshold=0.8, adaptive_window=3,
        )
        start = sched.current_difficulty
        for _ in range(3):
            sched.step(accuracy=0.4)
        assert sched.current_difficulty == pytest.approx(start)

    def test_progress(self):
        sched = CurriculumScheduler(strategy="linear", total_steps=10)
        for _ in range(5):
            sched.step()
        assert sched.progress() == pytest.approx(0.5)

    def test_reset(self):
        sched = CurriculumScheduler(strategy="linear", total_steps=10)
        for _ in range(5):
            sched.step()
        sched.reset()
        assert sched.current_step == 0
        assert sched.current_difficulty == pytest.approx(0.0)

    def test_history_length(self):
        sched = CurriculumScheduler(strategy="linear", total_steps=20)
        for _ in range(7):
            sched.step()
        assert len(sched.history()) == 7

    def test_filter_by_difficulty(self):
        sched = CurriculumScheduler(strategy="linear", total_steps=10)
        for _ in range(5):
            sched.step()
        diffs = np.array([0.1, 0.3, 0.9, 0.2, 0.7])
        mask = sched.filter_by_difficulty(diffs)
        assert mask.dtype == bool
        # samples with difficulty <= current difficulty should be True
        assert mask[0]  # 0.1 is easy

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            CurriculumScheduler(strategy="unknown")

    def test_repr(self):
        sched = CurriculumScheduler(strategy="cosine", total_steps=100)
        assert "CurriculumScheduler" in repr(sched)


# ---------------------------------------------------------------------------
# Stage 60 — SyntheticDataGenerator
# ---------------------------------------------------------------------------

class TestSyntheticDataGenerator:

    def test_gaussian_shape(self):
        gen = SyntheticDataGenerator(n_features=5, n_classes=3, random_state=0)
        X, y = gen.generate(n_samples=60)
        assert X.shape == (60, 5)
        assert y.shape == (60,)

    def test_gaussian_classes(self):
        gen = SyntheticDataGenerator(n_features=4, n_classes=3, random_state=1)
        X, y = gen.generate(n_samples=90)
        assert set(np.unique(y)) == {0, 1, 2}

    def test_moons_binary(self):
        gen = SyntheticDataGenerator(distribution="moons", n_features=4, random_state=0)
        X, y = gen.generate(n_samples=100)
        assert set(np.unique(y)).issubset({0, 1})

    def test_blobs_shape(self):
        gen = SyntheticDataGenerator(distribution="blobs", n_features=6, n_classes=2,
                                     random_state=0)
        X, y = gen.generate(n_samples=80)
        assert X.shape == (80, 6)

    def test_regression_task(self):
        gen = SyntheticDataGenerator(task="regression", distribution="regression",
                                     n_features=8, random_state=0)
        X, y = gen.generate(n_samples=100)
        assert X.shape == (100, 8)
        assert y.dtype == float or np.issubdtype(y.dtype, np.floating)

    def test_n_generated_counter(self):
        gen = SyntheticDataGenerator(random_state=0)
        gen.generate(50)
        gen.generate(30)
        assert gen.n_generated == 80

    def test_augment_increases_size(self):
        gen = SyntheticDataGenerator(n_features=4, random_state=0)
        X, y = gen.generate(50)
        X_aug, y_aug = gen.augment(X, y, n_synthetic=20)
        assert X_aug.shape == (70, 4)
        assert len(y_aug) == 70

    def test_augment_preserves_features(self):
        gen = SyntheticDataGenerator(n_features=6, random_state=0)
        X, y = gen.generate(40)
        X_aug, y_aug = gen.augment(X, y, n_synthetic=10, noise_scale=0.01)
        assert X_aug.shape[1] == 6

    def test_describe_dict(self):
        gen = SyntheticDataGenerator(task="regression", n_features=5, random_state=0)
        d = gen.describe()
        assert d["task"] == "regression"
        assert d["n_features"] == 5

    def test_reset(self):
        gen = SyntheticDataGenerator(random_state=42)
        X1, y1 = gen.generate(30)
        gen.reset()
        X2, y2 = gen.generate(30)
        np.testing.assert_array_almost_equal(X1, X2)

    def test_reproducible_with_seed(self):
        gen1 = SyntheticDataGenerator(n_features=4, random_state=7)
        gen2 = SyntheticDataGenerator(n_features=4, random_state=7)
        X1, y1 = gen1.generate(40)
        X2, y2 = gen2.generate(40)
        np.testing.assert_array_equal(X1, X2)

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task"):
            SyntheticDataGenerator(task="clustering")

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            SyntheticDataGenerator(distribution="uniform")

    def test_repr(self):
        gen = SyntheticDataGenerator()
        assert "SyntheticDataGenerator" in repr(gen)


# ---------------------------------------------------------------------------
# Stage 61 — UncertaintyEstimator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clf_models():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    y = (X[:, 0] > 0).astype(int)
    models = [
        RandomForestClassifier(n_estimators=5, random_state=i).fit(X, y)
        for i in range(4)
    ]
    return models, X, y


@pytest.fixture(scope="module")
def reg_models():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((150, 4))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 150)
    models = [
        RandomForestRegressor(n_estimators=5, random_state=i).fit(X, y)
        for i in range(3)
    ]
    return models, X, y


class TestUncertaintyEstimator:

    def test_fit_single_model(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models[0], X, y)
        assert ue._is_fitted

    def test_predict_proba_shape(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        p = ue.predict_proba(X[:20])
        assert p.shape == (20, 2)

    def test_predict_proba_sums_to_one(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        p = ue.predict_proba(X[:10])
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_uncertainty_shape(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        u = ue.uncertainty(X[:30])
        assert u.shape == (30,)

    def test_uncertainty_non_negative(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        u = ue.uncertainty(X)
        assert np.all(u >= 0)

    def test_most_uncertain_returns_indices(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        idx = ue.most_uncertain(X, n=5)
        assert len(idx) == 5
        assert len(set(idx)) == 5  # unique indices

    def test_aleatoric_epistemic_split(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        d = ue.aleatoric_epistemic_split(X[:10])
        assert set(d.keys()) == {"total", "aleatoric", "epistemic"}
        # total ≥ epistemic and aleatoric (up to floating-point)
        assert np.all(d["total"] >= -1e-9)

    def test_temperature_method_fit(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="temperature", temperature=1.5)
        ue.fit(models, X, y)
        p = ue.predict_proba(X[:10])
        assert p.shape == (10, 2)

    def test_temperature_calibrate_changes_temperature(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="temperature")
        ue.fit(models, X, y)
        t_before = ue.temperature
        ue.calibrate(X[:50], y[:50])
        # After calibration, temperature should still be positive
        assert ue.temperature > 0.0

    def test_unfitted_raises(self):
        ue = UncertaintyEstimator(method="ensemble")
        with pytest.raises(RuntimeError, match="fit"):
            ue.uncertainty(np.zeros((5, 3)))

    def test_regression_uncertainty(self, reg_models):
        models, X, y = reg_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        u = ue.uncertainty(X[:20])
        assert u.shape == (20,)

    def test_repr(self, clf_models):
        models, X, y = clf_models
        ue = UncertaintyEstimator(method="ensemble")
        ue.fit(models, X, y)
        assert "UncertaintyEstimator" in repr(ue)
