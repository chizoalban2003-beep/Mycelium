"""Tests for Stage 139 — GoalFeedbackStore (goal outcome feedback loop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List



# ---------------------------------------------------------------------------
# Helpers — minimal GoalRecord stub for tests without touching disk
# ---------------------------------------------------------------------------

class _Status(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class _GoalRecord:
    id: str
    description: str
    status: _Status
    steps: List[dict] = field(default_factory=list)
    elapsed: float = 1.0
    error: str = None


def _record(description, status=_Status.COMPLETED, steps=None):
    steps = steps or [
        {"description": "read file", "status": "ok"},
        {"description": "process data", "status": "ok"},
    ]
    return _GoalRecord(id="test01", description=description, status=status, steps=steps)


# ---------------------------------------------------------------------------
# Tests: GoalOutcome
# ---------------------------------------------------------------------------

class TestGoalOutcome:
    def test_import(self):
        from physml.goal_feedback import GoalOutcome
        assert GoalOutcome is not None

    def test_to_dict_roundtrip(self):
        from physml.goal_feedback import GoalOutcome
        o = GoalOutcome(
            goal_id="abc",
            description="test goal",
            status="completed",
            steps=["step one"],
            successful_steps=["step one"],
            elapsed=2.5,
        )
        d = o.to_dict()
        restored = GoalOutcome.from_dict(d)
        assert restored.goal_id == "abc"
        assert restored.description == "test goal"
        assert restored.status == "completed"
        assert restored.steps == ["step one"]
        assert restored.elapsed == 2.5

    def test_from_dict_defaults(self):
        from physml.goal_feedback import GoalOutcome
        o = GoalOutcome.from_dict({
            "goal_id": "x",
            "description": "d",
            "status": "failed",
            "steps": [],
            "successful_steps": [],
        })
        assert o.error is None
        assert o.elapsed == 0.0


# ---------------------------------------------------------------------------
# Tests: GoalFeedbackStore
# ---------------------------------------------------------------------------

class TestGoalFeedbackStore:
    def _make_store(self, tmp_path):
        from physml.goal_feedback import GoalFeedbackStore
        return GoalFeedbackStore(state_dir=str(tmp_path), min_similarity=0.2)

    def test_import(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store is not None

    def test_record_stores_outcome(self, tmp_path):
        store = self._make_store(tmp_path)
        r = _record("read sales.csv and summarise revenue")
        store.record(r)
        assert store.stats()["total"] == 1
        assert store.stats()["completed"] == 1

    def test_record_failed_goal(self, tmp_path):
        store = self._make_store(tmp_path)
        r = _record("train model", status=_Status.FAILED)
        store.record(r)
        assert store.stats()["failed"] == 1

    def test_find_similar_exact_match(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record(_record("read data and train model"))
        results = store.find_similar("read data and train model")
        assert len(results) == 1

    def test_find_similar_partial_match(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record(_record("read sales.csv and summarise revenue trends"))
        results = store.find_similar("read csv and summarise data")
        assert len(results) >= 1

    def test_find_similar_no_match(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record(_record("read sales.csv and summarise revenue"))
        results = store.find_similar("completely unrelated xyz query")
        assert len(results) == 0

    def test_find_similar_status_filter(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record(_record("train model on data", status=_Status.FAILED))
        # Should not return failed outcome when filtering for completed
        results = store.find_similar("train model on data", status_filter="completed")
        assert len(results) == 0

    def test_find_similar_returns_n(self, tmp_path):
        store = self._make_store(tmp_path)
        for i in range(5):
            store.record(_record(f"read data and process file {i}"))
        results = store.find_similar("read data and process", n=2)
        assert len(results) <= 2

    def test_best_steps_for_returns_successful_steps(self, tmp_path):
        store = self._make_store(tmp_path)
        steps = [
            {"description": "read file", "status": "ok"},
            {"description": "clean data", "status": "ok"},
            {"description": "save results", "status": "ok"},
        ]
        store.record(_record("read csv file and clean data then save", steps=steps))
        result = store.best_steps_for("read csv file and clean data")
        assert result == ["read file", "clean data", "save results"]

    def test_best_steps_for_no_match(self, tmp_path):
        store = self._make_store(tmp_path)
        result = store.best_steps_for("completely unrelated xyz")
        assert result == []

    def test_persistence_across_instances(self, tmp_path):
        from physml.goal_feedback import GoalFeedbackStore
        s1 = GoalFeedbackStore(state_dir=str(tmp_path))
        s1.record(_record("train model on sales data"))
        s2 = GoalFeedbackStore(state_dir=str(tmp_path))
        assert s2.stats()["total"] == 1

    def test_max_outcomes_prunes_oldest(self, tmp_path):
        from physml.goal_feedback import GoalFeedbackStore
        store = GoalFeedbackStore(state_dir=str(tmp_path), max_outcomes=3)
        for i in range(5):
            store.record(_record(f"goal number {i}"))
        assert store.stats()["total"] == 3

    def test_stats_keys(self, tmp_path):
        store = self._make_store(tmp_path)
        s = store.stats()
        assert "total" in s
        assert "completed" in s
        assert "failed" in s
        assert "blocked" in s

    def test_similarity_keyword_overlap(self):
        from physml.goal_feedback import _similarity
        assert _similarity("read and train model", "read and train model") == 1.0
        assert _similarity("read data", "write data") > 0.0
        assert _similarity("aaa bbb", "ccc ddd") == 0.0

    def test_keywords_strips_stop_words(self):
        from physml.goal_feedback import _keywords
        kw = _keywords("read the data and process it")
        assert "the" not in kw
        assert "and" not in kw
        assert "read" in kw
        assert "data" in kw


# ---------------------------------------------------------------------------
# Tests: GoalEngine integration
# ---------------------------------------------------------------------------

class TestGoalEngineFeedbackIntegration:
    def test_engine_has_feedback_store(self, tmp_path):
        from physml.goal_engine import GoalEngine
        engine = GoalEngine(state_dir=str(tmp_path))
        assert hasattr(engine, "_feedback")
        assert engine._feedback is not None

    def test_engine_records_completed_goal(self, tmp_path):
        from physml.goal_engine import GoalEngine
        engine = GoalEngine(state_dir=str(tmp_path))
        gid = engine.add_goal("read data and summarise", run_immediately=True)
        goal = engine.get(gid)
        # feedback should have been recorded
        stats = engine._feedback.stats()
        assert stats["total"] >= 1

    def test_engine_reuses_past_steps(self, tmp_path):
        from physml.goal_engine import GoalEngine
        engine = GoalEngine(state_dir=str(tmp_path))

        # Pre-seed the feedback store with a known past outcome
        from physml.goal_feedback import GoalFeedbackStore
        store = GoalFeedbackStore(state_dir=str(tmp_path))
        store.record(type("R", (), {
            "id": "prev01",
            "description": "read sales.csv and summarise revenue trends",
            "status": type("S", (), {"value": "completed"})(),
            "steps": [
                {"description": "read sales.csv", "status": "ok"},
                {"description": "summarise revenue", "status": "ok"},
            ],
            "elapsed": 1.0,
            "error": None,
        })())
        # Reload engine so it picks up the seeded feedback
        engine2 = GoalEngine(state_dir=str(tmp_path))
        steps = engine2._decompose("read sales.csv and summarise revenue")
        descriptions = [s.description for s in steps]
        assert "read sales.csv" in descriptions

    def test_public_export(self):
        from physml import GoalFeedbackStore, GoalOutcome
        assert GoalFeedbackStore is not None
        assert GoalOutcome is not None
