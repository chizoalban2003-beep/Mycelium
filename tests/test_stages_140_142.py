"""Tests for Stages 140–142.

Stage 140 — Goals REST API  (GET/POST/DELETE /goals, /schedules)
Stage 141 — Streaming chat  (POST /chat/stream  SSE)
Stage 142 — Daily digest    (GET /digest, companion.daily_digest())
"""

from __future__ import annotations

import json
import pytest

try:
    from physml.server import create_app
    from fastapi.testclient import TestClient
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

pytestmark = pytest.mark.skipif(not _FASTAPI_OK, reason="fastapi/httpx not installed")


@pytest.fixture(scope="module")
def client():
    app = create_app()
    client = TestClient(app)
    token = client.post("/auth/token", json={"user_id": "test", "password": ""}).json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


# ---------------------------------------------------------------------------
# Stage 140 — Goals REST API
# ---------------------------------------------------------------------------

class TestGoalsAPI:
    def test_list_goals_returns_200(self, client):
        r = client.get("/goals")
        assert r.status_code == 200
        data = r.json()
        assert "goals" in data
        assert "total" in data

    def test_list_goals_invalid_status(self, client):
        r = client.get("/goals?status=NOTAREAL")
        assert r.status_code == 400

    def test_create_goal_returns_201(self, client):
        r = client.post("/goals", json={"description": "test goal for api"})
        assert r.status_code == 201
        data = r.json()
        assert "id" in data
        assert "status" in data

    def test_created_goal_appears_in_list(self, client):
        r = client.post("/goals", json={"description": "unique api test goal xyz"})
        gid = r.json()["id"]
        r2 = client.get("/goals")
        ids = [g["id"] for g in r2.json()["goals"]]
        assert gid in ids

    def test_get_goal_by_id(self, client):
        r = client.post("/goals", json={"description": "goal to fetch"})
        gid = r.json()["id"]
        r2 = client.get(f"/goals/{gid}")
        assert r2.status_code == 200
        assert r2.json()["id"] == gid

    def test_get_goal_not_found(self, client):
        r = client.get("/goals/nonexistentid")
        assert r.status_code == 404

    def test_cancel_goal(self, client):
        r = client.post("/goals", json={"description": "goal to cancel"})
        gid = r.json()["id"]
        r2 = client.delete(f"/goals/{gid}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "cancelled"

    def test_cancel_already_cancelled(self, client):
        r = client.post("/goals", json={"description": "cancel twice"})
        gid = r.json()["id"]
        client.delete(f"/goals/{gid}")
        r2 = client.delete(f"/goals/{gid}")
        assert r2.status_code == 404

    def test_cancel_nonexistent(self, client):
        r = client.delete("/goals/doesnotexist")
        assert r.status_code == 404

    def test_filter_by_status(self, client):
        client.post("/goals", json={"description": "filter test goal"})
        r = client.get("/goals?status=pending")
        assert r.status_code == 200
        for g in r.json()["goals"]:
            assert g["status"] == "pending"


class TestSchedulesAPI:
    def test_list_schedules_returns_200(self, client):
        r = client.get("/schedules")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "schedules" in data

    def test_create_schedule_daily(self, client):
        r = client.post("/schedules", json={
            "description": "daily check api test",
            "schedule": "daily",
        })
        assert r.status_code == 201
        data = r.json()
        assert "id" in data

    def test_create_schedule_hourly(self, client):
        r = client.post("/schedules", json={
            "description": "hourly check api test",
            "schedule": "hourly",
        })
        assert r.status_code == 201

    def test_create_schedule_every_n_minutes(self, client):
        r = client.post("/schedules", json={
            "description": "every 30 minutes task",
            "schedule": "every 30 minutes",
        })
        assert r.status_code == 201

    def test_create_schedule_invalid(self, client):
        r = client.post("/schedules", json={
            "description": "bad schedule",
            "schedule": "not a valid schedule string xyz",
        })
        assert r.status_code == 400

    def test_remove_schedule(self, client):
        r = client.post("/schedules", json={
            "description": "schedule to remove",
            "schedule": "daily",
        })
        sid = r.json()["id"]
        r2 = client.delete(f"/schedules/{sid}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "removed"

    def test_remove_nonexistent_schedule(self, client):
        r = client.delete("/schedules/doesnotexist")
        assert r.status_code == 404

    def test_created_schedule_appears_in_list(self, client):
        r = client.post("/schedules", json={
            "description": "unique schedule listing test abc",
            "schedule": "weekly",
        })
        sid = r.json()["id"]
        r2 = client.get("/schedules")
        ids = [s["id"] for s in r2.json()["schedules"]]
        assert sid in ids


# ---------------------------------------------------------------------------
# Stage 141 — Streaming chat
# ---------------------------------------------------------------------------

class TestStreamingChat:
    def test_stream_endpoint_exists(self, client):
        r = client.post(
            "/chat/stream",
            json={"message": "hello", "user_id": "test"},
        )
        assert r.status_code == 200

    def test_stream_content_type_is_sse(self, client):
        r = client.post(
            "/chat/stream",
            json={"message": "hi", "user_id": "test"},
        )
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_response_contains_data_lines(self, client):
        r = client.post(
            "/chat/stream",
            json={"message": "hello world", "user_id": "test"},
        )
        lines = r.text.split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 1

    def test_stream_ends_with_done(self, client):
        r = client.post(
            "/chat/stream",
            json={"message": "test message", "user_id": "test"},
        )
        assert "data: [DONE]" in r.text

    def test_stream_tokens_are_valid_json(self, client):
        r = client.post(
            "/chat/stream",
            json={"message": "what can you do?", "user_id": "test"},
        )
        for line in r.text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                payload = line[6:]
                parsed = json.loads(payload)
                assert "token" in parsed or "error" in parsed

    def test_stream_tokens_concatenate_to_nonempty(self, client):
        r = client.post(
            "/chat/stream",
            json={"message": "hello", "user_id": "test"},
        )
        tokens = []
        for line in r.text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    d = json.loads(line[6:])
                    if "token" in d:
                        tokens.append(d["token"])
                except Exception:
                    pass
        full = "".join(tokens)
        assert len(full) > 0


# ---------------------------------------------------------------------------
# Stage 142 — Daily digest
# ---------------------------------------------------------------------------

class TestDailyDigest:
    def test_digest_endpoint_returns_200(self, client):
        r = client.get("/digest")
        assert r.status_code == 200
        assert "digest" in r.json()

    def test_digest_text_nonempty(self, client):
        r = client.get("/digest")
        assert len(r.json()["digest"]) > 10

    def test_digest_contains_goal_section(self, client):
        r = client.get("/digest")
        text = r.json()["digest"]
        assert "Goal" in text or "goal" in text

    def test_digest_method_directly(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            digest = c.daily_digest()
            assert isinstance(digest, str)
            assert len(digest) > 20
            assert "Digest" in digest
        finally:
            c.stop()

    def test_digest_contains_schedule_info(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            digest = c.daily_digest()
            assert "Schedule" in digest or "schedule" in digest
        finally:
            c.stop()

    def test_digest_contains_model_info(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            digest = c.daily_digest()
            assert "Model" in digest or "model" in digest
        finally:
            c.stop()


# ---------------------------------------------------------------------------
# Stage 140 — Goals handler in companion chat interface
# ---------------------------------------------------------------------------

class TestGoalsChatInterface:
    def test_chat_goal_intent_queues_goal(self, client):
        r = client.post("/chat", json={
            "message": "add a goal to read my data file",
            "user_id": "test",
        })
        assert r.status_code == 200
        resp = r.json()["response"]
        assert isinstance(resp, str)

    def test_goals_status_via_chat(self, client):
        r = client.post("/chat", json={
            "message": "show me my goals",
            "user_id": "test",
        })
        assert r.status_code == 200
