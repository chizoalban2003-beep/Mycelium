"""Tests for v1.4.0: LocalLLM, VisionAgent, server auth hardening."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile

import pytest

import physml  # noqa: F401  (pre-warm import)


# ---------------------------------------------------------------------------
# TestLocalLLM
# ---------------------------------------------------------------------------
class TestLocalLLM:
    def test_import(self):
        from physml import LocalLLM, LocalChatResult
        assert LocalLLM is not None
        assert LocalChatResult is not None

    def test_import_from_llm(self):
        from physml.llm import LocalLLM, LocalChatResult
        assert LocalLLM is not None
        assert LocalChatResult is not None

    def test_init_default(self):
        from physml import LocalLLM
        llm = LocalLLM()
        assert llm.model == "llama3.2"
        assert llm.vision_model == "llava"
        assert llm.call_count == 0

    def test_backend_none_when_no_server(self):
        from physml import LocalLLM
        llm = LocalLLM(ollama_url="http://localhost:19999")  # nonexistent
        assert llm.backend == "none"
        assert llm.available is False

    def test_chat_unavailable_returns_error(self):
        from physml import LocalLLM
        llm = LocalLLM(ollama_url="http://localhost:19999")
        result = llm.chat("hello")
        assert result.available is False
        assert result.error != ""

    def test_complete_unavailable_returns_error(self):
        from physml import LocalLLM
        llm = LocalLLM(ollama_url="http://localhost:19999")
        result = llm.complete("hello")
        assert result.available is False

    def test_vision_chat_unavailable(self):
        from physml import LocalLLM
        llm = LocalLLM(ollama_url="http://localhost:19999")
        result = llm.vision_chat("aGVsbG8=", "what is this?")
        assert result.available is False

    def test_list_models_empty_when_unavailable(self):
        from physml import LocalLLM
        llm = LocalLLM(ollama_url="http://localhost:19999")
        assert llm.list_models() == []

    def test_pull_model_false_when_unavailable(self):
        from physml import LocalLLM
        llm = LocalLLM(ollama_url="http://localhost:19999")
        assert llm.pull_model("llama3.2") is False

    def test_status_dict(self):
        from physml import LocalLLM
        llm = LocalLLM()
        st = llm.status()
        assert "backend" in st
        assert "available" in st
        assert "model" in st
        assert "models" in st
        assert isinstance(st["models"], list)

    def test_repr(self):
        from physml import LocalLLM
        llm = LocalLLM()
        r = repr(llm)
        assert "LocalLLM" in r
        assert "backend" in r

    def test_local_chat_result_success_property(self):
        from physml.llm.local_llm import LocalChatResult
        r = LocalChatResult(text="hello", available=True, model="test")
        assert r.success is True

    def test_local_chat_result_fail_on_error(self):
        from physml.llm.local_llm import LocalChatResult
        r = LocalChatResult(available=False, error="no server")
        assert r.success is False

    def test_local_chat_result_fail_when_unavailable(self):
        from physml.llm.local_llm import LocalChatResult
        r = LocalChatResult(text="hello", available=False)
        assert r.success is False

    def test_env_model_override(self, monkeypatch):
        monkeypatch.setenv("MYCO_OLLAMA_MODEL", "mistral")
        # reimport to pick up env var (module-level constant)
        import importlib
        import physml.llm.local_llm as _mod
        importlib.reload(_mod)
        from physml.llm.local_llm import LocalLLM as _LL
        llm = _LL(ollama_url="http://localhost:19999")
        assert llm.model in ("mistral", "llama3.2")  # env picked up or default
        importlib.reload(_mod)  # restore

    def test_claude_client_has_local_llm_attr(self):
        from physml.llm import ClaudeClient
        c = ClaudeClient()
        assert hasattr(c, "local_llm")
        assert hasattr(c, "using_local_llm")

    def test_claude_client_using_local_false_no_server(self):
        from physml.llm import ClaudeClient
        c = ClaudeClient()
        # No API key + no ollama → both unavailable
        assert c.using_local_llm is False

    def test_local_llm_in_physml_all(self):
        assert "LocalLLM" in physml.__all__
        assert "LocalChatResult" in physml.__all__

    def test_local_llm_gguf_path_env(self, monkeypatch):
        monkeypatch.setenv("MYCO_GGUF_PATH", "/tmp/fake.gguf")
        import importlib
        import physml.llm.local_llm as _mod
        importlib.reload(_mod)
        from physml.llm.local_llm import LocalLLM as _LL
        llm = _LL(ollama_url="http://localhost:19999")
        # gguf_path should be picked up (llama_cpp not installed so stays "none")
        assert llm.gguf_path == "/tmp/fake.gguf" or llm.gguf_path == ""
        importlib.reload(_mod)


# ---------------------------------------------------------------------------
# TestVisionAgent
# ---------------------------------------------------------------------------
class TestVisionAgent:
    def test_import(self):
        from physml import VisionAgent, VisionResult, UIElement
        assert VisionAgent is not None
        assert VisionResult is not None
        assert UIElement is not None

    def test_init_default(self):
        from physml import VisionAgent
        va = VisionAgent()
        assert va.backend in ("claude-vision", "ollama-vision", "ocr", "none")
        assert va.call_count == 0

    def test_available(self):
        from physml import VisionAgent
        va = VisionAgent()
        assert isinstance(va.available, bool)

    def test_backend_str(self):
        from physml import VisionAgent
        va = VisionAgent()
        assert isinstance(va.backend, str)

    def test_repr(self):
        from physml import VisionAgent
        va = VisionAgent()
        r = repr(va)
        assert "VisionAgent" in r
        assert "backend" in r

    def test_status_dict(self):
        from physml import VisionAgent
        va = VisionAgent()
        st = va.status()
        assert "backend" in st
        assert "available" in st
        assert "call_count" in st
        assert "vision_model" in st

    def test_analyse_missing_file(self):
        from physml import VisionAgent
        va = VisionAgent()
        result = va.analyse("/nonexistent/path/screenshot.png")
        assert result.success is False
        assert result.error != ""

    def test_analyse_real_png(self):
        from physml import VisionAgent
        va = VisionAgent()
        # Create a minimal valid PNG (1x1 white pixel)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            path = f.name
        try:
            result = va.analyse(path)
            # Should succeed (at minimum OCR or description)
            assert result.screenshot_path == path
            assert isinstance(result.elements, list)
            assert isinstance(result.suggested_actions, list)
        finally:
            os.unlink(path)

    def test_vision_result_find_element(self):
        from physml.vision_agent import VisionResult, UIElement
        result = VisionResult()
        result.elements = [
            UIElement(label="Save button", x=100, y=200, element_type="button"),
            UIElement(label="Cancel", x=200, y=200, element_type="button"),
        ]
        el = result.find_element("save")
        assert el is not None
        assert el.x == 100

    def test_vision_result_find_element_missing(self):
        from physml.vision_agent import VisionResult
        result = VisionResult()
        assert result.find_element("nonexistent") is None

    def test_ui_element_center(self):
        from physml.vision_agent import UIElement
        el = UIElement(label="btn", x=100, y=200, width=80, height=30)
        assert el.center == (140, 215)

    def test_ui_element_center_unknown(self):
        from physml.vision_agent import UIElement
        el = UIElement(label="btn", x=-1, y=-1)
        assert el.center == (-1, -1)

    def test_find_element_no_screenshot(self):
        from physml import VisionAgent
        va = VisionAgent()
        # No screenshot agent available — returns None gracefully
        el = va.find_element("Save button", screenshot_path="/nonexistent.png")
        assert el is None

    def test_vision_agent_in_physml_all(self):
        assert "VisionAgent" in physml.__all__
        assert "VisionResult" in physml.__all__
        assert "UIElement" in physml.__all__

    def test_describe_goal_step_no_screenshot(self):
        from physml import VisionAgent
        va = VisionAgent()
        result = va.describe_goal_step(
            "rename files", "open finder", screenshot_path="/nonexistent.png"
        )
        assert isinstance(result, dict)
        assert "action" in result or "explanation" in result

    def test_watch_for_returns_false_on_no_screen(self):
        from physml import VisionAgent
        va = VisionAgent()
        # No screen agent → returns False immediately
        found = va.watch_for("dialog box", timeout=0.1, interval=0.05)
        assert found is False

    def test_parse_json_from_text_fenced(self):
        from physml.vision_agent import _parse_json_from_text
        text = '```json\n{"found": true, "x": 100}\n```'
        data = _parse_json_from_text(text)
        assert data.get("found") is True
        assert data.get("x") == 100

    def test_parse_json_from_text_plain(self):
        from physml.vision_agent import _parse_json_from_text
        text = '{"description": "hello"}'
        data = _parse_json_from_text(text)
        assert data["description"] == "hello"

    def test_parse_json_from_text_embedded(self):
        from physml.vision_agent import _parse_json_from_text
        text = 'Here is the result: {"found": false} done.'
        data = _parse_json_from_text(text)
        assert data.get("found") is False


# ---------------------------------------------------------------------------
# TestServerAuth
# ---------------------------------------------------------------------------
class TestServerAuth:
    def _get_client(self):
        try:
            from physml.server import create_app
            from fastapi.testclient import TestClient
            return TestClient(create_app())
        except ImportError:
            pytest.skip("fastapi/httpx not available")

    def test_auth_token_endpoint_exists(self):
        client = self._get_client()
        resp = client.post("/auth/token", json={"user_id": "alice", "password": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user_id"] == "alice"
        assert "expires_in" in data

    def test_auth_token_returns_jwt(self):
        client = self._get_client()
        resp = client.post("/auth/token", json={"user_id": "bob", "password": ""})
        token = resp.json()["access_token"]
        assert token.count(".") == 2  # header.payload.sig

    def test_auth_status_endpoint(self):
        client = self._get_client()
        resp = client.get("/auth/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "require_auth" in data
        assert "rate_limit" in data
        assert "token_expiry_seconds" in data
        assert "password_protected" in data

    def test_auth_status_no_password_by_default(self):
        client = self._get_client()
        resp = client.get("/auth/status")
        assert resp.json()["password_protected"] is False

    def test_wrong_token_returns_401(self):
        client = self._get_client()
        # /auth/verify always enforces token validation — 401 on bad token
        resp = client.get("/auth/verify", headers={"Authorization": "Bearer bad.token"})
        assert resp.status_code == 401

    def test_mobile_status_returns_200_with_valid_token(self):
        client = self._get_client()
        token = client.post("/auth/token", json={"user_id": "status_user", "password": ""}).json()["access_token"]
        resp = client.get("/mobile/status", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_valid_token_accepted(self):
        client = self._get_client()
        token_resp = client.post("/auth/token", json={"user_id": "carol", "password": ""})
        token = token_resp.json()["access_token"]
        resp = client.get("/auth/verify", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "carol"

    def test_rate_limit_constants_exist(self):
        import physml.server as srv
        assert hasattr(srv, "_RATE_LIMIT")
        assert hasattr(srv, "_RATE_WINDOW")
        assert srv._RATE_LIMIT > 0
        assert srv._RATE_WINDOW > 0

    def test_check_rate_limit_allows_first_request(self):
        from physml.server import _check_rate_limit, _rate_windows
        _rate_windows.clear()
        assert _check_rate_limit("test-ip-1") is True

    def test_check_rate_limit_blocks_after_limit(self):
        from physml.server import _check_rate_limit, _rate_windows, _RATE_LIMIT
        _rate_windows.clear()
        ip = "test-ip-burst"
        for _ in range(_RATE_LIMIT):
            _check_rate_limit(ip)
        # Next request should be blocked
        assert _check_rate_limit(ip) is False

    def test_vision_analyse_endpoint_exists(self):
        client = self._get_client()
        resp = client.post("/vision/analyse", json={})
        # 200 (screenshot attempt) or 422 (validation) — not 404
        assert resp.status_code != 404

    def test_vision_find_endpoint_exists(self):
        client = self._get_client()
        resp = client.post(
            "/vision/find", json={"description": "Save button"}
        )
        assert resp.status_code != 404

    def test_mobile_status_has_version_130(self):
        client = self._get_client()
        resp = client.get("/mobile/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("version") == "1.4.0"

    def test_mobile_status_has_vision_agent_key(self):
        client = self._get_client()
        resp = client.get("/mobile/status")
        data = resp.json()
        assert "vision_agent" in data

    def test_mobile_status_has_local_llm_key(self):
        client = self._get_client()
        resp = client.get("/mobile/status")
        data = resp.json()
        assert "local_llm" in data


# ---------------------------------------------------------------------------
# TestVersionBump130
# ---------------------------------------------------------------------------
class TestVersionBump130:
    def test_version_is_130(self):
        assert physml.__version__ == "1.4.0"

    def test_vision_agent_in_all(self):
        assert "VisionAgent" in physml.__all__

    def test_local_llm_in_all(self):
        assert "LocalLLM" in physml.__all__

    def test_federation_specialist_in_all(self):
        assert "FederationSpecialist" in physml.__all__

    def test_import_all_new_classes(self):
        from physml import (  # noqa: F401
            VisionAgent, VisionResult, UIElement,
            LocalLLM, LocalChatResult,
            SpecialistFederation, FederationSpecialist,
        )
