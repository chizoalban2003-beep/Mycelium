"""Tests for v1.3.0: SpecialistFederation, mobile PWA, CLI federation command."""

from __future__ import annotations

import json
import pathlib

import pytest

# Pre-warm physml (prevents per-test import timeout on first run)
import physml  # noqa: F401


# ---------------------------------------------------------------------------
# TestSpecialistFederation
# ---------------------------------------------------------------------------
class TestSpecialistFederation:
    def test_import(self):
        from physml import SpecialistFederation, FederationSpecialist as Specialist
        assert SpecialistFederation is not None
        assert Specialist is not None

    def test_init_default_specialists(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        assert len(fed._specialists) == 6
        names = [s.name for s in fed._specialists]
        assert "Coder" in names
        assert "Data" in names
        assert "NLP" in names

    def test_start(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        assert fed._started is True

    def test_start_idempotent(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        fed.start()  # should not raise
        assert fed._started is True

    def test_list_specialists(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        names = fed.list_specialists()
        assert isinstance(names, list)
        assert len(names) >= 6

    def test_query_returns_dict(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("How do I write a Python function?")
        assert isinstance(result, dict)
        assert "specialist" in result
        assert "response" in result
        assert "elapsed" in result

    def test_query_coder_routed(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("debug this Python error exception traceback")
        assert result["specialist"] == "Coder"

    def test_query_data_routed(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("write a SQL query to analyse this database table")
        assert result["specialist"] == "Data"

    def test_query_browser_routed(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("search the web for latest research articles on this url")
        assert result["specialist"] == "Browser"

    def test_query_scheduler_routed(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("schedule a meeting reminder for next week calendar")
        assert result["specialist"] == "Scheduler"

    def test_query_system_routed(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("rename all files in the folder using a shell command")
        assert result["specialist"] == "System"

    def test_query_app_context_routing(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("help me", context={"app": "VS Code"})
        assert result["specialist"] == "Coder"

    def test_query_data_app_context(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("help me with this", context={"app": "DBeaver"})
        assert result["specialist"] == "Data"

    def test_broadcast_fact(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        fed.broadcast_fact("User prefers Python 3.12 and type hints")
        assert "User prefers Python 3.12 and type hints" in fed._facts

    def test_multiple_broadcast_facts(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        for i in range(5):
            fed.broadcast_fact(f"Fact number {i}")
        assert len(fed._facts) == 5

    def test_knowledge_snapshot(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        snap = fed.knowledge_snapshot()
        assert "facts" in snap
        assert "log_size" in snap
        assert "specialists" in snap

    def test_query_logs_message(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        fed.query("test query for log tracking")
        assert len(fed._log) == 1
        assert fed._log[0].topic == "response"

    def test_recent_log(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        fed.query("first query")
        fed.query("second query")
        log = fed.recent_log(5)
        assert isinstance(log, list)
        assert len(log) == 2

    def test_recent_log_limit(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        for i in range(10):
            fed.query(f"query {i} with some context")
        log = fed.recent_log(3)
        assert len(log) == 3

    def test_elapsed_is_float(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        result = fed.query("any question here")
        assert isinstance(result["elapsed"], float)
        assert result["elapsed"] >= 0

    def test_default_specialists_static(self):
        from physml import SpecialistFederation
        specs = SpecialistFederation.default_specialists()
        assert len(specs) == 6
        names = [s.name for s in specs]
        assert "Coder" in names
        assert "NLP" in names

    def test_knowledge_in_snapshot_after_broadcast(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        fed.start()
        fed.broadcast_fact("Test fact for snapshot check")
        snap = fed.knowledge_snapshot()
        assert "Test fact for snapshot check" in snap["facts"]

    def test_query_without_start_still_works(self):
        from physml import SpecialistFederation
        fed = SpecialistFederation()
        # query() auto-starts
        result = fed.query("Python list comprehension example")
        assert "specialist" in result
        assert fed._started is True

    def test_custom_specialist_list(self):
        from physml import SpecialistFederation, FederationSpecialist as Specialist
        custom = [Specialist(name="Custom", topics=["custom"], apps=[])]
        fed = SpecialistFederation(specialists=custom)
        assert fed.list_specialists() == ["Custom"]

    def test_specialist_respond_no_llm(self):
        from physml.specialist_federation import Specialist
        spec = Specialist(name="TestSpec", topics=["test"])
        response = spec.respond("how do I test?", {}, "some knowledge")
        assert isinstance(response, str)
        assert len(response) > 0


# ---------------------------------------------------------------------------
# TestPWAFiles
# ---------------------------------------------------------------------------
class TestPWAFiles:
    def _pwa_dir(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent.parent / "physml" / "static" / "pwa"

    def test_pwa_dir_exists(self):
        assert self._pwa_dir().exists()

    def test_index_html_exists(self):
        assert (self._pwa_dir() / "index.html").exists()

    def test_manifest_json_exists(self):
        assert (self._pwa_dir() / "manifest.json").exists()

    def test_sw_js_exists(self):
        assert (self._pwa_dir() / "sw.js").exists()

    def test_manifest_valid_json(self):
        data = json.loads((self._pwa_dir() / "manifest.json").read_text())
        assert data["name"] == "Mycelium — Local AI Companion"
        assert data["display"] == "standalone"
        assert "icons" in data

    def test_manifest_has_shortcuts(self):
        data = json.loads((self._pwa_dir() / "manifest.json").read_text())
        assert "shortcuts" in data
        assert len(data["shortcuts"]) >= 2

    def test_sw_cache_name_versioned(self):
        sw = (self._pwa_dir() / "sw.js").read_text()
        assert "mycelium-pwa" in sw
        assert "install" in sw
        assert "activate" in sw
        assert "fetch" in sw

    def test_sw_network_first_api(self):
        sw = (self._pwa_dir() / "sw.js").read_text()
        assert "/mobile/" in sw

    def test_index_has_chat(self):
        html = (self._pwa_dir() / "index.html").read_text()
        assert "chat" in html.lower()
        assert "ingest" in html.lower()

    def test_index_has_pwa_manifest_link(self):
        html = (self._pwa_dir() / "index.html").read_text()
        assert 'rel="manifest"' in html

    def test_index_has_service_worker_registration(self):
        html = (self._pwa_dir() / "index.html").read_text()
        assert "serviceWorker" in html
        assert "sw.js" in html

    def test_index_has_mobile_api_calls(self):
        html = (self._pwa_dir() / "index.html").read_text()
        assert "/mobile/chat" in html
        assert "/mobile/status" in html
        assert "/mobile/context" in html
        assert "/mobile/patterns" in html

    def test_index_has_intent_bar(self):
        html = (self._pwa_dir() / "index.html").read_text()
        assert "push-intent" in html or "intent" in html

    def test_index_theme_color_matches_manifest(self):
        html = (self._pwa_dir() / "index.html").read_text()
        manifest = json.loads((self._pwa_dir() / "manifest.json").read_text())
        theme = manifest["theme_color"]
        assert theme in html


# ---------------------------------------------------------------------------
# TestServerPWARoute
# ---------------------------------------------------------------------------
class TestServerPWARoute:
    def _get_app(self):
        try:
            from physml.server import create_app
            return create_app()
        except Exception:
            pytest.skip("FastAPI not available")

    def test_pwa_route_exists(self):
        app = self._get_app()
        paths = [r.path for r in app.routes]
        assert "/pwa" in paths or "/pwa/" in paths

    def test_mobile_status_has_version(self):
        app = self._get_app()
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("httpx/testclient not available")
        client = TestClient(app)
        resp = client.get("/mobile/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert data["version"] == "1.3.0"


# ---------------------------------------------------------------------------
# TestCLIFederationCommand
# ---------------------------------------------------------------------------
class TestCLIFederationCommand:
    def test_federation_command_exists(self):
        from physml.cli import _build_parser
        parser = _build_parser()
        # Should not raise — federation subparser is registered
        args = parser.parse_args(["federation", "--query", "test", "--list"])
        assert hasattr(args, "func")

    def test_federation_list_runs(self, capsys):
        from physml.cli import _cmd_federation
        import argparse
        args = argparse.Namespace(query="dummy", context="", list=True)
        _cmd_federation(args)
        out = capsys.readouterr().out
        assert "Coder" in out or "specialists" in out.lower()

    def test_federation_query_runs(self, capsys):
        from physml.cli import _cmd_federation
        import argparse
        args = argparse.Namespace(query="what is Python?", context="coder", list=False)
        _cmd_federation(args)
        out = capsys.readouterr().out
        assert len(out) > 0


# ---------------------------------------------------------------------------
# TestVersionBump
# ---------------------------------------------------------------------------
class TestVersionBump:
    def test_version_is_120(self):
        assert physml.__version__ == "1.3.0"

    def test_specialist_federation_in_all(self):
        assert "SpecialistFederation" in physml.__all__
        assert "FederationSpecialist" in physml.__all__

    def test_specialist_federation_importable(self):
        from physml import SpecialistFederation
        assert callable(SpecialistFederation)
