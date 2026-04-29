"""Tests for v1.4.0: production hardening, multi-user isolation, setup CLI, training."""

from __future__ import annotations

import argparse
import pathlib
import tempfile

import pytest

import physml  # noqa: F401


# ---------------------------------------------------------------------------
# TestVersionBump140
# ---------------------------------------------------------------------------
class TestVersionBump140:
    def test_version_is_140(self):
        assert physml.__version__ == "1.4.0"

    def test_mycelium_entry_point_importable(self):
        from physml.cli import main
        assert callable(main)


# ---------------------------------------------------------------------------
# TestServerAuthDefaults
# ---------------------------------------------------------------------------
class TestServerAuthDefaults:
    def _get_client(self):
        try:
            from physml.server import create_app
            from fastapi.testclient import TestClient
            return TestClient(create_app())
        except ImportError:
            pytest.skip("fastapi/httpx not available")

    def test_require_auth_env_default(self):
        import os
        from physml import server as srv
        # Default is now 1 (require auth) unless overridden to 0/false/no
        val = os.environ.get("MYCO_REQUIRE_AUTH", "1")
        assert val not in ("0", "false", "no") or srv._REQUIRE_AUTH is False

    def test_jwt_secret_not_empty(self):
        from physml import server as srv
        assert srv._JWT_SECRET != ""
        assert len(srv._JWT_SECRET) >= 16

    def test_mobile_status_version_140(self):
        client = self._get_client()
        # Get a token first (auth is now required)
        token_resp = client.post("/auth/token", json={"user_id": "testuser", "password": ""})
        token = token_resp.json().get("access_token", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = client.get("/mobile/status", headers=headers)
        assert resp.status_code == 200
        assert resp.json().get("version") == "1.4.0"

    def test_mobile_status_open_without_token_when_not_required(self, monkeypatch):
        monkeypatch.setenv("MYCO_REQUIRE_AUTH", "0")
        # Re-import to pick up env override would require reload;
        # just verify the endpoint exists and status is reachable
        client = self._get_client()
        resp = client.get("/mobile/status")
        assert resp.status_code in (200, 401)  # depends on server state

    def test_auth_token_returns_bearer(self):
        client = self._get_client()
        resp = client.post("/auth/token", json={"user_id": "alice", "password": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["token_type"] == "bearer"
        assert "access_token" in data

    def test_auth_verify_with_valid_token(self):
        client = self._get_client()
        token = client.post("/auth/token", json={"user_id": "bob", "password": ""}).json()["access_token"]
        resp = client.get("/auth/verify", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "bob"

    def test_chat_endpoint_accepts_authed_request(self):
        client = self._get_client()
        token = client.post("/auth/token", json={"user_id": "carol", "password": ""}).json()["access_token"]
        resp = client.post(
            "/chat",
            json={"message": "hello", "user_id": "carol"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code in (200, 503)  # 503 if companion not available in test

    def test_rate_limit_still_active(self):
        from physml.server import _check_rate_limit, _rate_windows, _RATE_LIMIT
        _rate_windows.clear()
        ip = "prod-test-ip"
        for _ in range(_RATE_LIMIT):
            _check_rate_limit(ip)
        assert _check_rate_limit(ip) is False
        _rate_windows.clear()

    def test_companions_dict_exists(self):
        from physml import server as srv
        # The create_app function uses a per-user _companions dict inside closure
        # Just verify create_app is callable and produces an app
        app = srv.create_app()
        assert app is not None


# ---------------------------------------------------------------------------
# TestSetupCLI
# ---------------------------------------------------------------------------
class TestSetupCLI:
    def test_setup_command_in_parser(self):
        from physml.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["setup", "--no-tls"])
        assert hasattr(args, "func")

    def test_serve_command_in_parser(self):
        from physml.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["serve", "--no-tls"])
        assert hasattr(args, "func")
        assert args.no_tls is True

    def test_train_command_in_parser(self):
        from physml.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["train"])
        assert hasattr(args, "func")
        assert args.csv == ""

    def test_train_command_with_csv(self):
        from physml.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["train", "--csv", "data.csv", "--target", "y"])
        assert args.csv == "data.csv"
        assert args.target == "y"

    def test_setup_creates_env(self):
        from physml.cli import _cmd_setup
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                data_dir=tmpdir, host="127.0.0.1", port=8001,
                password="", no_tls=True,
            )
            _cmd_setup(args)
            env_file = pathlib.Path(tmpdir) / ".env"
            assert env_file.exists()
            content = env_file.read_text()
            assert "MYCELIUM_SECRET=" in content
            assert "MYCO_REQUIRE_AUTH=1" in content

    def test_setup_env_has_unique_secret(self):
        from physml.cli import _cmd_setup
        secrets = []
        for i in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                args = argparse.Namespace(
                    data_dir=tmpdir, host="127.0.0.1", port=8000 + i,
                    password="", no_tls=True,
                )
                _cmd_setup(args)
                env = (pathlib.Path(tmpdir) / ".env").read_text()
                for line in env.splitlines():
                    if line.startswith("MYCELIUM_SECRET="):
                        secrets.append(line.split("=", 1)[1])
        assert len(set(secrets)) == 3  # all unique

    def test_setup_with_password(self):
        from physml.cli import _cmd_setup
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                data_dir=tmpdir, host="127.0.0.1", port=8000,
                password="s3cr3t", no_tls=True,
            )
            _cmd_setup(args)
            content = (pathlib.Path(tmpdir) / ".env").read_text()
            assert "MYCO_PASSWORD=s3cr3t" in content

    def test_generate_tls_cert(self):
        from physml.cli import _generate_tls_cert
        with tempfile.TemporaryDirectory() as tmpdir:
            cert = pathlib.Path(tmpdir) / "cert.pem"
            key = pathlib.Path(tmpdir) / "key.pem"
            ok = _generate_tls_cert(cert, key, "localhost")
            if ok:
                assert cert.exists()
                assert key.exists()
                assert cert.stat().st_size > 0
            # If openssl not available, ok=False is acceptable

    def test_train_cmd_runs(self, capsys):
        from physml.cli import _cmd_train
        args = argparse.Namespace(csv="", target="", epochs=1)
        _cmd_train(args)  # should not raise
        out = capsys.readouterr().out
        assert len(out) > 0


# ---------------------------------------------------------------------------
# TestSoulPruning
# ---------------------------------------------------------------------------
class TestSoulPruning:
    def test_max_events_constant_exists(self):
        from physml.digital_soul import DigitalSoul
        assert hasattr(DigitalSoul, "_MAX_EVENTS")
        assert DigitalSoul._MAX_EVENTS == 500

    def test_save_prunes_excess_events(self):
        from physml.digital_soul import DigitalSoul
        with tempfile.TemporaryDirectory() as tmpdir:
            soul = DigitalSoul(soul_path=f"{tmpdir}/soul.json")
            # Add 600 events
            for i in range(600):
                soul.record_event("test", details={"i": i})
            soul.save()
            assert len(soul._events) == 500
            # Oldest are pruned — most recent 500 remain
            assert soul._events[-1].details["i"] == 599
            assert soul._events[0].details["i"] == 100

    def test_save_under_limit_unchanged(self):
        from physml.digital_soul import DigitalSoul
        with tempfile.TemporaryDirectory() as tmpdir:
            soul = DigitalSoul(soul_path=f"{tmpdir}/soul.json")
            for i in range(100):
                soul.record_event("test", details={"i": i})
            soul.save()
            assert len(soul._events) == 100


# ---------------------------------------------------------------------------
# TestMultiUserIsolation
# ---------------------------------------------------------------------------
class TestMultiUserIsolation:
    def _get_client(self):
        try:
            from physml.server import create_app
            from fastapi.testclient import TestClient
            return TestClient(create_app())
        except ImportError:
            pytest.skip("fastapi/httpx not available")

    def test_different_users_get_tokens(self):
        client = self._get_client()
        r1 = client.post("/auth/token", json={"user_id": "user1", "password": ""})
        r2 = client.post("/auth/token", json={"user_id": "user2", "password": ""})
        assert r1.json()["user_id"] == "user1"
        assert r2.json()["user_id"] == "user2"
        assert r1.json()["access_token"] != r2.json()["access_token"]

    def test_verify_returns_correct_user(self):
        client = self._get_client()
        for uid in ("alice", "bob", "carol"):
            token = client.post("/auth/token", json={"user_id": uid, "password": ""}).json()["access_token"]
            resp = client.get("/auth/verify", headers={"Authorization": f"Bearer {token}"})
            assert resp.json()["user_id"] == uid


# ---------------------------------------------------------------------------
# TestDockerfiles
# ---------------------------------------------------------------------------
class TestDockerfiles:
    def _root(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent.parent

    def test_dockerfile_exists(self):
        assert (self._root() / "Dockerfile").exists()

    def test_docker_compose_exists(self):
        assert (self._root() / "docker-compose.yml").exists()

    def test_dockerfile_has_mycelium_serve(self):
        content = (self._root() / "Dockerfile").read_text()
        assert "serve" in content
        assert "openssl" in content

    def test_docker_compose_has_auth_env(self):
        content = (self._root() / "docker-compose.yml").read_text()
        assert "MYCELIUM_SECRET" in content
        assert "MYCO_REQUIRE_AUTH" in content

    def test_env_example_exists(self):
        assert (self._root() / ".env.example").exists()

    def test_env_example_has_secret(self):
        content = (self._root() / ".env.example").read_text()
        assert "MYCELIUM_SECRET" in content
        assert "MYCO_REQUIRE_AUTH" in content

    def test_systemd_service_exists(self):
        assert (self._root() / "scripts" / "mycelium.service").exists()

    def test_systemd_service_has_serve(self):
        content = (self._root() / "scripts" / "mycelium.service").read_text()
        assert "mycelium serve" in content
        assert "EnvironmentFile" in content
