"""Tests for Stages 143–145.

Stage 143 — CommBridge  (email/SMS/Slack/WhatsApp dispatch)
Stage 144 — DesktopBridge  (file I/O, clipboard, app launch, shell, screen)
Stage 145 — Voice loop activation  (start_voice / stop_voice on companion)
"""

from __future__ import annotations

import os
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Stage 143 — CommBridge
# ---------------------------------------------------------------------------

class TestCommBridge:
    def _bridge(self):
        from physml.comm_bridge import CommBridge
        return CommBridge()

    def test_import(self):
        from physml.comm_bridge import CommBridge, CommResult
        assert CommBridge is not None
        assert CommResult is not None

    def test_comm_result_str_success(self):
        from physml.comm_bridge import CommResult
        r = CommResult(True, "email", "sent to x")
        assert "sent" in str(r)
        assert "email" in str(r)

    def test_comm_result_str_failure(self):
        from physml.comm_bridge import CommResult
        r = CommResult(False, "sms", "no creds")
        assert "failed" in str(r)

    def test_send_email_no_config_returns_failure(self):
        bridge = self._bridge()
        result = bridge.send_email("x@x.com", "Hi", "Body")
        assert not result.success
        assert "email" in result.channel

    def test_send_sms_no_config_returns_failure(self):
        bridge = self._bridge()
        result = bridge.send_sms("+15550000000", "Hello")
        assert not result.success
        assert "sms" in result.channel

    def test_send_slack_no_config_returns_failure(self):
        bridge = self._bridge()
        result = bridge.send_slack("Hello")
        assert not result.success
        assert "slack" in result.channel

    def test_send_whatsapp_no_companion_returns_failure(self):
        bridge = self._bridge()
        result = bridge.send_whatsapp("Alice", "Hello")
        assert not result.success

    def test_parse_and_send_email_no_address(self):
        bridge = self._bridge()
        reply = bridge.parse_and_send_email("no address here")
        assert "No email" in reply

    def test_parse_and_send_email_extracts_address(self):
        bridge = self._bridge()
        # Will fail (no SMTP config) but must extract the address
        reply = bridge.parse_and_send_email("send email to alice@example.com about lunch")
        # Should attempt to send and get a config error, not "No email address found"
        assert "No email address found" not in reply

    def test_parse_and_send_sms_no_phone(self):
        bridge = self._bridge()
        reply = bridge.parse_and_send_sms("no phone here")
        assert "No phone" in reply

    def test_dispatch_email_route(self):
        bridge = self._bridge()
        reply = bridge.dispatch("send email to bob@corp.com about the meeting")
        # Config error expected but routed to email, not "No messaging channel matched"
        assert "No messaging channel matched" not in reply

    def test_dispatch_slack_route(self):
        bridge = self._bridge()
        reply = bridge.dispatch("post to slack: team meeting at 3pm")
        assert "No messaging channel matched" not in reply

    def test_dispatch_sms_route(self):
        bridge = self._bridge()
        reply = bridge.dispatch("send sms to +15551234567 you are late")
        assert "No messaging channel matched" not in reply

    def test_dispatch_unknown_returns_fallback(self):
        bridge = self._bridge()
        reply = bridge.dispatch("blah blah nothing specific here zzz")
        assert isinstance(reply, str)

    def test_status_returns_dict(self):
        bridge = self._bridge()
        s = bridge.status()
        assert isinstance(s, dict)
        assert "email" in s
        assert "sms_twilio" in s
        assert "slack" in s
        assert "whatsapp" in s

    def test_status_email_false_without_env(self):
        bridge = self._bridge()
        s = bridge.status()
        # Unless MYCO_EMAIL_HOST etc. are set, this should be False
        if not os.environ.get("MYCO_EMAIL_HOST"):
            assert not s["email"]

    def test_exported_from_physml(self):
        from physml import CommBridge, CommResult
        assert CommBridge is not None
        assert CommResult is not None


# ---------------------------------------------------------------------------
# Stage 144 — DesktopBridge
# ---------------------------------------------------------------------------

class TestDesktopBridge:
    def _bridge(self):
        from physml.desktop_bridge import DesktopBridge
        return DesktopBridge()

    def test_import(self):
        from physml.desktop_bridge import DesktopBridge, DesktopResult
        assert DesktopBridge is not None
        assert DesktopResult is not None

    def test_desktop_result_str_success(self):
        from physml.desktop_bridge import DesktopResult
        r = DesktopResult(True, "read_file", "content here")
        assert "ok" in str(r)
        assert "read_file" in str(r)

    def test_desktop_result_str_failure(self):
        from physml.desktop_bridge import DesktopResult
        r = DesktopResult(False, "shell", "blocked")
        assert "failed" in str(r)

    def test_read_file_nonexistent(self):
        bridge = self._bridge()
        r = bridge.read_file("/nonexistent/path/file.txt")
        assert not r.success
        assert "not found" in r.message.lower()

    def test_read_file_existing(self):
        bridge = self._bridge()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            fname = f.name
        try:
            r = bridge.read_file(fname)
            assert r.success
            assert "hello world" in r.message
        finally:
            os.unlink(fname)

    def test_write_file_no_permission(self):
        bridge = self._bridge()  # allow_writes=False by default when env unset
        if not os.environ.get("MYCO_ALLOW_WRITES"):
            r = bridge.write_file("/tmp/myco_test_write.txt", "test")
            assert not r.success
            assert "permission" in r.message.lower()

    def test_write_file_with_permission(self):
        from physml.desktop_bridge import DesktopBridge
        bridge = DesktopBridge(allow_writes=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.txt")
            r = bridge.write_file(path, "content")
            assert r.success
            assert os.path.exists(path)

    def test_list_dir_existing(self):
        bridge = self._bridge()
        r = bridge.list_dir("/tmp")
        assert r.success

    def test_list_dir_nonexistent(self):
        bridge = self._bridge()
        r = bridge.list_dir("/nonexistent/path/zzz")
        assert not r.success

    def test_delete_file_no_permission(self):
        bridge = self._bridge()
        if not os.environ.get("MYCO_ALLOW_WRITES"):
            r = bridge.delete_file("/tmp/doesnotmatter.txt")
            assert not r.success

    def test_delete_file_with_permission(self):
        from physml.desktop_bridge import DesktopBridge
        bridge = DesktopBridge(allow_writes=True)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            fname = f.name
        r = bridge.delete_file(fname)
        assert r.success
        assert not os.path.exists(fname)

    def test_status_returns_dict(self):
        bridge = self._bridge()
        s = bridge.status()
        assert isinstance(s, dict)
        assert "file_read" in s
        assert "file_write" in s
        assert "clipboard" in s
        assert "shell" in s

    def test_dispatch_read_file(self):
        bridge = self._bridge()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("read me")
            fname = f.name
        try:
            reply = bridge.dispatch(f"read file {fname}")
            assert "read me" in reply
        finally:
            os.unlink(fname)

    def test_dispatch_list_dir(self):
        bridge = self._bridge()
        reply = bridge.dispatch("list /tmp")
        assert isinstance(reply, str)
        assert "failed" not in reply.lower() or "not a directory" not in reply.lower()

    def test_dispatch_screenshot_no_agent(self):
        bridge = self._bridge()
        reply = bridge.dispatch("take a screenshot")
        # Should gracefully fail without crashing
        assert isinstance(reply, str)

    def test_dispatch_unknown_returns_string(self):
        bridge = self._bridge()
        reply = bridge.dispatch("zzz totally unknown action here")
        assert isinstance(reply, str)

    def test_exported_from_physml(self):
        from physml import DesktopBridge, DesktopResult
        assert DesktopBridge is not None
        assert DesktopResult is not None


# ---------------------------------------------------------------------------
# Stage 145 — Voice loop activation
# ---------------------------------------------------------------------------

class TestVoiceLoopActivation:
    def test_companion_has_start_voice(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "start_voice")
        assert callable(c.start_voice)

    def test_companion_has_stop_voice(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        assert hasattr(c, "stop_voice")
        assert callable(c.stop_voice)

    def test_stop_voice_when_not_started(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            msg = c.stop_voice()
            assert isinstance(msg, str)
            assert "not running" in msg.lower()
        finally:
            c.stop()

    def test_voice_loop_attr_initialised(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            # voice_loop attribute should exist (None until start_voice called)
            assert hasattr(c, "voice_loop")
        finally:
            c.stop()

    def test_comm_bridge_attr_initialised(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            assert c.comm_bridge is not None
        finally:
            c.stop()

    def test_desktop_bridge_attr_initialised(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            assert c.desktop_bridge is not None
        finally:
            c.stop()

    def test_status_includes_comm_channels(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            s = c.status()
            assert "comm_channels" in s
        finally:
            c.stop()

    def test_status_includes_desktop(self):
        from physml.companion import MyceliumCompanion
        c = MyceliumCompanion()
        c.start()
        try:
            s = c.status()
            assert "desktop" in s
        finally:
            c.stop()


# ---------------------------------------------------------------------------
# Stage 145 — Voice REST endpoints
# ---------------------------------------------------------------------------

try:
    from physml.server import create_app
    from fastapi.testclient import TestClient
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

pytestmark_voice = pytest.mark.skipif(not _FASTAPI_OK, reason="fastapi/httpx not installed")


@pytest.fixture(scope="module")
def client():
    if not _FASTAPI_OK:
        pytest.skip("fastapi not installed")
    app = create_app()
    return TestClient(app)


@pytest.mark.skipif(not _FASTAPI_OK, reason="fastapi/httpx not installed")
class TestVoiceRESTEndpoints:
    def test_voice_status_returns_200(self, client):
        r = client.get("/voice/status")
        assert r.status_code == 200
        assert "running" in r.json()

    def test_voice_status_not_running_initially(self, client):
        r = client.get("/voice/status")
        # Voice loop not started by default
        assert r.json()["running"] is False

    def test_voice_stop_when_not_running(self, client):
        r = client.post("/voice/stop")
        assert r.status_code == 200
        assert "not running" in r.json()["message"].lower()

    def test_comm_status_returns_200(self, client):
        r = client.get("/comm/status")
        assert r.status_code == 200
        data = r.json()
        assert "email" in data

    def test_desktop_status_returns_200(self, client):
        r = client.get("/desktop/status")
        assert r.status_code == 200
        data = r.json()
        assert "file_read" in data
