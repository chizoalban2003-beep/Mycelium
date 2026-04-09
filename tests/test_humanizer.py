"""Tests for the app name humanizer."""
from mycelium_app.humanizer import (
    humanize_app, humanize_signal, humanize_feature, humanize_layer, humanize_apps_dict,
)


def test_known_apps():
    assert humanize_app("chrome") == "Chrome Browser"
    assert humanize_app("firefox") == "Firefox Browser"
    assert humanize_app("code") == "VS Code"
    assert humanize_app("thunar") == "Files"
    assert humanize_app("nautilus") == "Files"
    assert humanize_app("slack") == "Slack"
    assert humanize_app("spotify") == "Spotify"
    assert humanize_app("bash") == "Terminal"
    assert humanize_app("node") == "Node.js App"
    assert humanize_app("python3") == "Python App"


def test_strip_extensions():
    assert humanize_app("chrome.exe") == "Chrome Browser"
    assert humanize_app("desktop-init.sh") == "Desktop Init"
    assert humanize_app("setup.py") == "Setup"


def test_unknown_app_capitalized():
    assert humanize_app("myweirdapp") == "Myweirdapp"
    assert humanize_app("some-tool") == "Some Tool"


def test_empty_and_none():
    assert humanize_app("") == "Unknown"
    assert humanize_app(None) == "Unknown"


def test_signal_types():
    assert humanize_signal("system_boot") == "Device started"
    assert humanize_signal("resource_pulse") == "System check"
    assert humanize_signal("app_open") == "App opened"
    assert humanize_signal("network_flow") == "Internet activity"
    assert humanize_signal("disk_io") == "Storage activity"


def test_unknown_signal_cleaned():
    assert humanize_signal("some_custom_signal") == "Some Custom Signal"


def test_features():
    assert humanize_feature("cpu_mean") == "CPU Usage"
    assert humanize_feature("net_recv_bytes") == "Data Downloaded"
    assert humanize_feature("context_switches") == "App Switching"
    assert humanize_feature("hour_of_day") == "Time of Day"


def test_app_feature_columns():
    assert "Chrome Browser" in humanize_feature("app_chrome")
    assert "usage" in humanize_feature("app_chrome").lower()


def test_layers():
    assert humanize_layer("bedrock") == "Foundation"
    assert humanize_layer("suspension") == "Active Patterns"
    assert humanize_layer("turbulent") == "Changing Signals"


def test_humanize_apps_dict():
    raw = {"chrome": 5, "chrome_crashpad_handler": 2, "thunar": 3}
    result = humanize_apps_dict(raw)

    assert "Chrome Browser" in result
    assert "Files" in result
    # Chrome + crashpad should merge
    assert result["Chrome Browser"] >= 5


def test_system_processes():
    assert humanize_app("dbus-daemon") == "System Bus"
    assert humanize_app("gpg-agent") == "Security Agent"
    assert humanize_app("xfdesktop") == "Desktop"
    assert humanize_app("ssh-agent") == "SSH Agent"
