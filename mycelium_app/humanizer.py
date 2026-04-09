"""Human-friendly label translation for raw OS signals.

Converts raw process names, signal types, and technical identifiers into
labels that any user can understand — regardless of technical background.
"""

from __future__ import annotations

# Process name → friendly display name
_APP_NAMES: dict[str, str] = {
    # Browsers
    "chrome": "Chrome Browser",
    "chrome.exe": "Chrome Browser",
    "google-chrome": "Chrome Browser",
    "google-chrome-stable": "Chrome Browser",
    "chromium": "Chromium Browser",
    "chromium-browser": "Chromium Browser",
    "firefox": "Firefox Browser",
    "firefox.exe": "Firefox Browser",
    "safari": "Safari Browser",
    "msedge": "Microsoft Edge",
    "msedge.exe": "Microsoft Edge",
    "opera": "Opera Browser",
    "brave": "Brave Browser",
    "vivaldi": "Vivaldi Browser",
    # Communication
    "slack": "Slack",
    "discord": "Discord",
    "telegram": "Telegram",
    "telegram-desktop": "Telegram",
    "whatsapp": "WhatsApp",
    "zoom": "Zoom",
    "zoom.us": "Zoom",
    "teams": "Microsoft Teams",
    "teams.exe": "Microsoft Teams",
    "skype": "Skype",
    "signal-desktop": "Signal",
    # Dev tools
    "code": "VS Code",
    "code.exe": "VS Code",
    "code-oss": "VS Code",
    "cursor": "Cursor Editor",
    "sublime_text": "Sublime Text",
    "atom": "Atom Editor",
    "vim": "Vim Editor",
    "nvim": "Neovim Editor",
    "emacs": "Emacs Editor",
    "idea": "IntelliJ IDEA",
    "pycharm": "PyCharm",
    "webstorm": "WebStorm",
    "android-studio": "Android Studio",
    "xcode": "Xcode",
    # Terminal
    "bash": "Terminal",
    "zsh": "Terminal",
    "sh": "Terminal",
    "fish": "Terminal",
    "powershell": "PowerShell",
    "powershell.exe": "PowerShell",
    "cmd.exe": "Command Prompt",
    "terminal": "Terminal",
    "gnome-terminal": "Terminal",
    "konsole": "Terminal",
    "alacritty": "Alacritty Terminal",
    "kitty": "Kitty Terminal",
    "wezterm": "WezTerm",
    "tmux: server": "Terminal (tmux)",
    "tmux: client": "Terminal (tmux)",
    # Office & productivity
    "libreoffice": "LibreOffice",
    "soffice": "LibreOffice",
    "word": "Microsoft Word",
    "winword.exe": "Microsoft Word",
    "excel": "Microsoft Excel",
    "excel.exe": "Microsoft Excel",
    "powerpoint": "PowerPoint",
    "powerpnt.exe": "PowerPoint",
    "outlook": "Outlook",
    "outlook.exe": "Outlook",
    "thunderbird": "Thunderbird Mail",
    "notion": "Notion",
    "obsidian": "Obsidian",
    "evernote": "Evernote",
    "onenote": "OneNote",
    # Media
    "spotify": "Spotify",
    "spotify.exe": "Spotify",
    "vlc": "VLC Player",
    "mpv": "MPV Player",
    "audacity": "Audacity",
    "gimp": "GIMP Editor",
    "inkscape": "Inkscape",
    "blender": "Blender 3D",
    "obs": "OBS Studio",
    "obs64.exe": "OBS Studio",
    # File managers
    "nautilus": "Files",
    "thunar": "Files",
    "dolphin": "Files",
    "nemo": "Files",
    "finder": "Finder",
    "explorer.exe": "File Explorer",
    # System
    "systemd": "System",
    "init": "System",
    "loginctl": "System",
    "pulseaudio": "Audio System",
    "pipewire": "Audio System",
    "pipewire-pulse": "Audio System",
    "wireplumber": "Audio System",
    "xorg": "Display Server",
    "xwayland": "Display Server",
    "gnome-shell": "Desktop",
    "plasmashell": "Desktop",
    "xfce4-panel": "Desktop Panel",
    "xfdesktop": "Desktop",
    "xfwm4": "Window Manager",
    "xfsettingsd": "System Settings",
    "xfconfd": "System Settings",
    "xfce4-session": "Desktop Session",
    "kwin": "Window Manager",
    "mutter": "Window Manager",
    "compiz": "Window Manager",
    "dwm": "Window Manager",
    "i3": "Window Manager",
    "sway": "Window Manager",
    # Version control
    "git": "Git",
    "git.exe": "Git",
    "gh": "GitHub CLI",
    # Docker/containers
    "docker": "Docker",
    "docker.exe": "Docker",
    "containerd": "Docker Engine",
    "dockerd": "Docker Engine",
    "podman": "Podman",
    # Node/Python/Java
    "node": "Node.js App",
    "python": "Python App",
    "python3": "Python App",
    "java": "Java App",
    "java.exe": "Java App",
    "ruby": "Ruby App",
    "go": "Go App",
    "rustc": "Rust Compiler",
    "cargo": "Rust Build",
    # Network
    "ssh": "SSH Connection",
    "ssh-agent": "SSH Agent",
    "nginx": "Web Server",
    "apache2": "Web Server",
    "httpd": "Web Server",
    "uvicorn": "Python Server",
    # System utilities (suppress noise)
    "dbus-daemon": "System Bus",
    "dbus-launch": "System Bus",
    "at-spi-bus-launcher": "Accessibility",
    "at-spi2-registryd": "Accessibility",
    "gpg-agent": "Security Agent",
    "polkitd": "Security",
    "gvfsd": "File System",
    "gvfs-udisks2-volume-monitor": "Disk Monitor",
    "udisksd": "Disk Manager",
    "networkmanager": "Network Manager",
    "nm-applet": "Network",
    "wpa_supplicant": "WiFi",
    "bluetoothd": "Bluetooth",
    "cups": "Printing",
    "cupsd": "Printing",
    "cron": "Scheduler",
    "crond": "Scheduler",
    "chrome_crashpad_handler": "Chrome (background)",
    "nacl_helper": "Chrome (background)",
}

# Signal type → friendly description
_SIGNAL_LABELS: dict[str, str] = {
    "system_boot": "Device started",
    "system_shutdown": "Device shutting down",
    "resource_pulse": "System check",
    "process_snapshot": "App scan",
    "app_open": "App opened",
    "app_close": "App closed",
    "app_focus": "Switched to app",
    "network_flow": "Internet activity",
    "disk_io": "Storage activity",
    "input_cadence": "Typing activity",
    "display_state": "Screen state",
    "auth_login": "Logged in",
    "auth_register": "Account created",
    "predict_submit": "Prediction run",
    "project_create": "Project created",
    "assistant_profile_update": "Profile updated",
    "login_success": "Logged in",
    "register_success": "Account created",
    "live_state_view": "Dashboard viewed",
    "password_recovery_request": "Password recovery",
}

# Layer names → friendly descriptions
_LAYER_LABELS: dict[str, str] = {
    "bedrock": "Foundation",
    "suspension": "Active Patterns",
    "turbulent": "Changing Signals",
}

# Feature names → friendly descriptions
_FEATURE_LABELS: dict[str, str] = {
    "cpu_mean": "CPU Usage",
    "cpu_max": "Peak CPU",
    "memory_mean": "Memory Usage",
    "battery_mean": "Battery Level",
    "net_sent_bytes": "Data Uploaded",
    "net_recv_bytes": "Data Downloaded",
    "disk_read_bytes": "Files Read",
    "disk_write_bytes": "Files Written",
    "context_switches": "App Switching",
    "unique_processes": "Active Apps",
    "total_processes": "All Processes",
    "process_diversity": "App Variety",
    "app_opens": "Apps Opened",
    "app_closes": "Apps Closed",
    "n_signals": "Signal Count",
    "hour_of_day": "Time of Day",
    "day_of_week": "Day of Week",
    "minute_of_day": "Time (minutes)",
    "bucket_index": "Time Window",
}


def humanize_app(raw_name: str) -> str:
    """Convert a raw process name to a human-friendly app name."""
    name = str(raw_name or "").strip().lower()
    if not name:
        return "Unknown"

    # Direct lookup
    friendly = _APP_NAMES.get(name)
    if friendly:
        return friendly

    # Strip common suffixes
    for suffix in (".exe", ".sh", ".py", ".app", ".bin"):
        if name.endswith(suffix):
            stripped = name[:-len(suffix)]
            friendly = _APP_NAMES.get(stripped)
            if friendly:
                return friendly
            name = stripped

    # Strip path prefixes
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
        friendly = _APP_NAMES.get(name)
        if friendly:
            return friendly

    # Capitalize and clean
    clean = name.replace("-", " ").replace("_", " ").strip()
    if clean:
        return clean.title()

    return "Unknown"


def humanize_signal(signal_type: str) -> str:
    """Convert a raw signal type to a human-friendly label."""
    s = str(signal_type or "").strip().lower()
    return _SIGNAL_LABELS.get(s, s.replace("_", " ").title())


def humanize_feature(feature_name: str) -> str:
    """Convert a raw feature name to a human-friendly label."""
    f = str(feature_name or "").strip()

    # Direct lookup
    friendly = _FEATURE_LABELS.get(f)
    if friendly:
        return friendly

    # App feature columns: "app_chrome" → "Chrome Browser usage"
    if f.startswith("app_"):
        app_raw = f[4:]
        return humanize_app(app_raw) + " usage"

    # Clean up
    return f.replace("_", " ").title()


def humanize_layer(layer: str) -> str:
    """Convert a layer name to a human-friendly label."""
    return _LAYER_LABELS.get(str(layer or "").lower(), str(layer or "").title())


def humanize_apps_dict(raw_dict: dict[str, int]) -> dict[str, int]:
    """Convert a dict of raw_app_name → count to friendly names."""
    result: dict[str, int] = {}
    for raw, count in raw_dict.items():
        friendly = humanize_app(raw)
        result[friendly] = result.get(friendly, 0) + int(count)
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
