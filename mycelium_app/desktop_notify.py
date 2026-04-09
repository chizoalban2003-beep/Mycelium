"""Desktop notification daemon — surfaces nudges as native OS notifications.

Supports:
    - Linux: notify-send (libnotify)
    - macOS: osascript (AppleScript)
    - Windows: PowerShell toast (fallback)
    - Browser: Notification API (via the web UI)

The daemon polls for unseen nudges and dispatches them as desktop notifications.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from mycelium_app.models import NexusNudge
from mycelium_app.humanizer import humanize_signal


def _detect_platform() -> str:
    system = platform.system().lower()
    if "linux" in system:
        return "linux"
    elif "darwin" in system:
        return "macos"
    elif "windows" in system:
        return "windows"
    return "unknown"


def send_desktop_notification(title: str, message: str, *, urgency: str = "normal") -> bool:
    """Send a native desktop notification. Returns True if successful."""
    plat = _detect_platform()

    if plat == "linux":
        try:
            cmd = ["notify-send", "--app-name=Myco", f"--urgency={urgency}", title, message]
            subprocess.run(cmd, timeout=5, capture_output=True)
            return True
        except Exception:
            return False

    elif plat == "macos":
        try:
            script = f'display notification "{message}" with title "{title}" subtitle "Myco"'
            subprocess.run(["osascript", "-e", script], timeout=5, capture_output=True)
            return True
        except Exception:
            return False

    elif plat == "windows":
        try:
            ps_script = (
                f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
                f'ContentType = WindowsRuntime] > $null; '
                f'$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0); '
                f'$text = $template.GetElementsByTagName("text"); '
                f'$text[0].AppendChild($template.CreateTextNode("{title}")); '
                f'$text[1].AppendChild($template.CreateTextNode("{message}")); '
                f'$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Myco"); '
                f'$notifier.Show([Windows.UI.Notifications.ToastNotification]::new($template))'
            )
            subprocess.run(["powershell", "-Command", ps_script], timeout=10, capture_output=True)
            return True
        except Exception:
            return False

    return False


def dispatch_pending_notifications(
    session: Session,
    *,
    user_id: int,
    max_per_tick: int = 3,
    lookback_minutes: int = 30,
) -> int:
    """Check for unseen nudges and send desktop notifications.

    Returns the number of notifications sent.
    """
    since = datetime.utcnow() - timedelta(minutes=lookback_minutes)

    nudges = session.exec(
        select(NexusNudge)
        .where(
            NexusNudge.created_by_user_id == int(user_id),
            NexusNudge.created_at >= since,
            NexusNudge.seen_at.is_(None),
        )
        .order_by(NexusNudge.created_at.desc())
        .limit(max_per_tick)
    ).all()

    sent = 0
    for nudge in nudges:
        title = str(nudge.title or "Myco").strip()
        message = str(nudge.message or "").strip()
        if not message:
            continue

        kind = str(nudge.kind or "").lower()
        urgency = "critical" if "anomal" in kind else "normal"

        if send_desktop_notification(title, message[:200], urgency=urgency):
            nudge.seen_at = datetime.utcnow()
            session.add(nudge)
            sent += 1

    if sent > 0:
        session.commit()

    return sent
