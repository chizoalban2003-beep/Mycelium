"""Stage 133 — Notifier: proactive desktop/OS notifications.

Sends native desktop notifications via ``plyer`` (cross-platform) with
a silent fallback for headless environments.  Used by the companion to
push proactive alerts: model drift, completed tasks, reminders.

Usage
-----
::

    from physml.notifier import Notifier

    n = Notifier(app_name="Myco")
    n.send("Training complete", "Your model achieved 94% accuracy.")
    n.remind("Review your sales forecast", delay_seconds=3600)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

try:
    from plyer import notification as _plyer_notif  # type: ignore
    _PLYER_OK = True
except Exception:
    _PLYER_OK = False


class Notifier:
    """Cross-platform desktop notification sender.

    Parameters
    ----------
    app_name : str, default "Mycelium"
        Name shown in the notification header.
    app_icon : str, optional
        Path to an icon file.
    timeout : int, default 5
        Notification display duration in seconds.
    """

    def __init__(
        self,
        app_name: str = "Mycelium",
        app_icon: str = "",
        timeout: int = 5,
    ) -> None:
        self.app_name = app_name
        self.app_icon = app_icon
        self.timeout = timeout
        self._log: List[dict] = []

    @property
    def available(self) -> bool:
        return _PLYER_OK

    def send(
        self,
        title: str,
        message: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """Send a desktop notification immediately.

        Parameters
        ----------
        title : str
            Notification headline.
        message : str
            Body text.
        timeout : int, optional
            Override instance default timeout.

        Returns
        -------
        bool
            True if the notification was sent, False if plyer unavailable.
        """
        entry = {
            "title": title,
            "message": message,
            "timestamp": time.time(),
            "sent": False,
        }
        self._log.append(entry)

        if _PLYER_OK:
            try:
                _plyer_notif.notify(
                    title=title,
                    message=message,
                    app_name=self.app_name,
                    app_icon=self.app_icon,
                    timeout=timeout or self.timeout,
                )
                entry["sent"] = True
                _logger.info("Notifier: sent '%s'", title)
                return True
            except Exception as exc:
                _logger.warning("Notifier: plyer send failed: %s", exc)
        else:
            _logger.info("Notifier [%s]: %s — %s", self.app_name, title, message)

        return entry["sent"]

    def remind(
        self,
        message: str,
        delay_seconds: float = 60,
        title: str = "Myco Reminder",
    ) -> None:
        """Schedule a notification to fire after *delay_seconds*."""
        def _fire() -> None:
            time.sleep(delay_seconds)
            self.send(title, message)

        t = threading.Thread(target=_fire, daemon=True)
        t.start()
        _logger.info(
            "Notifier: reminder scheduled in %.0fs — '%s'",
            delay_seconds, message,
        )

    def send_alert(self, message: str) -> bool:
        """Send a high-priority alert (prefixed title)."""
        return self.send("⚠ Myco Alert", message)

    def send_success(self, message: str) -> bool:
        """Send a success notification."""
        return self.send("✓ Myco", message)

    def history(self, n: int = 20) -> List[dict]:
        """Return the last *n* notifications."""
        return list(self._log[-n:])

    def status(self) -> dict:
        return {
            "available": _PLYER_OK,
            "app_name": self.app_name,
            "total_sent": sum(1 for e in self._log if e["sent"]),
            "total_logged": len(self._log),
        }
