"""physml.screen_observer — Passive screen observation and activity logging.

Runs a background thread that:

* Takes periodic screenshots (default every 60 s)
* Records the active window title + application name
* Optionally asks Claude to describe screen content (requires API key)
* Builds a focus-time-per-app log
* Feeds observations into :class:`~physml.multimodal_ingester.MultiModalIngester`
  and :class:`~physml.vector_memory.VectorMemory`

No raw screenshots are stored permanently unless ``save_screenshots=True``
is set — only the text description is kept.

Usage::

    from physml.screen_observer import ScreenObserver

    observer = ScreenObserver(interval=30.0)
    observer.start()          # background thread
    # ...
    observer.stop()
    print(observer.focus_summary())
    # {"Code Editor": 3620, "Browser": 1200, ...}   (seconds)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class ScreenSnapshot:
    """One observation frame.

    Attributes
    ----------
    timestamp : float
        Unix timestamp.
    app_name : str
        Active application name (best-effort).
    window_title : str
        Active window title.
    description : str
        Plain-text description of screen content (LLM or OCR).
    screenshot_path : str or None
        Path to saved screenshot PNG, if ``save_screenshots=True``.
    """

    timestamp: float
    app_name: str = "unknown"
    window_title: str = ""
    description: str = ""
    screenshot_path: Optional[str] = None


class ScreenObserver:
    """Background screen activity monitor.

    Parameters
    ----------
    interval : float
        Seconds between observations (default 60).
    save_screenshots : bool
        Whether to persist PNG files to disk.
    screenshot_dir : str
        Directory for saved screenshots.
    llm_describe : bool
        Use Claude vision to describe screen content when API key is set.
    ingester : MultiModalIngester or None
        Feed text descriptions into the ingestion pipeline.
    vector_memory : VectorMemory or None
        Direct semantic store (used when ingester is None).
    on_snapshot : callable or None
        ``fn(ScreenSnapshot)`` called after each observation.
    """

    def __init__(
        self,
        interval: float = 60.0,
        save_screenshots: bool = False,
        screenshot_dir: str = "~/.mycelium/screenshots",
        llm_describe: bool = True,
        ingester: Any = None,
        vector_memory: Any = None,
        on_snapshot: Optional[Callable] = None,
    ) -> None:
        self.interval = interval
        self.save_screenshots = save_screenshots
        self.screenshot_dir = Path(screenshot_dir).expanduser()
        self.llm_describe = llm_describe
        self._ingester = ingester
        self._vm = vector_memory
        self._on_snapshot = on_snapshot

        self._snapshots: List[ScreenSnapshot] = []
        self._focus_time: Dict[str, float] = {}   # app_name → total seconds
        self._last_app: Optional[str] = None
        self._last_tick: float = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background observation loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ScreenObserver")
        self._thread.start()
        _logger.info("ScreenObserver started (interval=%.0fs)", self.interval)

    def stop(self) -> None:
        """Stop background observation loop."""
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        _logger.info("ScreenObserver stopped. %d snapshots collected.", len(self._snapshots))

    def observe_once(self) -> ScreenSnapshot:
        """Take a single observation synchronously and return the snapshot."""
        return self._take_snapshot()

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def snapshots(self) -> List[ScreenSnapshot]:
        return list(self._snapshots)

    def focus_summary(self) -> Dict[str, float]:
        """Return total focus-time seconds per application name."""
        return dict(self._focus_time)

    def top_apps(self, n: int = 5) -> List[tuple]:
        """Return top-n apps by focus time as [(app, seconds), ...]."""
        return sorted(self._focus_time.items(), key=lambda x: x[1], reverse=True)[:n]

    def recent_context(self, n: int = 5) -> str:
        """Return a plain-text summary of the last *n* observations."""
        snaps = self._snapshots[-n:]
        if not snaps:
            return "No screen observations yet."
        lines = []
        for s in snaps:
            ts = time.strftime("%H:%M:%S", time.localtime(s.timestamp))
            desc = s.description[:120] or s.window_title or "no description"
            lines.append(f"[{ts}] {s.app_name}: {desc}")
        return "\n".join(lines)

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "interval": self.interval,
            "snapshots": len(self._snapshots),
            "top_apps": self.top_apps(3),
            "screenshot_available": self._check_screenshot_available(),
            "llm_describe": self.llm_describe,
        }

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._take_snapshot()
            except Exception as exc:
                _logger.debug("ScreenObserver._loop error: %s", exc)
            self._stop_event.wait(self.interval)

    def _take_snapshot(self) -> ScreenSnapshot:
        now = time.time()
        snap = ScreenSnapshot(timestamp=now)

        # Active window info
        snap.app_name, snap.window_title = self._get_active_window()

        # Track focus time
        if self._last_app is not None and self._last_tick > 0:
            elapsed = now - self._last_tick
            self._focus_time[self._last_app] = self._focus_time.get(self._last_app, 0.0) + elapsed
        self._last_app = snap.app_name
        self._last_tick = now

        # Screenshot
        screenshot_bytes = self._take_screenshot()

        # Save to disk if requested
        if self.save_screenshots and screenshot_bytes:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            fname = f"snap_{int(now)}.png"
            fpath = self.screenshot_dir / fname
            try:
                fpath.write_bytes(screenshot_bytes)
                snap.screenshot_path = str(fpath)
            except Exception as exc:
                _logger.debug("Screenshot save error: %s", exc)

        # Describe via LLM or OCR
        if screenshot_bytes:
            snap.description = self._describe(screenshot_bytes, snap.window_title)

        # Feed into learning pipeline
        if snap.description:
            self._store_description(snap)

        # User callback
        if self._on_snapshot:
            try:
                self._on_snapshot(snap)
            except Exception as exc:
                _logger.debug("ScreenObserver on_snapshot callback error: %s", exc)

        self._snapshots.append(snap)
        return snap

    def _get_active_window(self) -> tuple[str, str]:
        """Return (app_name, window_title) best-effort across platforms."""
        import platform
        system = platform.system()

        if system == "Linux":
            return self._active_window_linux()
        elif system == "Darwin":
            return self._active_window_macos()
        elif system == "Windows":
            return self._active_window_windows()
        return "unknown", ""

    def _active_window_linux(self) -> tuple[str, str]:
        try:
            import subprocess
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=3,
            )
            title = result.stdout.strip()
            # Derive app name from title (last token after " — " or " - " separator)
            app = title.split(" — ")[-1].split(" - ")[-1].strip() or "unknown"
            return app[:40], title[:120]
        except Exception:
            pass
        try:
            import subprocess
            result = subprocess.run(
                ["xprop", "-id", "$(xprop -root _NET_ACTIVE_WINDOW | awk '{print $5}')", "WM_CLASS"],
                shell=True, capture_output=True, text=True, timeout=3,
            )
            parts = result.stdout.strip().split('"')
            app = parts[-2] if len(parts) >= 2 else "unknown"
            return app[:40], ""
        except Exception:
            return "unknown", ""

    def _active_window_macos(self) -> tuple[str, str]:
        try:
            import subprocess
            script = 'tell application "System Events" to get name of first application process whose frontmost is true'
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
            app = r.stdout.strip()
            return app[:40], ""
        except Exception:
            return "unknown", ""

    def _active_window_windows(self) -> tuple[str, str]:
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            return title[:40], title[:120]
        except Exception:
            return "unknown", ""

    def _take_screenshot(self) -> Optional[bytes]:
        try:
            import mss  # type: ignore
            import io
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                img = sct.grab(monitor)
                # Convert to PNG bytes
                from mss.tools import to_png
                return to_png(img.rgb, img.size)
        except Exception:
            pass
        try:
            import pyautogui  # type: ignore
            import io
            img = pyautogui.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def _describe(self, screenshot_bytes: bytes, window_title: str) -> str:
        """Generate a text description of the screenshot."""
        # Try Claude vision
        if self.llm_describe:
            desc = self._llm_describe(screenshot_bytes)
            if desc:
                return desc

        # Try OCR fallback
        desc = self._ocr_describe(screenshot_bytes)
        if desc:
            return f"[OCR] {desc[:300]}"

        return f"Active window: {window_title}" if window_title else ""

    def _llm_describe(self, screenshot_bytes: bytes) -> str:
        try:
            import base64
            from physml.llm.claude_client import ClaudeClient
            client = ClaudeClient()
            if not client.available:
                return ""
            b64 = base64.standard_b64encode(screenshot_bytes).decode()
            msg = client._client.messages.create(
                model=client.model,
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                        {"type": "text", "text": "Describe what the user is working on in one sentence."},
                    ],
                }],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            _logger.debug("LLM screen describe error: %s", exc)
            return ""

    def _ocr_describe(self, screenshot_bytes: bytes) -> str:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            import io
            img = Image.open(io.BytesIO(screenshot_bytes))
            text = pytesseract.image_to_string(img)
            # Return first 200 non-blank chars
            cleaned = " ".join(text.split())
            return cleaned[:200]
        except Exception:
            return ""

    def _store_description(self, snap: ScreenSnapshot) -> None:
        text = f"[{snap.app_name}] {snap.description}"
        meta = {
            "type": "screen_observation",
            "app": snap.app_name,
            "window": snap.window_title,
            "timestamp": snap.timestamp,
        }
        if self._ingester is not None:
            try:
                self._ingester.ingest(text, topic="screen_activity")
            except Exception as exc:
                _logger.debug("ScreenObserver ingester error: %s", exc)
        elif self._vm is not None:
            try:
                self._vm.add(text, metadata=meta)
            except Exception as exc:
                _logger.debug("ScreenObserver vm.add error: %s", exc)

    def _check_screenshot_available(self) -> bool:
        for mod in ("mss", "pyautogui"):
            try:
                __import__(mod)
                return True
            except ImportError:
                pass
        return False

    def __repr__(self) -> str:
        return (
            f"ScreenObserver(running={self._running}, "
            f"interval={self.interval}s, "
            f"snapshots={len(self._snapshots)})"
        )
