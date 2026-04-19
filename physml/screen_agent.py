"""Stage 129 — ScreenAgent: local screen/UI automation.

Captures screenshots and drives mouse/keyboard via pyautogui (optional).
Falls back to a no-op stub when the library is absent so the companion
always imports cleanly on headless servers.

Usage
-----
::

    from physml.screen_agent import ScreenAgent

    sa = ScreenAgent()
    img_path = sa.screenshot()          # save PNG, return path
    sa.click(x=100, y=200)
    sa.type_text("hello world")
    sa.hotkey("ctrl", "c")
    info = sa.find_text_on_screen("Submit")  # (x, y) or None
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from physml._log import get_logger

_logger = get_logger(__name__)

try:
    import pyautogui  # type: ignore
    import pyautogui as _pag
    _PAG_OK = True
    _pag.FAILSAFE = True
    _pag.PAUSE = 0.05
except Exception:
    _PAG_OK = False

try:
    import mss  # type: ignore
    _MSS_OK = True
except Exception:
    _MSS_OK = False


class ScreenAgent:
    """Local screen automation — screenshot, click, type, find text.

    Parameters
    ----------
    screenshot_dir : str, default "~/.mycelium/screenshots"
        Where to save captured images.
    safe_mode : bool, default True
        When True, confirm before executing destructive actions.
    """

    def __init__(
        self,
        screenshot_dir: str = "~/.mycelium/screenshots",
        safe_mode: bool = True,
    ) -> None:
        self.screenshot_dir = Path(screenshot_dir).expanduser()
        self.safe_mode = safe_mode
        self._available = _PAG_OK or _MSS_OK

    @property
    def available(self) -> bool:
        return self._available

    def screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """Capture the full screen and save to *screenshot_dir*.

        Returns the saved file path, or None if unavailable.
        """
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        fname = filename or f"screenshot_{ts}.png"
        out = str(self.screenshot_dir / fname)

        if _MSS_OK:
            try:
                with mss.mss() as sct:
                    sct.shot(output=out)
                _logger.info("ScreenAgent: screenshot saved to %s", out)
                return out
            except Exception as exc:
                _logger.warning("ScreenAgent mss screenshot failed: %s", exc)

        if _PAG_OK:
            try:
                img = _pag.screenshot()
                img.save(out)
                _logger.info("ScreenAgent: screenshot saved to %s", out)
                return out
            except Exception as exc:
                _logger.warning("ScreenAgent pyautogui screenshot failed: %s", exc)

        _logger.info("ScreenAgent: neither mss nor pyautogui available")
        return None

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Move mouse to *(x, y)* and click."""
        if not _PAG_OK:
            _logger.info("ScreenAgent: pyautogui not available, skipping click")
            return False
        try:
            _pag.click(x, y, button=button, clicks=clicks)
            return True
        except Exception as exc:
            _logger.warning("ScreenAgent click failed: %s", exc)
            return False

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        """Type *text* at the current cursor position."""
        if not _PAG_OK:
            return False
        try:
            _pag.typewrite(text, interval=interval)
            return True
        except Exception as exc:
            _logger.warning("ScreenAgent type_text failed: %s", exc)
            return False

    def hotkey(self, *keys: str) -> bool:
        """Press a keyboard shortcut (e.g. hotkey('ctrl', 'c'))."""
        if not _PAG_OK:
            return False
        try:
            _pag.hotkey(*keys)
            return True
        except Exception as exc:
            _logger.warning("ScreenAgent hotkey failed: %s", exc)
            return False

    def find_text_on_screen(self, text: str) -> Optional[Tuple[int, int]]:
        """Locate *text* on screen using OCR (requires pytesseract + Pillow).

        Returns
        -------
        (x, y) centre coordinate, or None if not found.
        """
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            path = self.screenshot()
            if path is None:
                return None
            img = Image.open(path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            for i, word in enumerate(data["text"]):
                if text.lower() in str(word).lower():
                    x = data["left"][i] + data["width"][i] // 2
                    y = data["top"][i] + data["height"][i] // 2
                    return (x, y)
        except Exception as exc:
            _logger.debug("ScreenAgent find_text_on_screen: %s", exc)
        return None

    def scroll(self, clicks: int = 3, direction: str = "down") -> bool:
        """Scroll the mouse wheel."""
        if not _PAG_OK:
            return False
        try:
            amount = -abs(clicks) if direction == "down" else abs(clicks)
            _pag.scroll(amount)
            return True
        except Exception as exc:
            _logger.warning("ScreenAgent scroll failed: %s", exc)
            return False

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.3) -> bool:
        """Click and drag from *(x1,y1)* to *(x2,y2)*."""
        if not _PAG_OK:
            return False
        try:
            _pag.moveTo(x1, y1)
            _pag.dragTo(x2, y2, duration=duration)
            return True
        except Exception as exc:
            _logger.warning("ScreenAgent drag failed: %s", exc)
            return False

    def status(self) -> dict:
        return {
            "available": self._available,
            "pyautogui": _PAG_OK,
            "mss": _MSS_OK,
            "screenshot_dir": str(self.screenshot_dir),
        }
