"""Stage 130 — BrowserAgent: local browser automation via Playwright.

Controls a real Chromium/Firefox browser session entirely on-device —
no cloud, no data leaves the machine.  Falls back to a no-op stub when
Playwright is not installed.

Installation
------------
::

    pip install playwright
    playwright install chromium

Usage
-----
::

    from physml.browser_agent import BrowserAgent

    ba = BrowserAgent(headless=True)
    ba.open()
    ba.goto("https://example.com")
    text = ba.get_text()
    ba.click_text("Learn More")
    ba.fill("#email", "user@example.com")
    result = ba.extract_links()
    ba.close()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

try:
    from playwright.sync_api import sync_playwright, Browser, Page  # type: ignore
    _PW_OK = True
except Exception:
    _PW_OK = False
    Browser = Any  # type: ignore
    Page = Any  # type: ignore


class BrowserAgent:
    """Local browser automation with a clean companion-friendly API.

    Parameters
    ----------
    headless : bool, default True
        Run the browser without a visible window.
    browser_type : str, default "chromium"
        Browser engine: ``"chromium"``, ``"firefox"``, or ``"webkit"``.
    timeout_ms : int, default 10000
        Default element-wait timeout in milliseconds.
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        timeout_ms: int = 10_000,
    ) -> None:
        self.headless = headless
        self.browser_type = browser_type
        self.timeout_ms = timeout_ms
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._available = _PW_OK

    @property
    def available(self) -> bool:
        return _PW_OK

    @property
    def page(self) -> Any:
        return self._page

    def open(self) -> bool:
        """Launch the browser and open a blank page."""
        if not _PW_OK:
            _logger.info("BrowserAgent: playwright not installed")
            return False
        try:
            self._pw = sync_playwright().start()
            launcher = getattr(self._pw, self.browser_type)
            self._browser = launcher.launch(headless=self.headless)
            self._page = self._browser.new_page()
            self._page.set_default_timeout(self.timeout_ms)
            _logger.info("BrowserAgent: %s browser opened", self.browser_type)
            return True
        except Exception as exc:
            _logger.warning("BrowserAgent open failed: %s", exc)
            return False

    def close(self) -> None:
        """Close the browser session."""
        for obj in (self._page, self._browser, self._pw):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._page = self._browser = self._pw = None

    def goto(self, url: str, wait_until: str = "domcontentloaded") -> bool:
        """Navigate to *url*."""
        if self._page is None:
            return False
        try:
            self._page.goto(url, wait_until=wait_until)
            return True
        except Exception as exc:
            _logger.warning("BrowserAgent goto failed: %s", exc)
            return False

    def get_text(self, selector: str = "body") -> str:
        """Return the visible text content of *selector*."""
        if self._page is None:
            return ""
        try:
            return self._page.inner_text(selector)
        except Exception:
            return ""

    def get_html(self, selector: str = "body") -> str:
        """Return inner HTML of *selector*."""
        if self._page is None:
            return ""
        try:
            return self._page.inner_html(selector)
        except Exception:
            return ""

    def click(self, selector: str) -> bool:
        """Click the element matching *selector*."""
        if self._page is None:
            return False
        try:
            self._page.click(selector)
            return True
        except Exception as exc:
            _logger.warning("BrowserAgent click failed: %s", exc)
            return False

    def click_text(self, text: str) -> bool:
        """Click the first element whose visible text matches *text*."""
        return self.click(f"text={text}")

    def fill(self, selector: str, value: str) -> bool:
        """Fill an input field identified by *selector*."""
        if self._page is None:
            return False
        try:
            self._page.fill(selector, value)
            return True
        except Exception as exc:
            _logger.warning("BrowserAgent fill failed: %s", exc)
            return False

    def screenshot(self, path: str = "/tmp/browser_screenshot.png") -> Optional[str]:
        """Take a screenshot of the current page."""
        if self._page is None:
            return None
        try:
            self._page.screenshot(path=path)
            return path
        except Exception as exc:
            _logger.warning("BrowserAgent screenshot failed: %s", exc)
            return None

    def extract_links(self) -> List[Dict[str, str]]:
        """Return all anchor tags as list of {text, href} dicts."""
        if self._page is None:
            return []
        try:
            links = self._page.eval_on_selector_all(
                "a[href]",
                "els => els.map(el => ({text: el.innerText.trim(), href: el.href}))",
            )
            return [l for l in links if l.get("href")]
        except Exception:
            return []

    def current_url(self) -> str:
        """Return the current page URL."""
        if self._page is None:
            return ""
        try:
            return self._page.url
        except Exception:
            return ""

    def title(self) -> str:
        """Return the current page title."""
        if self._page is None:
            return ""
        try:
            return self._page.title()
        except Exception:
            return ""

    def wait_for_selector(self, selector: str, timeout_ms: Optional[int] = None) -> bool:
        """Wait until *selector* appears on the page."""
        if self._page is None:
            return False
        try:
            self._page.wait_for_selector(
                selector, timeout=timeout_ms or self.timeout_ms
            )
            return True
        except Exception:
            return False

    def run_js(self, script: str) -> Any:
        """Execute arbitrary JavaScript and return the result."""
        if self._page is None:
            return None
        try:
            return self._page.evaluate(script)
        except Exception as exc:
            _logger.warning("BrowserAgent run_js failed: %s", exc)
            return None

    def fetch_text(self, url: str) -> str:
        """Open *url*, extract visible text, close — convenience one-shot."""
        opened = self.open()
        if not opened:
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=10) as r:
                    raw = r.read().decode("utf-8", errors="replace")
                import re
                return re.sub(r"<[^>]+>", " ", raw)[:5000]
            except Exception:
                return ""
        self.goto(url)
        text = self.get_text()
        self.close()
        return text

    def status(self) -> dict:
        return {
            "available": _PW_OK,
            "open": self._page is not None,
            "url": self.current_url(),
        }
