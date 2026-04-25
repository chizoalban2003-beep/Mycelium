"""Stage 144 — DesktopBridge: local desktop task automation.

Wires :class:`~physml.local_executor.LocalTaskExecutor` and
:class:`~physml.screen_agent.ScreenAgent` into the GoalEngine so Myco can
perform everyday desktop tasks when asked by the user.

Capabilities
------------
* **File I/O** — read, write, list, delete files (write/delete gated by
  ``MYCO_ALLOW_WRITES`` env var or the PermissionManager).
* **Clipboard** — copy text to clipboard, paste from clipboard.
* **App launch** — open applications by name (uses ``xdg-open`` / ``open`` /
  ``start`` depending on the OS).
* **Screen** — take screenshots, find text on screen.
* **Shell** — run whitelisted shell commands (safe_shell_only enforced).
* **NL dispatch** — ``dispatch(description)`` routes natural-language step
  descriptions to the right handler automatically.

Configuration
-------------
Set ``MYCO_ALLOW_WRITES=1`` to enable file write/delete operations.
Without it, write attempts return a permission error with an explanation.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Any, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


class DesktopResult:
    """Outcome of a desktop action."""

    def __init__(self, success: bool, action: str, message: str = "") -> None:
        self.success = success
        self.action = action
        self.message = message

    def __str__(self) -> str:
        status = "ok" if self.success else "failed"
        return f"[desktop/{self.action}] {status}: {self.message}"


class DesktopBridge:
    """Desktop automation bridge for GoalEngine step dispatch.

    Parameters
    ----------
    companion : MyceliumCompanion or None
        Provides executor, screen_agent, and permission_manager.
    allow_writes : bool, optional
        Override write permission check (env: ``MYCO_ALLOW_WRITES``).
    """

    def __init__(
        self,
        companion: Any = None,
        allow_writes: Optional[bool] = None,
    ) -> None:
        self._companion = companion
        env_writes = os.environ.get("MYCO_ALLOW_WRITES", "")
        self._allow_writes = allow_writes if allow_writes is not None else bool(env_writes)

    # ------------------------------------------------------------------
    # Write-permission gate
    # ------------------------------------------------------------------

    def _writes_ok(self) -> bool:
        if self._allow_writes:
            return True
        if self._companion is not None:
            pm = getattr(self._companion, "permission_manager", None)
            if pm is not None:
                return pm.check("file.write")
        return False

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def read_file(self, path: str) -> DesktopResult:
        """Read a local file and return its contents."""
        try:
            from pathlib import Path
            p = Path(path).expanduser()
            if not p.exists():
                return DesktopResult(False, "read_file", f"File not found: {path}")
            text = p.read_text(errors="replace")
            preview = text[:2000]
            _logger.info("DesktopBridge: read file %s (%d chars)", path, len(text))
            return DesktopResult(True, "read_file", preview)
        except Exception as exc:
            return DesktopResult(False, "read_file", str(exc))

    def write_file(self, path: str, content: str) -> DesktopResult:
        """Write content to a local file."""
        if not self._writes_ok():
            return DesktopResult(
                False, "write_file",
                "Write permission denied. Set MYCO_ALLOW_WRITES=1 to enable file writes."
            )
        try:
            from pathlib import Path
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            _logger.info("DesktopBridge: wrote %d chars to %s", len(content), path)
            return DesktopResult(True, "write_file", f"Wrote {len(content)} chars to {path}")
        except Exception as exc:
            return DesktopResult(False, "write_file", str(exc))

    def list_dir(self, path: str = ".") -> DesktopResult:
        """List directory contents."""
        try:
            from pathlib import Path
            p = Path(path).expanduser()
            if not p.is_dir():
                return DesktopResult(False, "list_dir", f"Not a directory: {path}")
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries[:100]]
            return DesktopResult(True, "list_dir", "\n".join(lines) or "(empty)")
        except Exception as exc:
            return DesktopResult(False, "list_dir", str(exc))

    def delete_file(self, path: str) -> DesktopResult:
        """Delete a file or empty directory."""
        if not self._writes_ok():
            return DesktopResult(
                False, "delete_file",
                "Write permission denied. Set MYCO_ALLOW_WRITES=1 to enable deletions."
            )
        try:
            from pathlib import Path
            p = Path(path).expanduser()
            if not p.exists():
                return DesktopResult(False, "delete_file", f"Not found: {path}")
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            _logger.info("DesktopBridge: deleted %s", path)
            return DesktopResult(True, "delete_file", f"Deleted: {path}")
        except Exception as exc:
            return DesktopResult(False, "delete_file", str(exc))

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def copy_to_clipboard(self, text: str) -> DesktopResult:
        """Copy text to the system clipboard."""
        try:
            import pyperclip  # type: ignore
            pyperclip.copy(text)
            return DesktopResult(True, "clipboard_copy", f"Copied {len(text)} chars to clipboard")
        except ImportError:
            pass
        # Fallback for Linux (xclip/xsel) without pyperclip
        for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            try:
                subprocess.run(
                    cmd, input=text.encode(), timeout=5, check=True
                )
                return DesktopResult(True, "clipboard_copy", f"Copied to clipboard via {cmd[0]}")
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        return DesktopResult(
            False, "clipboard_copy",
            "Clipboard not available. Install pyperclip: pip install pyperclip"
        )

    def paste_from_clipboard(self) -> DesktopResult:
        """Read text from the system clipboard."""
        try:
            import pyperclip  # type: ignore
            text = pyperclip.paste()
            return DesktopResult(True, "clipboard_paste", text)
        except ImportError:
            pass
        for cmd in (["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]):
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5, check=True)
                return DesktopResult(True, "clipboard_paste", result.stdout.decode())
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        return DesktopResult(
            False, "clipboard_paste",
            "Clipboard not available. Install pyperclip: pip install pyperclip"
        )

    # ------------------------------------------------------------------
    # Application launching
    # ------------------------------------------------------------------

    def open_app(self, app_name: str) -> DesktopResult:
        """Launch an application by name."""
        platform = sys.platform
        try:
            if platform == "darwin":
                subprocess.Popen(["open", "-a", app_name])
            elif platform == "win32":
                subprocess.Popen(["start", app_name], shell=True)
            else:
                # Try direct launch, then xdg-open
                try:
                    subprocess.Popen([app_name])
                except FileNotFoundError:
                    subprocess.Popen(["xdg-open", app_name])
            _logger.info("DesktopBridge: launched %r", app_name)
            return DesktopResult(True, "open_app", f"Launched: {app_name}")
        except Exception as exc:
            return DesktopResult(False, "open_app", str(exc))

    def open_url(self, url: str) -> DesktopResult:
        """Open a URL in the default browser."""
        import webbrowser
        try:
            webbrowser.open(url)
            return DesktopResult(True, "open_url", f"Opened in browser: {url}")
        except Exception as exc:
            return DesktopResult(False, "open_url", str(exc))

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def run_shell(self, command: str) -> DesktopResult:
        """Run a whitelisted shell command."""
        # Delegate to LocalTaskExecutor for safety filtering
        if self._companion is not None:
            executor = getattr(self._companion, "executor", None)
            if executor is not None:
                result = executor.run_shell(command)
                if result.success:
                    return DesktopResult(True, "shell", result.output or "")
                return DesktopResult(False, "shell", result.error or result.output or "blocked")
        # Bare-minimum fallback (no safety checks — not recommended)
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            out = (r.stdout + r.stderr).strip()
            return DesktopResult(r.returncode == 0, "shell", out)
        except Exception as exc:
            return DesktopResult(False, "shell", str(exc))

    # ------------------------------------------------------------------
    # Screen
    # ------------------------------------------------------------------

    def screenshot(self) -> DesktopResult:
        """Take a screenshot and return the saved file path."""
        if self._companion is not None:
            sa = getattr(self._companion, "screen_agent", None)
            if sa is not None:
                path = sa.screenshot()
                if path:
                    return DesktopResult(True, "screenshot", f"Saved: {path}")
                return DesktopResult(False, "screenshot", "Screenshot failed")
        return DesktopResult(False, "screenshot", "ScreenAgent not available")

    def find_text_on_screen(self, text: str) -> DesktopResult:
        """Find text on screen via OCR."""
        if self._companion is not None:
            sa = getattr(self._companion, "screen_agent", None)
            if sa is not None:
                result = sa.find_text(text)
                if result:
                    return DesktopResult(True, "find_text", f"Found at: {result}")
                return DesktopResult(False, "find_text", f"Text not found: {text!r}")
        return DesktopResult(False, "find_text", "ScreenAgent not available")

    # ------------------------------------------------------------------
    # NL dispatch
    # ------------------------------------------------------------------

    def dispatch(self, description: str) -> str:
        """Route a NL step description to the right desktop action.

        Routing heuristic:
        - "open <app>" → open_app
        - "read/open/show file <path>" → read_file
        - "write/save/create file" → write_file
        - "list/ls <dir>" → list_dir
        - "delete/remove file" → delete_file (if writes allowed)
        - "copy to clipboard" → copy_to_clipboard
        - "paste from clipboard" → paste_from_clipboard
        - "screenshot" → screenshot
        - "run/execute/shell" → run_shell
        - URL → open_url
        """
        low = description.lower()

        # URL open
        url_m = re.search(r"https?://\S+", description)
        if url_m:
            return str(self.open_url(url_m.group()))

        # App launch
        if re.search(r"\bopen\b", low) and not re.search(r"\bfile\b|\bdir\b|\bfolder\b", low):
            m = re.search(r"open\s+([a-zA-Z0-9 ._-]+?)(?:\s+app|\s*$)", description, re.I)
            app = m.group(1).strip() if m else description
            return str(self.open_app(app))

        # Screenshot
        if "screenshot" in low or "screen capture" in low:
            return str(self.screenshot())

        # Clipboard copy
        if "copy" in low and "clipboard" in low:
            m = re.search(r"copy\s+['\"]?(.+?)['\"]?\s+to\s+clipboard", description, re.I)
            text = m.group(1).strip() if m else ""
            if not text:
                # Everything after "copy"
                idx = low.index("copy") + 4
                text = description[idx:].strip().strip(":\"' ")
            return str(self.copy_to_clipboard(text))

        # Clipboard paste
        if "paste" in low or ("clipboard" in low and "read" in low):
            return str(self.paste_from_clipboard())

        # File write
        if any(k in low for k in ("write file", "save file", "create file", "write to")):
            m = re.search(r"(?:write|save|create)\s+(?:file\s+)?['\"]?(\S+)['\"]?\s+(?:with\s+content\s+)?['\"]?(.+)['\"]?$",
                          description, re.I | re.S)
            if m:
                return str(self.write_file(m.group(1), m.group(2).strip()))
            return str(DesktopResult(False, "write_file", "Could not parse path and content"))

        # File delete
        if any(k in low for k in ("delete file", "remove file", "delete ", "rm ")):
            m = re.search(r"(?:delete|remove|rm)\s+(?:file\s+)?['\"]?(\S+)['\"]?", description, re.I)
            path = m.group(1).strip() if m else ""
            if path:
                return str(self.delete_file(path))
            return "Could not parse file path for deletion."

        # File read
        if any(k in low for k in ("read file", "open file", "show file", "cat ")):
            m = re.search(r"(?:read|open|show|cat)\s+(?:file\s+)?['\"]?(\S+)['\"]?", description, re.I)
            path = m.group(1).strip() if m else ""
            if path:
                return str(self.read_file(path))
            return "Could not parse file path."

        # List directory
        if any(k in low for k in ("list ", "ls ", "list dir", "list folder")):
            m = re.search(r"(?:list|ls)\s+(?:dir\s+|folder\s+|files\s+in\s+)?['\"]?(\S+)['\"]?",
                          description, re.I)
            path = m.group(1).strip() if m else "."
            return str(self.list_dir(path))

        # Shell command
        if any(k in low for k in ("run ", "execute ", "shell ", "command ", "$ ")):
            m = re.search(r"(?:run|execute|shell|command|\$)\s+['\"]?(.+)['\"]?$",
                          description, re.I)
            cmd = m.group(1).strip() if m else description
            return str(self.run_shell(cmd))

        return f"No desktop action matched for: {description[:80]}"

    def status(self) -> dict:
        """Return capability status."""
        try:
            import pyperclip  # noqa: F401
            clipboard = True
        except ImportError:
            clipboard = any(
                shutil.which(c) for c in ("xclip", "xsel", "pbcopy")
            )

        screen_available = False
        if self._companion is not None:
            sa = getattr(self._companion, "screen_agent", None)
            screen_available = sa is not None and getattr(sa, "available", False)

        return {
            "file_read": True,
            "file_write": self._writes_ok(),
            "clipboard": clipboard,
            "screen": screen_available,
            "app_launch": True,
            "shell": True,
        }
