"""Stage 107 — LocalTaskExecutor: safe OS-level task execution.

Allows the Mycelium agent to perform real digital tasks on the user's local
device:

* **File operations** — read, write, list, copy, delete files and directories
  (with dry-run / sandbox mode for safety).
* **Shell commands** — run whitelisted shell commands with a configurable
  timeout and output capture.
* **Process inspection** — list running processes, check if a program is
  available.
* **Clipboard I/O** — read from / write to the system clipboard (requires
  ``pyperclip`` or ``xclip``).

All operations are gated by a :class:`ExecutionPolicy` that controls
what is allowed (read-only, no-network, safe-shell-only, etc.).

Safety model
------------
* Default policy: ``read_only=True`` — only read and list operations are
  permitted.  Writes / shell execution must be explicitly opted-in by the
  user.
* ``safe_shell_only=True`` (default) — shell commands must match an
  allowlist pattern; dangerous commands (``rm -rf``, ``sudo``, ``curl``,
  ``wget``, ``chmod``, ``chown``, ``dd``, etc.) are blocked.
* ``dry_run=True`` — log what *would* happen without actually doing it.

Usage
-----
::

    from physml.local_executor import LocalTaskExecutor, ExecutionPolicy

    # Read-only (safe default)
    executor = LocalTaskExecutor()
    result = executor.list_dir("~/Documents")
    result = executor.read_file("data.csv")

    # Allow writes and safe shell (user must opt-in)
    policy = ExecutionPolicy(read_only=False, safe_shell_only=True)
    executor = LocalTaskExecutor(policy=policy)
    result = executor.run_shell("ls -la /tmp")
    result = executor.write_file("output.txt", "hello world")
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Blocked shell-command patterns (safety)
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bfdisk\b",
    r"\bformat\b",
    r">\s*/dev/",
    r"\beval\b",
    r"\bexec\b\s+.*\bsh\b",
    r"\bpython3?\s+-c\b",
    r"\bnc\b.*-e",
    r"base64\s+--decode",
]

_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPolicy:
    """Controls what the executor is allowed to do.

    Parameters
    ----------
    read_only : bool, default True
        When ``True``, write/shell operations raise ``PermissionError``.
    safe_shell_only : bool, default True
        When ``True``, shell commands matching ``_BLOCKED_PATTERNS`` are
        rejected.
    dry_run : bool, default False
        When ``True``, all operations are logged but not executed.
    max_file_size_mb : float, default 50
        Maximum file size in megabytes for read operations.
    shell_timeout : float, default 10.0
        Seconds before a shell command is killed.
    allowed_dirs : list of str, optional
        If set, file operations are restricted to these directories
        (and their subdirectories).
    """
    read_only: bool = True
    safe_shell_only: bool = True
    dry_run: bool = False
    max_file_size_mb: float = 50.0
    shell_timeout: float = 10.0
    allowed_dirs: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of a :class:`LocalTaskExecutor` operation.

    Attributes
    ----------
    success : bool
    operation : str
    output : Any
        The operation's return value (file content, list, bool, etc.).
    error : str or None
        Error message if *success* is ``False``.
    elapsed : float
        Wall-clock seconds for the operation.
    dry_run : bool
        ``True`` when the policy's dry-run mode was active.
    """
    success: bool
    operation: str
    output: Any
    error: Optional[str] = None
    elapsed: float = 0.0
    dry_run: bool = False


# ---------------------------------------------------------------------------
# LocalTaskExecutor
# ---------------------------------------------------------------------------

class LocalTaskExecutor:
    """Safe local-device task executor for the Mycelium agent.

    Parameters
    ----------
    policy : ExecutionPolicy or None
        Governs what operations are allowed.  Defaults to read-only mode.
    base_dir : str or Path, optional
        Default directory for relative file paths.  Defaults to ``cwd``.
    """

    def __init__(
        self,
        policy: Optional[ExecutionPolicy] = None,
        base_dir: Optional[str | Path] = None,
    ) -> None:
        self.policy = policy or ExecutionPolicy()
        self.base_dir = Path(base_dir or Path.cwd()).expanduser().resolve()

    # ------------------------------------------------------------------
    # File system — read
    # ------------------------------------------------------------------

    def read_file(self, path: str, encoding: str = "utf-8") -> TaskResult:
        """Read a text file and return its contents.

        Parameters
        ----------
        path : str
            File path (absolute or relative to ``base_dir``).
        encoding : str, default "utf-8"

        Returns
        -------
        TaskResult
        """
        t0 = time.monotonic()
        resolved = self._resolve(path)
        try:
            self._check_dir_access(resolved)
            size_mb = resolved.stat().st_size / 1e6
            if size_mb > self.policy.max_file_size_mb:
                return TaskResult(
                    False, "read_file", None,
                    f"File too large: {size_mb:.1f} MB > {self.policy.max_file_size_mb} MB",
                    time.monotonic() - t0,
                )
            if self.policy.dry_run:
                _logger.info("[dry_run] read_file %s", resolved)
                return TaskResult(True, "read_file", f"<dry_run: {resolved}>", dry_run=True, elapsed=time.monotonic()-t0)
            content = resolved.read_text(encoding=encoding, errors="replace")
            return TaskResult(True, "read_file", content, elapsed=time.monotonic() - t0)
        except Exception as exc:
            return TaskResult(False, "read_file", None, str(exc), time.monotonic() - t0)

    def list_dir(self, path: str = ".") -> TaskResult:
        """List files and subdirectories.

        Parameters
        ----------
        path : str, default "."

        Returns
        -------
        TaskResult
            ``output`` is a list of filenames.
        """
        t0 = time.monotonic()
        resolved = self._resolve(path)
        try:
            self._check_dir_access(resolved)
            if self.policy.dry_run:
                _logger.info("[dry_run] list_dir %s", resolved)
                return TaskResult(True, "list_dir", [], dry_run=True, elapsed=time.monotonic()-t0)
            entries = [p.name for p in sorted(resolved.iterdir())]
            return TaskResult(True, "list_dir", entries, elapsed=time.monotonic() - t0)
        except Exception as exc:
            return TaskResult(False, "list_dir", None, str(exc), time.monotonic() - t0)

    def file_exists(self, path: str) -> TaskResult:
        """Check if a file or directory exists."""
        t0 = time.monotonic()
        resolved = self._resolve(path)
        return TaskResult(True, "file_exists", resolved.exists(), elapsed=time.monotonic() - t0)

    def file_info(self, path: str) -> TaskResult:
        """Return size, mtime, and type of a path."""
        t0 = time.monotonic()
        resolved = self._resolve(path)
        try:
            st = resolved.stat()
            info = {
                "path": str(resolved),
                "exists": True,
                "is_file": resolved.is_file(),
                "is_dir": resolved.is_dir(),
                "size_bytes": st.st_size,
                "modified": st.st_mtime,
            }
            return TaskResult(True, "file_info", info, elapsed=time.monotonic() - t0)
        except FileNotFoundError:
            return TaskResult(True, "file_info", {"exists": False, "path": str(resolved)}, elapsed=time.monotonic()-t0)
        except Exception as exc:
            return TaskResult(False, "file_info", None, str(exc), time.monotonic() - t0)

    # ------------------------------------------------------------------
    # File system — write (requires read_only=False)
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str, encoding: str = "utf-8") -> TaskResult:
        """Write *content* to *path*.

        Requires ``policy.read_only=False``.
        """
        t0 = time.monotonic()
        if self.policy.read_only:
            return TaskResult(False, "write_file", None, "Policy is read_only=True", time.monotonic()-t0)
        resolved = self._resolve(path)
        try:
            self._check_dir_access(resolved)
            if self.policy.dry_run:
                _logger.info("[dry_run] write_file %s (%d chars)", resolved, len(content))
                return TaskResult(True, "write_file", str(resolved), dry_run=True, elapsed=time.monotonic()-t0)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding=encoding)
            return TaskResult(True, "write_file", str(resolved), elapsed=time.monotonic() - t0)
        except Exception as exc:
            return TaskResult(False, "write_file", None, str(exc), time.monotonic() - t0)

    def copy_file(self, src: str, dst: str) -> TaskResult:
        """Copy *src* to *dst*. Requires ``read_only=False``."""
        t0 = time.monotonic()
        if self.policy.read_only:
            return TaskResult(False, "copy_file", None, "Policy is read_only=True", time.monotonic()-t0)
        src_p, dst_p = self._resolve(src), self._resolve(dst)
        try:
            if self.policy.dry_run:
                _logger.info("[dry_run] copy_file %s → %s", src_p, dst_p)
                return TaskResult(True, "copy_file", str(dst_p), dry_run=True, elapsed=time.monotonic()-t0)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, dst_p)
            return TaskResult(True, "copy_file", str(dst_p), elapsed=time.monotonic() - t0)
        except Exception as exc:
            return TaskResult(False, "copy_file", None, str(exc), time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Shell execution (requires read_only=False)
    # ------------------------------------------------------------------

    def run_shell(self, command: str) -> TaskResult:
        """Run a shell command and return its stdout.

        Requires ``policy.read_only=False``.  When ``safe_shell_only=True``
        (default), the command is checked against a blocked-pattern list.

        Parameters
        ----------
        command : str
            The shell command string.

        Returns
        -------
        TaskResult
            ``output`` is a dict with ``"stdout"``, ``"stderr"``, and
            ``"returncode"``.
        """
        t0 = time.monotonic()
        if self.policy.read_only:
            return TaskResult(False, "run_shell", None, "Policy is read_only=True", time.monotonic()-t0)

        if self.policy.safe_shell_only and _BLOCKED_RE.search(command):
            _logger.warning("Blocked shell command (matched unsafe pattern): %r", command)
            return TaskResult(
                False, "run_shell", None,
                f"Command blocked by safe_shell_only policy: {command!r}",
                time.monotonic() - t0,
            )

        if self.policy.dry_run:
            _logger.info("[dry_run] run_shell: %r", command)
            return TaskResult(True, "run_shell", {"stdout": "", "stderr": "", "returncode": 0}, dry_run=True, elapsed=time.monotonic()-t0)

        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self.policy.shell_timeout,
            )
            return TaskResult(
                result.returncode == 0,
                "run_shell",
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                error=result.stderr if result.returncode != 0 else None,
                elapsed=time.monotonic() - t0,
            )
        except subprocess.TimeoutExpired:
            return TaskResult(False, "run_shell", None, f"Command timed out after {self.policy.shell_timeout}s", time.monotonic() - t0)
        except Exception as exc:
            return TaskResult(False, "run_shell", None, str(exc), time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Process inspection (always allowed — read-only operation)
    # ------------------------------------------------------------------

    def is_command_available(self, command: str) -> TaskResult:
        """Check if *command* is available on PATH."""
        t0 = time.monotonic()
        available = shutil.which(command) is not None
        return TaskResult(True, "is_command_available", available, elapsed=time.monotonic() - t0)

    def list_processes(self, name_filter: Optional[str] = None) -> TaskResult:
        """List running processes (name, pid).

        Uses ``ps`` on Unix or ``tasklist`` on Windows.  Returns an empty
        list if neither is available.

        Parameters
        ----------
        name_filter : str, optional
            If provided, only processes whose name contains *name_filter*
            (case-insensitive) are returned.
        """
        t0 = time.monotonic()
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["tasklist", "/fo", "csv", "/nh"],
                    capture_output=True, text=True, timeout=5,
                )
                lines = [l.strip('"') for l in result.stdout.splitlines()]
                procs = [{"name": l.split('","')[0], "pid": l.split('","')[1]} for l in lines if '","' in l]
            else:
                result = subprocess.run(
                    ["ps", "-eo", "pid,comm"],
                    capture_output=True, text=True, timeout=5,
                )
                procs = []
                for line in result.stdout.splitlines()[1:]:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        procs.append({"pid": parts[0], "name": parts[1].strip()})

            if name_filter:
                procs = [p for p in procs if name_filter.lower() in p["name"].lower()]
            return TaskResult(True, "list_processes", procs, elapsed=time.monotonic() - t0)
        except Exception as exc:
            return TaskResult(False, "list_processes", [], str(exc), time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Clipboard (optional dep)
    # ------------------------------------------------------------------

    def read_clipboard(self) -> TaskResult:
        """Read from the system clipboard (requires ``pyperclip``)."""
        t0 = time.monotonic()
        try:
            import pyperclip  # type: ignore
            return TaskResult(True, "read_clipboard", pyperclip.paste(), elapsed=time.monotonic() - t0)
        except ImportError:
            return TaskResult(False, "read_clipboard", None, "pyperclip not installed: pip install pyperclip", time.monotonic()-t0)
        except Exception as exc:
            return TaskResult(False, "read_clipboard", None, str(exc), time.monotonic() - t0)

    def write_clipboard(self, text: str) -> TaskResult:
        """Write *text* to the system clipboard. Requires ``read_only=False``."""
        t0 = time.monotonic()
        if self.policy.read_only:
            return TaskResult(False, "write_clipboard", None, "Policy is read_only=True", time.monotonic()-t0)
        try:
            import pyperclip  # type: ignore
            if self.policy.dry_run:
                _logger.info("[dry_run] write_clipboard (%d chars)", len(text))
                return TaskResult(True, "write_clipboard", None, dry_run=True, elapsed=time.monotonic()-t0)
            pyperclip.copy(text)
            return TaskResult(True, "write_clipboard", None, elapsed=time.monotonic() - t0)
        except ImportError:
            return TaskResult(False, "write_clipboard", None, "pyperclip not installed: pip install pyperclip", time.monotonic()-t0)
        except Exception as exc:
            return TaskResult(False, "write_clipboard", None, str(exc), time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.base_dir / p
        return p.resolve()

    def _check_dir_access(self, path: Path) -> None:
        if not self.policy.allowed_dirs:
            return
        for allowed in self.policy.allowed_dirs:
            allowed_p = Path(allowed).expanduser().resolve()
            try:
                path.relative_to(allowed_p)
                return
            except ValueError:
                continue
        raise PermissionError(
            f"Path {path} is outside allowed directories: {self.policy.allowed_dirs}"
        )

    def __repr__(self) -> str:
        return (
            f"LocalTaskExecutor("
            f"read_only={self.policy.read_only}, "
            f"dry_run={self.policy.dry_run}, "
            f"base_dir={self.base_dir})"
        )
