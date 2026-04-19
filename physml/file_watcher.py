"""Stage 132 — FileWatcher: proactive learning from new files.

Monitors one or more directories for new or changed files.  When a
supported file appears, the companion is notified so it can ingest
new training data, documents, or configuration automatically — without
the user having to ask.

Requires ``watchdog`` (``pip install watchdog``).  Falls back to a
polling stub when watchdog is absent.

Usage
-----
::

    from physml.file_watcher import FileWatcher

    def on_new_file(path):
        print(f"New file: {path}")

    watcher = FileWatcher(
        watch_dirs=["~/Downloads", "~/Documents"],
        callback=on_new_file,
        extensions={".csv", ".txt", ".pdf"},
    )
    watcher.start()
    # ... runs in background ...
    watcher.stop()
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Set

from physml._log import get_logger

_logger = get_logger(__name__)

try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent  # type: ignore
    _WD_OK = True
except Exception:
    _WD_OK = False
    Observer = None
    FileSystemEventHandler = object


_DEFAULT_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".pdf", ".json",
    ".xlsx", ".xls", ".parquet", ".md",
}


class _Handler(FileSystemEventHandler if _WD_OK else object):
    def __init__(self, callback: Callable, extensions: Set[str]) -> None:
        if _WD_OK:
            super().__init__()
        self._callback = callback
        self._extensions = extensions

    def on_created(self, event: Any) -> None:
        self._dispatch(event)

    def on_modified(self, event: Any) -> None:
        self._dispatch(event)

    def _dispatch(self, event: Any) -> None:
        path = getattr(event, "src_path", None)
        if path is None:
            return
        p = Path(path)
        if p.suffix.lower() in self._extensions:
            try:
                self._callback(str(p))
            except Exception as exc:
                _logger.warning("FileWatcher callback error: %s", exc)


class _PollingWatcher:
    """Fallback polling watcher when watchdog is not installed."""

    def __init__(
        self,
        watch_dirs: List[str],
        callback: Callable,
        extensions: Set[str],
        poll_interval: float = 5.0,
    ) -> None:
        self._dirs = [Path(d).expanduser() for d in watch_dirs]
        self._callback = callback
        self._extensions = extensions
        self._interval = poll_interval
        self._seen: Set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._seed()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _seed(self) -> None:
        for d in self._dirs:
            if not d.exists():
                continue
            for p in d.rglob("*"):
                if p.suffix.lower() in self._extensions:
                    self._seen.add(str(p))

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            for d in self._dirs:
                if not d.exists():
                    continue
                for p in d.rglob("*"):
                    sp = str(p)
                    if p.suffix.lower() in self._extensions and sp not in self._seen:
                        self._seen.add(sp)
                        try:
                            self._callback(sp)
                        except Exception as exc:
                            _logger.warning("PollingWatcher callback error: %s", exc)
            self._stop_event.wait(self._interval)


class FileWatcher:
    """Monitor directories and fire a callback when relevant files appear.

    Parameters
    ----------
    watch_dirs : list of str
        Directories to watch.  ``~`` is expanded.
    callback : callable
        Called with the file path (str) when a new/changed file is detected.
    extensions : set of str, optional
        File extensions to watch.  Defaults to common data/doc types.
    recursive : bool, default True
        Watch subdirectories recursively.
    poll_interval : float, default 5.0
        Seconds between polls (polling fallback only).
    """

    def __init__(
        self,
        watch_dirs: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        extensions: Optional[Set[str]] = None,
        recursive: bool = True,
        poll_interval: float = 5.0,
    ) -> None:
        self.watch_dirs = watch_dirs or []
        self.callback = callback or (lambda p: None)
        self.extensions = extensions or _DEFAULT_EXTENSIONS
        self.recursive = recursive
        self._poll_interval = poll_interval
        self._observer: Any = None
        self._polling: Optional[_PollingWatcher] = None
        self._running = False

    @property
    def available(self) -> bool:
        return _WD_OK

    def start(self) -> None:
        """Start watching in a background thread."""
        if self._running:
            return
        if _WD_OK and self.watch_dirs:
            handler = _Handler(self.callback, self.extensions)
            self._observer = Observer()
            for d in self.watch_dirs:
                path = Path(d).expanduser()
                path.mkdir(parents=True, exist_ok=True)
                self._observer.schedule(handler, str(path), recursive=self.recursive)
            self._observer.start()
            _logger.info("FileWatcher: watchdog watching %s", self.watch_dirs)
        else:
            self._polling = _PollingWatcher(
                self.watch_dirs, self.callback, self.extensions, self._poll_interval
            )
            self._polling.start()
            _logger.info("FileWatcher: polling %s (watchdog not installed)", self.watch_dirs)
        self._running = True

    def stop(self) -> None:
        """Stop all watchers."""
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception:
                pass
            self._observer = None
        if self._polling is not None:
            self._polling.stop()
            self._polling = None
        self._running = False

    def add_directory(self, path: str) -> None:
        """Add a new directory to watch (must call start() again if already running)."""
        p = str(Path(path).expanduser())
        if p not in self.watch_dirs:
            self.watch_dirs.append(p)

    def status(self) -> dict:
        return {
            "running": self._running,
            "backend": "watchdog" if _WD_OK else "polling",
            "watching": self.watch_dirs,
            "extensions": sorted(self.extensions),
        }
