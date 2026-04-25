"""Stage 116 — DeviceMonitor: device state monitoring.

Monitors the user's device state: CPU usage, RAM, disk space, network
connectivity, battery level, and connected USB devices.

Primary backend: ``psutil`` (optional; graceful fallback to subprocess).
Emits :class:`DeviceSnapshot` dataclasses.  Can run as a background thread
polling at a configurable interval.

Usage
-----
::

    from physml.device_monitor import DeviceMonitor

    monitor = DeviceMonitor(poll_interval=30)
    snap = monitor.snapshot()
    print(snap.cpu_percent)
    print(snap.ram_used_mb)
    print(snap.disk_free_gb)
    print(snap.network_available)
    print(snap.battery_percent)   # None if desktop

    monitor.start_background()
    # ... later ...
    monitor.stop()
    print(monitor.history)   # list of DeviceSnapshot
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# Try psutil once at import time
try:
    import psutil  # type: ignore

    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    _logger.info("DeviceMonitor: psutil not installed; using subprocess fallback")


@dataclass
class DeviceSnapshot:
    """A point-in-time device state snapshot.

    Attributes
    ----------
    timestamp : float
        Unix time of the snapshot.
    cpu_percent : float or None
        CPU utilisation percentage (0–100).
    ram_used_mb : float or None
        Used RAM in megabytes.
    ram_total_mb : float or None
        Total RAM in megabytes.
    disk_free_gb : float or None
        Free disk space in gigabytes (checked on root / home).
    network_available : bool
        ``True`` if a basic network check succeeds.
    battery_percent : float or None
        Battery charge percentage, or ``None`` for desktop machines.
    load_avg_1m : float or None
        1-minute load average (Unix only).
    """

    timestamp: float = field(default_factory=time.time)
    cpu_percent: Optional[float] = None
    ram_used_mb: Optional[float] = None
    ram_total_mb: Optional[float] = None
    disk_free_gb: Optional[float] = None
    network_available: bool = False
    battery_percent: Optional[float] = None
    load_avg_1m: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class DeviceMonitor:
    """Monitor device state with an optional background polling thread.

    Parameters
    ----------
    poll_interval : float, default 30
        Seconds between background snapshots.
    max_history : int, default 100
        Maximum number of snapshots to keep in :attr:`history`.
    check_network_host : str, default "8.8.8.8"
        Host to ping for network check.  Set to ``None`` to skip.
    """

    def __init__(
        self,
        poll_interval: float = 30.0,
        max_history: int = 100,
        check_network_host: Optional[str] = "8.8.8.8",
    ) -> None:
        self.poll_interval = float(poll_interval)
        self.max_history = int(max_history)
        self.check_network_host = check_network_host
        self._history: List[DeviceSnapshot] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> DeviceSnapshot:
        """Take a single device snapshot.

        Returns
        -------
        DeviceSnapshot
        """
        if _PSUTIL:
            snap = self._snap_psutil()
        else:
            snap = self._snap_subprocess()
        snap.network_available = self._check_network()
        return snap

    @property
    def history(self) -> List[DeviceSnapshot]:
        """List of recent :class:`DeviceSnapshot` objects."""
        return list(self._history)

    def start_background(self) -> None:
        """Start background polling thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        _logger.info("DeviceMonitor: background polling started (interval=%.1fs)", self.poll_interval)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 1)
            self._thread = None
        _logger.info("DeviceMonitor: stopped")

    # ------------------------------------------------------------------
    # Snapshot implementations
    # ------------------------------------------------------------------

    def _snap_psutil(self) -> DeviceSnapshot:
        snap = DeviceSnapshot()
        try:
            snap.cpu_percent = psutil.cpu_percent(interval=0.1)
        except Exception as e:
            _logger.warning("DeviceMonitor: cpu_percent failed: %s", e)
        try:
            mem = psutil.virtual_memory()
            snap.ram_used_mb = mem.used / 1e6
            snap.ram_total_mb = mem.total / 1e6
        except Exception as e:
            _logger.warning("DeviceMonitor: memory failed: %s", e)
        try:
            disk = psutil.disk_usage("/")
            snap.disk_free_gb = disk.free / 1e9
        except Exception as e:
            _logger.warning("DeviceMonitor: disk_usage failed: %s", e)
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                snap.battery_percent = battery.percent
        except Exception as e:
            _logger.warning("DeviceMonitor: battery check failed: %s", e)
        try:
            load = psutil.getloadavg()
            snap.load_avg_1m = load[0]
        except Exception as e:
            _logger.warning("DeviceMonitor: load_avg failed: %s", e)
        return snap

    def _snap_subprocess(self) -> DeviceSnapshot:
        """Fallback snapshot using subprocess commands."""
        snap = DeviceSnapshot()
        # CPU (Linux/Mac: top)
        try:
            result = subprocess.run(
                ["top", "-bn1"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "Cpu(s)" in line or "cpu" in line.lower():
                    import re
                    m = re.search(r"(\d+\.\d+)\s*%?\s*(?:us|id|user)", line)
                    if m:
                        snap.cpu_percent = float(m.group(1))
                        break
        except Exception as e:
            _logger.warning("DeviceMonitor (subprocess): cpu failed: %s", e)

        # Memory (Linux: /proc/meminfo)
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    meminfo[k.strip()] = int(v.strip().split()[0])
            total_mb = meminfo.get("MemTotal", 0) / 1024
            free_mb = (meminfo.get("MemFree", 0) + meminfo.get("Buffers", 0)
                       + meminfo.get("Cached", 0)) / 1024
            snap.ram_total_mb = total_mb
            snap.ram_used_mb = total_mb - free_mb
        except Exception as e:
            _logger.warning("DeviceMonitor (subprocess): memory failed: %s", e)

        # Disk
        try:
            result = subprocess.run(
                ["df", "-BG", "/"], capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    snap.disk_free_gb = float(parts[3].rstrip("G"))
        except Exception as e:
            _logger.warning("DeviceMonitor (subprocess): disk failed: %s", e)

        # Load avg
        try:
            with open("/proc/loadavg", "r") as f:
                snap.load_avg_1m = float(f.read().split()[0])
        except Exception as e:
            _logger.warning("DeviceMonitor (subprocess): load_avg failed: %s", e)

        return snap

    # ------------------------------------------------------------------
    # Network check
    # ------------------------------------------------------------------

    def _check_network(self) -> bool:
        if not self.check_network_host:
            return False
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", self.check_network_host],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self.snapshot()
                self._history.append(snap)
                if len(self._history) > self.max_history:
                    self._history = self._history[-self.max_history:]
            except Exception as e:
                _logger.warning("DeviceMonitor: poll error: %s", e)
            self._stop_event.wait(timeout=self.poll_interval)

    def __repr__(self) -> str:
        return (
            f"DeviceMonitor("
            f"poll_interval={self.poll_interval}s, "
            f"history={len(self._history)} snaps)"
        )
