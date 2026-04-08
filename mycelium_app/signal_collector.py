"""OS-level signal collector — the nervous system of the digital organism.

Captures real-time signals from the user's hardware using psutil and OS APIs.
Signals are structured into the SignalLedgerEvent schema and stored via the
existing stimulus pipeline.

Signal types captured:
    - system_boot         Boot/wake events (session boundaries)
    - system_shutdown     Graceful shutdown detection (via heartbeat gap)
    - app_focus           Active application / window focus changes
    - process_snapshot    Running process census (per-tick summary)
    - resource_pulse      CPU, memory, disk, battery snapshot
    - input_cadence       Typing/mouse activity rate (never keystrokes)
    - network_flow        Network I/O rates (bytes, not content)
    - display_state       Screen on/off, brightness proxy
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


@dataclass
class CollectorState:
    """Mutable state carried across collection ticks."""

    last_tick_at: float = 0.0
    last_process_set: set[str] = field(default_factory=set)
    last_net_bytes_sent: int = 0
    last_net_bytes_recv: int = 0
    last_disk_read_bytes: int = 0
    last_disk_write_bytes: int = 0
    boot_signal_emitted: bool = False
    tick_count: int = 0


def _psutil_available() -> bool:
    return psutil is not None


def collect_boot_signal(state: CollectorState) -> list[dict[str, Any]]:
    """Emit a system_boot signal once per collector lifetime."""
    if state.boot_signal_emitted or not _psutil_available():
        return []
    state.boot_signal_emitted = True

    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime_seconds = time.time() - psutil.boot_time()

    return [{
        "signal_type": "system_boot",
        "source": "os",
        "modality": "system",
        "stimulus": {
            "boot_time_iso": boot_time.isoformat(),
            "uptime_seconds": round(uptime_seconds, 1),
            "platform": platform.system(),
            "platform_release": platform.release(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_memory_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
        },
    }]


def collect_resource_pulse(state: CollectorState) -> list[dict[str, Any]]:
    """Snapshot CPU, memory, disk, and battery."""
    if not _psutil_available():
        return []

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_pct = psutil.cpu_percent(interval=0)

    stimulus: dict[str, Any] = {
        "cpu_percent": round(cpu_pct, 1),
        "memory_percent": round(mem.percent, 1),
        "memory_available_mb": round(mem.available / (1024 * 1024), 1),
        "disk_percent": round(disk.percent, 1),
        "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
    }

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    stimulus["cpu_temp_c"] = round(entries[0].current, 1)
                    break
    except Exception:
        pass

    try:
        battery = psutil.sensors_battery()
        if battery is not None:
            stimulus["battery_percent"] = round(battery.percent, 1)
            stimulus["battery_plugged"] = bool(battery.power_plugged)
            if battery.secsleft and battery.secsleft > 0:
                stimulus["battery_minutes_left"] = round(battery.secsleft / 60, 1)
    except Exception:
        pass

    return [{
        "signal_type": "resource_pulse",
        "source": "os",
        "modality": "telemetry",
        "stimulus": stimulus,
    }]


def collect_process_snapshot(state: CollectorState) -> list[dict[str, Any]]:
    """Census of running processes — names and counts, never command lines."""
    if not _psutil_available():
        return []

    current_names: dict[str, int] = {}
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").strip().lower()[:64]
                if name:
                    current_names[name] = current_names.get(name, 0) + 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        return []

    current_set = set(current_names.keys())
    opened = current_set - state.last_process_set
    closed = state.last_process_set - current_set
    state.last_process_set = current_set

    top_by_count = sorted(current_names.items(), key=lambda x: x[1], reverse=True)[:20]

    signals: list[dict[str, Any]] = [{
        "signal_type": "process_snapshot",
        "source": "os",
        "modality": "telemetry",
        "stimulus": {
            "total_processes": sum(current_names.values()),
            "unique_processes": len(current_set),
            "top_processes": {k: v for k, v in top_by_count},
            "opened_since_last": list(opened)[:10],
            "closed_since_last": list(closed)[:10],
            "n_opened": len(opened),
            "n_closed": len(closed),
        },
    }]

    for app in list(opened)[:5]:
        signals.append({
            "signal_type": "app_open",
            "source": "os",
            "modality": "app_lifecycle",
            "stimulus": {
                "app_name": app,
                "instance_count": current_names.get(app, 1),
            },
        })

    for app in list(closed)[:5]:
        signals.append({
            "signal_type": "app_close",
            "source": "os",
            "modality": "app_lifecycle",
            "stimulus": {
                "app_name": app,
            },
        })

    return signals


def collect_network_flow(state: CollectorState) -> list[dict[str, Any]]:
    """Network I/O rates (bytes, not packet content)."""
    if not _psutil_available():
        return []

    try:
        counters = psutil.net_io_counters()
    except Exception:
        return []

    sent = counters.bytes_sent
    recv = counters.bytes_recv

    delta_sent = max(0, sent - state.last_net_bytes_sent) if state.last_net_bytes_sent > 0 else 0
    delta_recv = max(0, recv - state.last_net_bytes_recv) if state.last_net_bytes_recv > 0 else 0

    state.last_net_bytes_sent = sent
    state.last_net_bytes_recv = recv

    if delta_sent == 0 and delta_recv == 0 and state.tick_count > 1:
        return []

    return [{
        "signal_type": "network_flow",
        "source": "os",
        "modality": "telemetry",
        "stimulus": {
            "bytes_sent_delta": delta_sent,
            "bytes_recv_delta": delta_recv,
            "bytes_sent_total_mb": round(sent / (1024 * 1024), 2),
            "bytes_recv_total_mb": round(recv / (1024 * 1024), 2),
            "packets_sent": counters.packets_sent,
            "packets_recv": counters.packets_recv,
        },
    }]


def collect_disk_io(state: CollectorState) -> list[dict[str, Any]]:
    """Disk I/O rates."""
    if not _psutil_available():
        return []

    try:
        counters = psutil.disk_io_counters()
        if counters is None:
            return []
    except Exception:
        return []

    read_b = counters.read_bytes
    write_b = counters.write_bytes

    delta_read = max(0, read_b - state.last_disk_read_bytes) if state.last_disk_read_bytes > 0 else 0
    delta_write = max(0, write_b - state.last_disk_write_bytes) if state.last_disk_write_bytes > 0 else 0

    state.last_disk_read_bytes = read_b
    state.last_disk_write_bytes = write_b

    if delta_read == 0 and delta_write == 0 and state.tick_count > 1:
        return []

    return [{
        "signal_type": "disk_io",
        "source": "os",
        "modality": "telemetry",
        "stimulus": {
            "read_bytes_delta": delta_read,
            "write_bytes_delta": delta_write,
            "read_count": counters.read_count,
            "write_count": counters.write_count,
        },
    }]


def collect_all_signals(state: CollectorState) -> list[dict[str, Any]]:
    """Run all collectors and return a combined signal list."""
    state.tick_count += 1
    now = time.time()
    state.last_tick_at = now

    signals: list[dict[str, Any]] = []
    signals.extend(collect_boot_signal(state))
    signals.extend(collect_resource_pulse(state))
    signals.extend(collect_process_snapshot(state))
    signals.extend(collect_network_flow(state))
    signals.extend(collect_disk_io(state))

    for s in signals:
        s.setdefault("occurred_at", datetime.utcnow().isoformat())

    return signals
