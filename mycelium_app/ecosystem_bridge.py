"""Signal-to-tabular bridge — converts accumulated SignalLedgerEvent rows into
a DataFrame suitable for the physics predictor or sedimentation engine.

This is the "spinal cord" connecting the signal collector (nervous system) to
the physics engine (brain). It windows recent signals, pivots them into
columnar features, and produces a DataFrame where each row is a time bucket.

Feature groups produced:
    - App usage: minutes per app (top N), context switch count
    - Resource: mean/max CPU, memory, battery
    - Network: bytes sent/received
    - Temporal: hour of day, day of week, session duration
    - Behavioral: process diversity, input cadence, screen time
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from mycelium_app.models import SignalLedgerEvent


def _loads(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _extract_stimulus(payload: dict) -> dict:
    """Drill into the stimulus envelope to get the actual signal data."""
    surface = payload.get("surface")
    if isinstance(surface, dict):
        return surface
    stimulus = payload.get("stimulus")
    if isinstance(stimulus, dict):
        return stimulus
    return payload


def build_ecosystem_dataframe(
    session: Session,
    *,
    user_id: int,
    project_id: int | None = None,
    window_hours: int = 6,
    bucket_minutes: int = 30,
    max_signals: int = 10_000,
) -> pd.DataFrame:
    """Query recent signals and pivot them into a tabular ecosystem DataFrame.

    Each row represents one time bucket. Columns are derived features from
    the signal stream: app usage, resource metrics, network flow, temporal
    markers, and behavioral patterns.

    Parameters
    ----------
    session : SQLModel Session
    user_id : int
    project_id : int | None
    window_hours : int — how far back to look
    bucket_minutes : int — size of each time bucket (row)
    max_signals : int — cap for query

    Returns
    -------
    pd.DataFrame with one row per time bucket, columns as derived features.
    """
    since = datetime.utcnow() - timedelta(hours=max(1, min(window_hours, 168)))
    bucket_size = timedelta(minutes=max(5, min(bucket_minutes, 120)))

    q = (
        select(SignalLedgerEvent)
        .where(
            SignalLedgerEvent.created_by_user_id == int(user_id),
            SignalLedgerEvent.created_at >= since,
        )
        .order_by(SignalLedgerEvent.created_at)
    )
    if project_id is not None:
        q = q.where(SignalLedgerEvent.project_id == int(project_id))

    rows = session.exec(q.limit(max_signals)).all()
    if not rows:
        return pd.DataFrame()

    # Determine bucket boundaries
    t_start = min(r.created_at for r in rows)
    t_end = max(r.created_at for r in rows)
    total_span = (t_end - t_start).total_seconds()
    n_buckets = max(1, int(math.ceil(total_span / bucket_size.total_seconds())))

    # Assign each signal to a bucket
    buckets: dict[int, list[tuple[SignalLedgerEvent, dict]]] = defaultdict(list)
    for r in rows:
        elapsed = (r.created_at - t_start).total_seconds()
        b_idx = min(n_buckets - 1, int(elapsed / bucket_size.total_seconds()))
        payload = _loads(r.payload_json)
        stim = _extract_stimulus(payload)
        buckets[b_idx].append((r, stim))

    # Build feature rows
    records: list[dict[str, Any]] = []
    all_apps: Counter[str] = Counter()

    # First pass: discover top apps
    for b_idx in range(n_buckets):
        for r, stim in buckets.get(b_idx, []):
            sig_type = str(r.signal_type or "").lower()
            if sig_type in ("app_open", "app_focus"):
                app = str(stim.get("app_name", "")).strip().lower()[:32]
                if app:
                    all_apps[app] += 1
            if sig_type == "process_snapshot":
                top_procs = stim.get("top_processes")
                if isinstance(top_procs, dict):
                    for pname, cnt in top_procs.items():
                        try:
                            all_apps[str(pname).lower()[:32]] += int(cnt)
                        except (ValueError, TypeError):
                            continue

    top_app_names = [name for name, _ in all_apps.most_common(15)]

    # Second pass: aggregate per bucket
    for b_idx in range(n_buckets):
        bucket_items = buckets.get(b_idx, [])
        bucket_start = t_start + timedelta(seconds=b_idx * bucket_size.total_seconds())

        rec: dict[str, Any] = {
            "bucket_index": b_idx,
            "hour_of_day": bucket_start.hour,
            "day_of_week": bucket_start.weekday(),
            "minute_of_day": bucket_start.hour * 60 + bucket_start.minute,
            "n_signals": len(bucket_items),
        }

        # Initialize app columns
        for app in top_app_names:
            rec[f"app_{app}"] = 0

        # Resource accumulators
        cpu_vals: list[float] = []
        mem_vals: list[float] = []
        battery_vals: list[float] = []
        net_sent: list[int] = []
        net_recv: list[int] = []
        disk_read: list[int] = []
        disk_write: list[int] = []

        # Behavioral
        n_app_opens = 0
        n_app_closes = 0
        context_switches = 0
        unique_processes = 0
        total_processes = 0

        for r, stim in bucket_items:
            sig_type = str(r.signal_type or "").lower()

            if sig_type in ("app_open", "app_focus"):
                app = str(stim.get("app_name", "")).strip().lower()[:32]
                if app and f"app_{app}" in rec:
                    rec[f"app_{app}"] += 1
                n_app_opens += 1

            elif sig_type == "app_close":
                n_app_closes += 1

            elif sig_type == "resource_pulse":
                cpu = stim.get("cpu_percent")
                if cpu is not None:
                    cpu_vals.append(float(cpu))
                mem = stim.get("memory_percent")
                if mem is not None:
                    mem_vals.append(float(mem))
                bat = stim.get("battery_percent")
                if bat is not None:
                    battery_vals.append(float(bat))

            elif sig_type == "network_flow":
                s = stim.get("bytes_sent_delta")
                if s is not None:
                    net_sent.append(int(s))
                rv = stim.get("bytes_recv_delta")
                if rv is not None:
                    net_recv.append(int(rv))

            elif sig_type == "disk_io":
                dr = stim.get("read_bytes_delta")
                if dr is not None:
                    disk_read.append(int(dr))
                dw = stim.get("write_bytes_delta")
                if dw is not None:
                    disk_write.append(int(dw))

            elif sig_type == "process_snapshot":
                up = stim.get("unique_processes")
                if up is not None:
                    unique_processes = max(unique_processes, int(up))
                tp = stim.get("total_processes")
                if tp is not None:
                    total_processes = max(total_processes, int(tp))
                cs = stim.get("n_opened", 0)
                context_switches += int(cs or 0)

        # Aggregate resource metrics
        rec["cpu_mean"] = round(float(np.mean(cpu_vals)), 2) if cpu_vals else 0.0
        rec["cpu_max"] = round(float(max(cpu_vals)), 2) if cpu_vals else 0.0
        rec["memory_mean"] = round(float(np.mean(mem_vals)), 2) if mem_vals else 0.0
        rec["battery_mean"] = round(float(np.mean(battery_vals)), 2) if battery_vals else 0.0
        rec["net_sent_bytes"] = sum(net_sent)
        rec["net_recv_bytes"] = sum(net_recv)
        rec["disk_read_bytes"] = sum(disk_read)
        rec["disk_write_bytes"] = sum(disk_write)

        # Behavioral metrics
        rec["app_opens"] = n_app_opens
        rec["app_closes"] = n_app_closes
        rec["context_switches"] = context_switches
        rec["unique_processes"] = unique_processes
        rec["total_processes"] = total_processes
        rec["process_diversity"] = round(
            unique_processes / max(1, total_processes), 4
        ) if total_processes > 0 else 0.0

        records.append(rec)

    df = pd.DataFrame(records)

    # Drop columns that are all zero (uninformative)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_cols = [c for c in numeric_cols if (df[c] == 0).all()]
    df = df.drop(columns=zero_cols)

    return df


def build_ecosystem_summary(
    session: Session,
    *,
    user_id: int,
    project_id: int | None = None,
    window_hours: int = 24,
) -> dict[str, Any]:
    """Build a high-level ecosystem summary for the narrative layer."""
    since = datetime.utcnow() - timedelta(hours=max(1, min(window_hours, 168)))

    q = (
        select(SignalLedgerEvent)
        .where(
            SignalLedgerEvent.created_by_user_id == int(user_id),
            SignalLedgerEvent.created_at >= since,
        )
    )
    if project_id is not None:
        q = q.where(SignalLedgerEvent.project_id == int(project_id))

    rows = session.exec(q).all()

    signal_counts: Counter[str] = Counter()
    app_counts: Counter[str] = Counter()
    cpu_vals: list[float] = []
    battery_vals: list[float] = []

    for r in rows:
        sig_type = str(r.signal_type or "").lower()
        signal_counts[sig_type] += 1
        payload = _loads(r.payload_json)
        stim = _extract_stimulus(payload)

        if sig_type in ("app_open", "app_focus"):
            app = str(stim.get("app_name", "")).strip().lower()[:32]
            if app:
                app_counts[app] += 1

        if sig_type == "resource_pulse":
            cpu = stim.get("cpu_percent")
            if cpu is not None:
                cpu_vals.append(float(cpu))
            bat = stim.get("battery_percent")
            if bat is not None:
                battery_vals.append(float(bat))

    hours_active = 0
    if rows:
        t_first = min(r.created_at for r in rows)
        t_last = max(r.created_at for r in rows)
        hours_active = round((t_last - t_first).total_seconds() / 3600, 1)

    return {
        "window_hours": window_hours,
        "n_signals": len(rows),
        "hours_active": hours_active,
        "signal_types": dict(signal_counts.most_common(20)),
        "top_apps": dict(app_counts.most_common(10)),
        "cpu_mean": round(float(np.mean(cpu_vals)), 1) if cpu_vals else None,
        "cpu_max": round(float(max(cpu_vals)), 1) if cpu_vals else None,
        "battery_mean": round(float(np.mean(battery_vals)), 1) if battery_vals else None,
    }
