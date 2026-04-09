"""Pattern intelligence engine — detects behavioral patterns from the signal stream.

Extracts circadian rhythms, app usage patterns, anomalies, routine sequences,
and generates proactive suggestions. This is the brain behind JARVIS-like
intelligence — turning raw signals into actionable understanding.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from mycelium_app.humanizer import humanize_app, humanize_signal
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
    surface = payload.get("surface")
    if isinstance(surface, dict):
        return surface
    stimulus = payload.get("stimulus")
    if isinstance(stimulus, dict):
        return stimulus
    return payload


def analyze_patterns(
    session: Session,
    *,
    user_id: int,
    window_hours: int = 48,
) -> dict[str, Any]:
    """Analyze behavioral patterns from recent signals."""
    since = datetime.utcnow() - timedelta(hours=max(1, min(window_hours, 168)))

    rows = session.exec(
        select(SignalLedgerEvent)
        .where(SignalLedgerEvent.created_by_user_id == int(user_id))
        .where(SignalLedgerEvent.created_at >= since)
        .order_by(SignalLedgerEvent.created_at)
    ).all()

    if not rows:
        return {"ok": True, "n_signals": 0, "patterns": [], "insights": []}

    patterns: list[dict[str, Any]] = []
    insights: list[str] = []

    # --- Circadian rhythm ---
    hourly_activity: dict[int, int] = defaultdict(int)
    hourly_apps: dict[int, Counter] = defaultdict(Counter)
    app_durations: dict[str, float] = defaultdict(float)
    app_counts: Counter[str] = Counter()
    transitions: list[tuple[str, str]] = []
    cpu_readings: list[tuple[int, float]] = []
    focus_sessions: list[dict] = []

    for r in rows:
        hour = r.created_at.hour if r.created_at else 12
        sig_type = str(r.signal_type or "").lower()
        hourly_activity[hour] += 1

        payload = _loads(r.payload_json)
        stim = _extract_stimulus(payload)

        if sig_type in ("app_open", "app_focus"):
            app = str(stim.get("app_name", "")).strip().lower()[:32]
            if app:
                hourly_apps[hour][app] += 1
                app_counts[app] += 1
                prev = str(stim.get("previous_app", "")).strip().lower()[:32]
                if prev and prev != app:
                    transitions.append((prev, app))

        if sig_type == "app_session_end":
            app = str(stim.get("app_name", "")).strip().lower()[:32]
            secs = float(stim.get("session_seconds", 0) or 0)
            if app and secs > 0:
                app_durations[app] += secs
                focus_sessions.append({"app": app, "seconds": secs, "hour": hour})

        if sig_type == "resource_pulse":
            cpu = stim.get("cpu_percent")
            if cpu is not None:
                cpu_readings.append((hour, float(cpu)))

    # Peak activity hours
    if hourly_activity:
        sorted_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)
        peak_hour = sorted_hours[0][0]
        quiet_hours = [h for h in range(24) if hourly_activity.get(h, 0) == 0]

        patterns.append({
            "type": "circadian_rhythm",
            "peak_hour": peak_hour,
            "peak_activity": sorted_hours[0][1],
            "active_hours": len([h for h, c in hourly_activity.items() if c > 0]),
            "quiet_hours": quiet_hours[:5],
            "hourly_distribution": dict(sorted(hourly_activity.items())),
        })
        insights.append(
            f"You're most active around {peak_hour}:00. "
            f"You're active during {len(hourly_activity)} different hours."
        )

    # Top apps by usage time
    if app_durations:
        sorted_apps = sorted(app_durations.items(), key=lambda x: x[1], reverse=True)
        top_apps = []
        for app, secs in sorted_apps[:8]:
            minutes = round(secs / 60, 1)
            top_apps.append({
                "app": humanize_app(app),
                "raw_app": app,
                "minutes": minutes,
                "sessions": len([f for f in focus_sessions if f["app"] == app]),
            })
        patterns.append({"type": "app_usage", "top_apps": top_apps})

        top_app = top_apps[0]
        insights.append(
            f"You've spent {top_app['minutes']} minutes in {top_app['app']} "
            f"across {top_app['sessions']} sessions."
        )

    # App transition sequences (routines)
    if transitions:
        transition_counts = Counter(transitions)
        common_transitions = transition_counts.most_common(5)
        routines = []
        for (from_app, to_app), count in common_transitions:
            if count >= 2:
                routines.append({
                    "from": humanize_app(from_app),
                    "to": humanize_app(to_app),
                    "count": count,
                })
        if routines:
            patterns.append({"type": "routines", "transitions": routines})
            top_r = routines[0]
            insights.append(
                f"Common routine: {top_r['from']} → {top_r['to']} ({top_r['count']} times)."
            )

    # Focus session analysis
    if focus_sessions:
        avg_session = sum(f["seconds"] for f in focus_sessions) / len(focus_sessions)
        longest = max(focus_sessions, key=lambda f: f["seconds"])
        patterns.append({
            "type": "focus_analysis",
            "avg_session_minutes": round(avg_session / 60, 1),
            "longest_session_minutes": round(longest["seconds"] / 60, 1),
            "longest_session_app": humanize_app(longest["app"]),
            "total_sessions": len(focus_sessions),
        })
        insights.append(
            f"Average focus session: {round(avg_session / 60, 1)} minutes. "
            f"Longest: {round(longest['seconds'] / 60, 1)} min in {humanize_app(longest['app'])}."
        )

    # CPU anomaly detection
    if cpu_readings:
        cpu_values = [c for _, c in cpu_readings]
        mean_cpu = sum(cpu_values) / len(cpu_values)
        std_cpu = (sum((c - mean_cpu) ** 2 for c in cpu_values) / len(cpu_values)) ** 0.5 if len(cpu_values) > 1 else 0
        high_cpu = [c for c in cpu_values if c > mean_cpu + 2 * std_cpu] if std_cpu > 0 else []

        if high_cpu:
            patterns.append({
                "type": "cpu_anomaly",
                "mean_cpu": round(mean_cpu, 1),
                "std_cpu": round(std_cpu, 1),
                "spike_count": len(high_cpu),
                "max_spike": round(max(high_cpu), 1),
            })
            insights.append(
                f"Detected {len(high_cpu)} CPU spikes above {round(mean_cpu + 2 * std_cpu, 0)}%. "
                f"Your average is {round(mean_cpu, 0)}%."
            )

    # Context switching rate
    total_switches = len(transitions)
    active_hours_count = max(1, len([h for h, c in hourly_activity.items() if c > 0]))
    switches_per_hour = round(total_switches / active_hours_count, 1)
    if total_switches > 0:
        patterns.append({
            "type": "context_switching",
            "total_switches": total_switches,
            "switches_per_hour": switches_per_hour,
            "active_hours": active_hours_count,
        })
        if switches_per_hour > 10:
            insights.append(
                f"High context switching: {switches_per_hour} app switches/hour. "
                "Consider longer focus blocks."
            )

    return {
        "ok": True,
        "n_signals": len(rows),
        "window_hours": window_hours,
        "patterns": patterns,
        "insights": insights,
        "n_apps_tracked": len(app_counts),
        "n_transitions": len(transitions),
        "n_focus_sessions": len(focus_sessions),
    }


def generate_proactive_suggestions(
    patterns: list[dict[str, Any]],
    *,
    stage: str = "infant",
    mood: str = "curious",
) -> list[dict[str, str]]:
    """Generate JARVIS-like proactive suggestions from detected patterns."""
    suggestions: list[dict[str, str]] = []

    for p in patterns:
        ptype = p.get("type", "")

        if ptype == "circadian_rhythm":
            peak = p.get("peak_hour", 12)
            suggestions.append({
                "type": "schedule",
                "title": "Optimal focus window",
                "message": f"Your peak activity is around {peak}:00. Schedule your hardest tasks for this window.",
                "priority": "medium",
            })

        elif ptype == "app_usage":
            top = (p.get("top_apps") or [{}])[0]
            if top.get("minutes", 0) > 60:
                suggestions.append({
                    "type": "awareness",
                    "title": f"Heavy usage: {top.get('app', 'an app')}",
                    "message": f"You've spent {top.get('minutes', 0)} minutes here. Consider a break.",
                    "priority": "low",
                })

        elif ptype == "context_switching":
            rate = p.get("switches_per_hour", 0)
            if rate > 15:
                suggestions.append({
                    "type": "focus",
                    "title": "High context switching detected",
                    "message": f"You're switching apps {rate} times/hour. Try 25-minute focus blocks.",
                    "priority": "high",
                })

        elif ptype == "cpu_anomaly":
            spikes = p.get("spike_count", 0)
            if spikes > 3:
                suggestions.append({
                    "type": "system",
                    "title": "CPU spikes detected",
                    "message": f"Your system had {spikes} CPU spikes. An app might be misbehaving.",
                    "priority": "medium",
                })

        elif ptype == "routines":
            transitions = p.get("transitions", [])
            if transitions:
                t = transitions[0]
                suggestions.append({
                    "type": "routine",
                    "title": "I noticed a routine",
                    "message": f"You often go from {t['from']} to {t['to']}. Want me to streamline this?",
                    "priority": "low",
                })

    return suggestions
