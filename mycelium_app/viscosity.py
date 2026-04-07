from __future__ import annotations

import json

from mycelium_app.models import SignalLedgerEvent
from mycelium_app.schemas import LiveViscositySnapshot


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _to_float(v: object) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def calculate_live_viscosity(signals: list[SignalLedgerEvent]) -> LiveViscositySnapshot:
    battery_vals: list[float] = []
    temp_vals: list[float] = []
    interruptions = 0

    for s in signals:
        signal_type = str(s.signal_type or "").strip().lower()
        payload = _loads_dict(s.payload_json)

        battery = (
            _to_float(payload.get("battery_level"))
            or _to_float(payload.get("battery_pct"))
            or _to_float(payload.get("battery"))
        )
        if battery is not None:
            battery_vals.append(max(0.0, min(float(battery), 100.0)))

        temp = (
            _to_float(payload.get("cpu_temp"))
            or _to_float(payload.get("cpu_temperature"))
            or _to_float(payload.get("temp_c"))
            or _to_float(payload.get("thermal_c"))
        )
        if temp is not None:
            temp_vals.append(max(0.0, min(float(temp), 120.0)))

        ic = _to_float(payload.get("interruption_count"))
        if ic is not None:
            interruptions += max(0, int(ic))

        if signal_type in {"notification", "interrupt", "app_switch", "call", "message"}:
            interruptions += 1

    n_signals = max(1, len(signals))
    avg_battery = (sum(battery_vals) / len(battery_vals)) if battery_vals else None
    avg_temp = (sum(temp_vals) / len(temp_vals)) if temp_vals else None

    if avg_battery is None:
        battery_factor = _clamp01(0.45 - min(0.25, (len(signals) / 500.0)))
    else:
        battery_factor = _clamp01(1.0 - (float(avg_battery) / 100.0))

    if avg_temp is None:
        batteryish = sum(1 for s in signals if str(s.signal_type or "").lower() in {"app_open", "network"})
        thermal_factor = _clamp01(float(batteryish) / float(n_signals))
    else:
        thermal_factor = _clamp01(float(avg_temp) / 100.0)

    interruption_factor = _clamp01(float(interruptions) / float(max(4, n_signals // 2)))
    viscosity = _clamp01((battery_factor * 0.4) + (thermal_factor * 0.4) + (interruption_factor * 0.2))

    if viscosity >= 0.75:
        band = "high"
        prediction_state = "gated"
    elif viscosity <= 0.35:
        band = "low"
        prediction_state = "flow"
    else:
        band = "medium"
        prediction_state = "observe"

    return LiveViscositySnapshot(
        score=float(round(viscosity, 3)),
        band=band,
        prediction_state=prediction_state,
        battery_factor=float(round(battery_factor, 3)),
        thermal_factor=float(round(thermal_factor, 3)),
        interruption_factor=float(round(interruption_factor, 3)),
        battery_level=(None if avg_battery is None else float(round(avg_battery, 1))),
        cpu_temp_c=(None if avg_temp is None else float(round(avg_temp, 1))),
        recent_interruptions=int(interruptions),
    )
