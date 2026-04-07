from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from sqlmodel import Session

from mycelium_app.models import SignalLedgerEvent


_SECRET_KEY_HINTS = (
    "password",
    "passcode",
    "token",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "session",
    "cookie",
    "auth",
)


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_scalar(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            text = value.strip()
            if len(text) > 160:
                text = text[:160].rstrip() + "…"
            return text
        return value
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    return str(value)[:160]


def _should_redact_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def _sanitize_payload(value: Any, *, depth: int = 0, max_depth: int = 2, max_list_items: int = 10) -> Any:
    if depth >= max_depth:
        return _normalize_scalar(value)

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 50:
                out["__truncated__"] = True
                break
            key_text = str(key)[:80]
            if _should_redact_key(key_text):
                out[key_text] = "[redacted]"
            else:
                out[key_text] = _sanitize_payload(item, depth=depth + 1, max_depth=max_depth, max_list_items=max_list_items)
        return out

    if isinstance(value, list):
        out_list: list[Any] = []
        for item in value[:max_list_items]:
            out_list.append(_sanitize_payload(item, depth=depth + 1, max_depth=max_depth, max_list_items=max_list_items))
        if len(value) > max_list_items:
            out_list.append("[truncated]")
        return out_list

    return _normalize_scalar(value)


def _flatten(value: Any, *, prefix: str = "payload", depth: int = 0, max_depth: int = 2) -> dict[str, Any]:
    rows: dict[str, Any] = {}

    if depth >= max_depth:
        rows[prefix] = _normalize_scalar(value)
        return rows

    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{str(key)[:64]}"
            if isinstance(item, dict):
                rows.update(_flatten(item, prefix=child, depth=depth + 1, max_depth=max_depth))
            elif isinstance(item, list):
                rows[f"{child}.len"] = len(item)
                scalar_items = [x for x in item if isinstance(x, (bool, int, float, str))][:5]
                if scalar_items:
                    rows[f"{child}.sample"] = "|".join(str(_normalize_scalar(x)) for x in scalar_items)
            else:
                rows[child] = _normalize_scalar(item)
        return rows

    if isinstance(value, list):
        rows[f"{prefix}.len"] = len(value)
        scalar_items = [x for x in value if isinstance(x, (bool, int, float, str))][:5]
        if scalar_items:
            rows[f"{prefix}.sample"] = "|".join(str(_normalize_scalar(x)) for x in scalar_items)
        return rows

    rows[prefix] = _normalize_scalar(value)
    return rows


def _walk_stats(value: Any, *, depth: int = 0) -> dict[str, int]:
    stats = {
        "dicts": 0,
        "lists": 0,
        "scalars": 0,
        "strings": 0,
        "numbers": 0,
        "bools": 0,
        "nulls": 0,
        "max_depth": depth,
        "leaf_count": 0,
    }

    def visit(obj: Any, current_depth: int) -> None:
        stats["max_depth"] = max(stats["max_depth"], current_depth)
        if obj is None:
            stats["nulls"] += 1
            stats["scalars"] += 1
            stats["leaf_count"] += 1
            return
        if isinstance(obj, bool):
            stats["bools"] += 1
            stats["scalars"] += 1
            stats["leaf_count"] += 1
            return
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            stats["numbers"] += 1
            stats["scalars"] += 1
            stats["leaf_count"] += 1
            return
        if isinstance(obj, str):
            stats["strings"] += 1
            stats["scalars"] += 1
            stats["leaf_count"] += 1
            return
        if isinstance(obj, dict):
            stats["dicts"] += 1
            for item in obj.values():
                visit(item, current_depth + 1)
            return
        if isinstance(obj, list):
            stats["lists"] += 1
            for item in obj:
                visit(item, current_depth + 1)
            return
        stats["scalars"] += 1
        stats["leaf_count"] += 1

    visit(value, depth)
    return stats


def _safe_digest(value: Any) -> str:
    try:
        payload = _sanitize_payload(value, max_depth=3, max_list_items=10)
        return hashlib.sha256(_dumps(payload).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(str(datetime.utcnow().isoformat()).encode("utf-8")).hexdigest()


def build_stimulus_tabular_payload(
    *,
    stimulus: Any,
    source: str,
    modality: str,
    signal_type: str,
    device_id: str,
    project_id: int | None,
    occurred_at: datetime,
) -> dict[str, Any]:
    """Convert arbitrary app stimulus into a safe, tabular learning envelope."""

    safe_payload = _sanitize_payload(stimulus, max_depth=2, max_list_items=10)
    stats = _walk_stats(stimulus)
    flat = _flatten(safe_payload, prefix="stimulus", depth=0, max_depth=2)

    text_repr = _dumps(safe_payload)
    tabular = {
        "source": str(source or "").strip()[:32],
        "modality": str(modality or "").strip()[:32],
        "signal_type": str(signal_type or "").strip()[:64],
        "device_id": str(device_id or "").strip()[:64],
        "project_id": project_id,
        "occurred_at": occurred_at.isoformat() + "Z",
        "payload_kind": type(stimulus).__name__.lower(),
        "payload_digest": _safe_digest(stimulus),
        "payload_text_length": int(len(text_repr)),
        "payload_key_count": int(len(stimulus)) if isinstance(stimulus, dict) else 0,
        "payload_list_length": int(len(stimulus)) if isinstance(stimulus, list) else 0,
        "payload_leaf_count": int(stats["leaf_count"]),
        "payload_scalar_count": int(stats["scalars"]),
        "payload_dict_count": int(stats["dicts"]),
        "payload_list_count": int(stats["lists"]),
        "payload_string_count": int(stats["strings"]),
        "payload_numeric_count": int(stats["numbers"]),
        "payload_bool_count": int(stats["bools"]),
        "payload_null_count": int(stats["nulls"]),
        "payload_max_depth": int(stats["max_depth"]),
    }
    tabular.update(flat)

    return {
        "meta": {
            "source": tabular["source"],
            "modality": tabular["modality"],
            "signal_type": tabular["signal_type"],
            "payload_kind": tabular["payload_kind"],
            "payload_digest": tabular["payload_digest"],
        },
        "surface": safe_payload,
        "tabular": tabular,
    }


def recommend_learning_profile(*, stimulus: Any, signal_type: str, modality: str) -> dict[str, Any]:
    """Describe how a stimulus should be encoded and evaluated."""

    kind = type(stimulus).__name__.lower()
    if isinstance(stimulus, dict):
        family = "structured"
        encoder = "tabular"
    elif isinstance(stimulus, list):
        family = "sequence"
        encoder = "sequence_to_tabular"
    elif isinstance(stimulus, str):
        family = "text"
        encoder = "text_to_tabular"
    elif isinstance(stimulus, (bytes, bytearray, memoryview)):
        family = "binary"
        encoder = "binary_summary"
    else:
        family = "generic"
        encoder = "tabular"

    if family in {"text", "sequence", "binary"}:
        metrics = ["accuracy", "f1_macro", "balanced_accuracy", "log_loss"]
        if family == "binary":
            metrics.append("auroc")
    else:
        metrics = ["mae", "rmse", "r2", "median_ae"]

    return {
        "signal_kind": kind,
        "signal_type": str(signal_type or "").strip()[:64],
        "modality": str(modality or "").strip()[:32],
        "family": family,
        "encoder": encoder,
        "recommended_metrics": metrics,
        "recommended_model_head": "classifier" if family in {"text", "sequence", "binary"} else "regressor",
        "preserve_physics_core": True,
    }


def record_stimulus_event(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    device_id: str,
    source: str,
    modality: str,
    signal_type: str,
    stimulus: Any,
    occurred_at: datetime,
) -> tuple[SignalLedgerEvent, dict[str, Any]]:
    envelope = build_stimulus_tabular_payload(
        stimulus=stimulus,
        source=source,
        modality=modality,
        signal_type=signal_type,
        device_id=device_id,
        project_id=project_id,
        occurred_at=occurred_at,
    )
    learning_profile = recommend_learning_profile(stimulus=stimulus, signal_type=signal_type, modality=modality)
    payload = {"kind": "digital_stimulus", **envelope, "learning_profile": learning_profile}

    row = SignalLedgerEvent(
        created_at=occurred_at,
        created_by_user_id=int(user_id),
        project_id=project_id,
        device_id=str(device_id or "local")[:64],
        signal_type=str(signal_type or "stimulus")[:64],
        payload_json=_dumps(payload),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row, payload
