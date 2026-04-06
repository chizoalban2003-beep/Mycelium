from __future__ import annotations

import json
from datetime import datetime

from sqlmodel import Session, select

from mycelium_app.models import NexusPolicy
from mycelium_app.settings import settings


def default_policy() -> dict[str, object]:
    return {
        "privacy": {
            "export_enabled": bool(getattr(settings, "hive_export_enabled_default", False)),
        },
        "deny_sources": ["social", "social_media"],
        "allow_modalities": ["auto", "finance", "style", "grammar", "telemetry"],
        "intro": {
            "mode": str(getattr(settings, "nexus_intro_mode", "ask")),
            "observe_hours": int(getattr(settings, "nexus_observe_hours", 24)),
        },
    }


def _loads(s: str | None) -> dict[str, object]:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _dumps(v: dict[str, object]) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_policy(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_policy(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def normalize_policy(policy: dict[str, object]) -> dict[str, object]:
    base = default_policy()
    merged = merge_policy(base, policy or {})

    deny_sources = merged.get("deny_sources")
    if not isinstance(deny_sources, list):
        deny_sources = []
    merged["deny_sources"] = [str(x).strip().lower()[:32] for x in deny_sources if str(x).strip()][:50]

    allow_modalities = merged.get("allow_modalities")
    if not isinstance(allow_modalities, list):
        allow_modalities = base["allow_modalities"]
    merged["allow_modalities"] = [str(x).strip().lower()[:32] for x in allow_modalities if str(x).strip()][:50]

    privacy = merged.get("privacy")
    if not isinstance(privacy, dict):
        privacy = {}
    merged["privacy"] = {
        **(base.get("privacy") if isinstance(base.get("privacy"), dict) else {}),
        **privacy,
        "export_enabled": bool((privacy or {}).get("export_enabled", base["privacy"]["export_enabled"])),
    }

    intro = merged.get("intro")
    if not isinstance(intro, dict):
        intro = {}
    mode = str(intro.get("mode", base["intro"]["mode"])).strip().lower()
    if mode not in ("ask", "observe"):
        mode = str(base["intro"]["mode"])
    observe_hours = intro.get("observe_hours", base["intro"]["observe_hours"])
    try:
        observe_hours_i = int(observe_hours)
    except Exception:
        observe_hours_i = int(base["intro"]["observe_hours"])
    observe_hours_i = max(0, min(observe_hours_i, 168))
    merged["intro"] = {"mode": mode, "observe_hours": observe_hours_i}

    return merged


def get_policy(session: Session, user_id: int) -> dict[str, object]:
    row = session.exec(select(NexusPolicy).where(NexusPolicy.user_id == user_id)).first()
    if not row:
        return default_policy()
    return normalize_policy(_loads(row.policy_json))


def set_policy(session: Session, user_id: int, policy: dict[str, object]) -> dict[str, object]:
    normalized = normalize_policy(policy)
    row = session.exec(select(NexusPolicy).where(NexusPolicy.user_id == user_id)).first()
    now = datetime.utcnow()
    if not row:
        row = NexusPolicy(user_id=user_id, created_at=now, updated_at=now, policy_json=_dumps(normalized))
        session.add(row)
    else:
        row.updated_at = now
        row.policy_json = _dumps(normalized)
        session.add(row)
    session.commit()
    session.refresh(row)
    return normalized
