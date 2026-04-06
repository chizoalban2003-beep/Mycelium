from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from sqlmodel import Session

from mycelium_app.hive_empathy import queue_outbox_message
from mycelium_app.models import ExperienceBufferEntry
from mycelium_app.parental_policy import get_policy
from mycelium_app.settings import settings


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_WORD_RE = re.compile(r"[a-zA-Z0-9_\-\+\.]{2,}")


def normalize_concept(text: str, *, max_len: int = 80) -> str:
    """Normalize a short user-provided concept string.

    Keeps it compact and safe-ish:
    - trims
    - collapses whitespace
    - limits length
    """

    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > int(max_len):
        s = s[: int(max_len)].rstrip() + "…"
    return s


def concept_digest(tag: str, concept: str) -> str:
    obj = {"tag": str(tag or ""), "concept": str(concept or "")}
    return hashlib.sha256(_dumps(obj).encode("utf-8")).hexdigest()


def ionize_user_feedback(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    nudge_id: int | None,
    hint_tag: str,
    concept_text: str,
    action: str,
    export_to_hive: bool,
) -> dict[str, Any]:
    """Persist parent feedback locally and optionally queue a Hive message.

    Local persistence: ExperienceBufferEntry (source=ui, modality=curiosity_feedback)
    Hive export: outbox message kind=curiosity_concept (policy-gated)
    """

    hint_tag = str(hint_tag or "").strip()[:64]
    concept = normalize_concept(concept_text, max_len=120)
    action = str(action or "confirm").strip().lower()
    if action not in ("confirm", "correct"):
        action = "confirm"

    if not hint_tag:
        raise ValueError("hint_tag_required")
    if not concept:
        raise ValueError("concept_required")

    # Minimal tokenization for local searchability.
    tokens = [t.lower() for t in _WORD_RE.findall(concept)][:20]

    digest = concept_digest(hint_tag, concept)

    # Store locally.
    now = datetime.utcnow()
    entry = ExperienceBufferEntry(
        created_by_user_id=int(user_id),
        project_id=int(project_id) if project_id is not None else None,
        device_id=str(getattr(settings, "nexus_device_id", "local") or "local"),
        source="ui",
        modality="curiosity_feedback",
        raw_text=f"{action}:{hint_tag}:{concept}",
        extracted_json=_dumps(
            {
                "kind": "user_feedback_ionized",
                "action": action,
                "hint_tag": hint_tag,
                "concept": concept,
                "tokens": tokens,
                "nudge_id": (None if nudge_id is None else int(nudge_id)),
                "digest": digest,
                "created_at": now.isoformat() + "Z",
            }
        ),
        physics_used_json=_dumps({"nudge_id": nudge_id, "hint_tag": hint_tag}),
        confidence=1.0,
        feedback="",
        tags_json=_dumps(["feedback", action, hint_tag] + tokens[:5]),
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)

    exported = False
    export_reason: str | None = None

    # Optional export to Hive (policy-gated).
    if bool(export_to_hive):
        try:
            if not bool(getattr(settings, "hive_enabled", False)):
                export_reason = "hive_disabled"
            else:
                policy = get_policy(session, int(user_id))
                privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
                if not bool(privacy.get("export_enabled")):
                    export_reason = "export_disabled_by_policy"
                else:
                    payload = {
                        "meta": {
                            "created_at": now.isoformat() + "Z",
                            "kind": "curiosity_concept",
                            "version": "1",
                            "project_id": project_id,
                            "device_id": str(getattr(settings, "nexus_device_id", "local") or "local"),
                        },
                        "concept": {
                            "action": action,
                            "hint_tag": hint_tag,
                            # We include the concept text because the user explicitly provided it.
                            # Still keep it short and avoid any surrounding row context.
                            "text": concept,
                            "digest": digest,
                        },
                    }
                    queue_outbox_message(
                        session,
                        user_id=int(user_id),
                        project_id=project_id,
                        device_id=str(getattr(settings, "nexus_device_id", "local") or "local"),
                        kind="curiosity_concept",
                        payload=payload,
                    )
                    exported = True
        except Exception as e:
            export_reason = f"export_failed:{type(e).__name__}"

    return {
        "ok": True,
        "entry_uuid": str(entry.entry_uuid),
        "entry_id": int(entry.id or 0),
        "digest": str(digest),
        "exported_to_hive": bool(exported),
        "export_reason": export_reason,
    }
