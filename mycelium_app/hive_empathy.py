from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import re

from sqlmodel import Session, select

from mycelium_app.knowledge_sync import extract_recallable_kwargs
from mycelium_app.models import HiveGlobalUpdate, HiveOutboxMessage, HomeostasisState, PhysicsLedgerEntry
from mycelium_app.parental_policy import get_policy
from mycelium_app.privacy_membrane import check_hive_payload, redact_hive_payload
from mycelium_app.settings import settings


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_digest(obj: object) -> str:
    """SHA-256 of canonical JSON (used for change detection)."""

    return hashlib.sha256(_dumps(obj).encode("utf-8")).hexdigest()


def _loads_dict(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _get_latest_homeostasis(session: Session, *, user_id: int, project_id: int | None) -> HomeostasisState | None:
    q = select(HomeostasisState).where(HomeostasisState.user_id == int(user_id))
    if project_id is None:
        q = q.where(HomeostasisState.project_id.is_(None))
    else:
        q = q.where(HomeostasisState.project_id == int(project_id))
    q = q.order_by(HomeostasisState.updated_at.desc()).limit(1)
    return session.exec(q).first()


@dataclass(frozen=True)
class WisdomWhisper:
    payload: dict[str, Any]


def aggregate_recommended_kwargs(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of safe kwargs into a single recommended kwargs dict.

    - numbers -> median
    - strings/bools -> mode
    - otherwise -> stringified mode
    """

    by_key: dict[str, list[Any]] = {}
    for d in dicts:
        for k, v in d.items():
            by_key.setdefault(str(k), []).append(v)

    out: dict[str, Any] = {}
    for k, vals in by_key.items():
        if not vals:
            continue

        # numeric median
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            xs = sorted(float(v) for v in vals)
            mid = len(xs) // 2
            if len(xs) % 2 == 1:
                out[k] = xs[mid]
            else:
                out[k] = (xs[mid - 1] + xs[mid]) / 2.0
            continue

        # mode for simple types
        if all(isinstance(v, (str, bool)) or v is None for v in vals):
            c = Counter(vals)
            out[k] = c.most_common(1)[0][0]
            continue

        # fallback: stringify and mode
        svals = [str(v) for v in vals]
        c = Counter(svals)
        out[k] = c.most_common(1)[0][0]

    return out


def recommended_kwargs_from_whisper(whisper: dict[str, Any]) -> dict[str, Any]:
    """Extract (and re-filter) recommended kwargs from a wisdom_whisper payload."""

    if not isinstance(whisper, dict):
        return {}
    wisdom = whisper.get("wisdom") if isinstance(whisper.get("wisdom"), dict) else {}
    rec = wisdom.get("recommended_kwargs") if isinstance(wisdom.get("recommended_kwargs"), dict) else {}

    # Backward-compat: some callers store recommended_kwargs at the whisper top-level.
    if not rec:
        rec = whisper.get("recommended_kwargs") if isinstance(whisper.get("recommended_kwargs"), dict) else {}

    # Re-filter defensively: never trust imported global updates.
    try:
        return extract_recallable_kwargs(dict(rec))
    except Exception:
        return {}


def whisper_from_global_update(update_obj: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the whisper payload out of a HiveGlobalUpdate.update_json object."""

    if not isinstance(update_obj, dict):
        return None

    # Preferred wrapper form produced by /whisper/import.
    if str(update_obj.get("kind", "")) == "wisdom_whisper":
        w = update_obj.get("whisper")
        return w if isinstance(w, dict) else None

    # Fallback: some callers may store the whisper directly.
    meta = update_obj.get("meta") if isinstance(update_obj.get("meta"), dict) else {}
    if str(meta.get("kind", "")) == "wisdom_whisper":
        return update_obj

    return None


def curiosity_feedback_from_global_update(update_obj: dict[str, Any]) -> dict[str, Any] | None:
    """Pull curiosity_feedback payload out of a HiveGlobalUpdate.update_json object."""

    if not isinstance(update_obj, dict):
        return None

    # Preferred wrapper form produced by /curiosity/import.
    if str(update_obj.get("kind", "")) == "curiosity_feedback":
        fb = update_obj.get("feedback")
        return fb if isinstance(fb, dict) else None

    # Fallback: store the feedback directly.
    meta = update_obj.get("meta") if isinstance(update_obj.get("meta"), dict) else {}
    if str(meta.get("kind", "")) == "curiosity_feedback":
        return update_obj

    return None


def curiosity_concept_from_global_update(update_obj: dict[str, Any]) -> dict[str, Any] | None:
    """Pull curiosity_concept payload out of a HiveGlobalUpdate.update_json object."""

    if not isinstance(update_obj, dict):
        return None

    # Preferred wrapper form produced by /curiosity/concept/import.
    if str(update_obj.get("kind", "")) == "curiosity_concept":
        c = update_obj.get("concept")
        return c if isinstance(c, dict) else None

    # Fallback: store the concept directly.
    meta = update_obj.get("meta") if isinstance(update_obj.get("meta"), dict) else {}
    if str(meta.get("kind", "")) == "curiosity_concept":
        return update_obj

    return None


@dataclass(frozen=True)
class WisdomLatest:
    as_of: datetime | None
    n_updates_considered: int
    n_whispers_used: int
    n_curiosity_feedback_used: int
    recommended_kwargs: dict[str, Any]
    evidence: dict[str, Any]


def compute_wisdom_latest(
    session: Session,
    *,
    project_id: int | None,
    include_project_scoped: bool,
    limit: int,
) -> WisdomLatest:
    limit = max(1, min(int(limit), 200))
    include_project_scoped = bool(include_project_scoped)

    q = select(HiveGlobalUpdate).order_by(HiveGlobalUpdate.created_at.desc()).limit(limit)
    rows = session.exec(q).all()

    recs: list[dict[str, Any]] = []
    used_update_uuids: list[str] = []
    as_of: datetime | None = None

    curiosity_tags: Counter[str] = Counter()
    curiosity_concepts: Counter[str] = Counter()
    n_feedback_used = 0
    n_concepts_used = 0

    word_re = re.compile(r"[a-zA-Z0-9_\-\+\.]{2,}")

    for r in rows:
        update_obj = _loads_dict(r.update_json)

        fb = curiosity_feedback_from_global_update(update_obj)
        if fb:
            meta_fb = fb.get("meta") if isinstance(fb.get("meta"), dict) else {}
            fb_project_id = meta_fb.get("project_id")
            if project_id is not None:
                if fb_project_id != project_id:
                    fb = None
            else:
                if (fb_project_id is not None) and not include_project_scoped:
                    fb = None

        if fb:
            feedback_obj = fb.get("feedback") if isinstance(fb.get("feedback"), dict) else {}
            tags = feedback_obj.get("tags") if isinstance(feedback_obj.get("tags"), list) else []
            for t in tags[:50]:
                ts = str(t).strip()
                if ts:
                    curiosity_tags[ts] += 1
            n_feedback_used += 1

        concept = curiosity_concept_from_global_update(update_obj)
        if concept:
            meta_c = concept.get("meta") if isinstance(concept.get("meta"), dict) else {}
            c_project_id = meta_c.get("project_id")
            if project_id is not None:
                if c_project_id != project_id:
                    concept = None
            else:
                if (c_project_id is not None) and not include_project_scoped:
                    concept = None

        if concept:
            obj = concept.get("concept") if isinstance(concept.get("concept"), dict) else {}
            action = str(obj.get("action", "confirm")).strip().lower()
            hint_tag = str(obj.get("hint_tag", "")).strip()
            text = str(obj.get("text", "")).strip()

            # Corrections are treated as new tag signals; confirmations reinforce existing tags.
            if action == "correct" and text:
                toks = [t.lower() for t in word_re.findall(text)][:10]
                for t in toks:
                    curiosity_tags[t] += 1
                if toks:
                    curiosity_concepts[" ".join(toks[:3])] += 1
            else:
                if hint_tag:
                    curiosity_tags[hint_tag] += 1
                if text:
                    curiosity_concepts[text[:80]] += 1
            n_concepts_used += 1

        whisper = whisper_from_global_update(update_obj)
        if not whisper:
            continue

        meta = whisper.get("meta") if isinstance(whisper.get("meta"), dict) else {}
        w_project_id = meta.get("project_id")
        if project_id is not None:
            if w_project_id != project_id:
                continue
        else:
            if (w_project_id is not None) and not include_project_scoped:
                continue

        rec = recommended_kwargs_from_whisper(whisper)
        if rec:
            recs.append(rec)
            used_update_uuids.append(str(r.update_uuid))
            if as_of is None:
                as_of = r.created_at

    merged = aggregate_recommended_kwargs(recs) if recs else {}
    evidence = {
        "update_uuids": used_update_uuids[:50],
        "limit": int(limit),
        "include_project_scoped": bool(include_project_scoped),
        "curiosity": {
            "n_feedback_used": int(n_feedback_used),
            "n_concepts_used": int(n_concepts_used),
            "top_tags": [{"tag": k, "count": int(v)} for k, v in curiosity_tags.most_common(10)],
            "top_concepts": [{"concept": k, "count": int(v)} for k, v in curiosity_concepts.most_common(5)],
        },
    }

    return WisdomLatest(
        as_of=as_of,
        n_updates_considered=int(len(rows)),
        n_whispers_used=int(len(recs)),
        n_curiosity_feedback_used=int(n_feedback_used),
        recommended_kwargs=merged,
        evidence=evidence,
    )


def summarize_kwargs_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old = old or {}
    new = new or {}
    keys = set(old.keys()) | set(new.keys())
    changed = 0
    max_rel = 0.0
    for k in keys:
        a = old.get(k)
        b = new.get(k)
        if a == b:
            continue
        changed += 1
        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) != 0.0:
                rel = abs(float(b) - float(a)) / abs(float(a))
                if rel > max_rel:
                    max_rel = rel
        except Exception:
            pass
    return {"changed_keys": int(changed), "max_rel_change": float(max_rel)}


def build_wisdom_whisper_from_physics_ledger(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    device_id: str,
    limit: int = 200,
) -> WisdomWhisper:
    """Build a "wisdom whisper" describing stable, high-scoring physics kwargs.

    Privacy stance:
    - does NOT include raw feature names or raw data
    - exports only allowlisted physics/cleaning knobs (via extract_recallable_kwargs)
    - includes only coarse evidence summary stats
    """

    limit = max(5, min(int(limit), 2000))

    stmt = (
        select(PhysicsLedgerEntry)
        .where(PhysicsLedgerEntry.created_by_user_id == int(user_id))
        .order_by(PhysicsLedgerEntry.score_value.desc(), PhysicsLedgerEntry.created_at.desc())
        .limit(int(limit))
    )
    if project_id is not None:
        stmt = stmt.where(PhysicsLedgerEntry.project_id == int(project_id))

    rows = session.exec(stmt).all()

    safe_kwargs: list[dict[str, Any]] = []
    score_values: list[float] = []
    metric_counts: Counter[str] = Counter()
    target_kind_counts: Counter[str] = Counter()

    for r in rows:
        raw = _loads_dict(r.applied_kwargs_json)
        safe = extract_recallable_kwargs(raw if isinstance(raw, dict) else {})
        if safe:
            safe_kwargs.append(safe)
        score_values.append(float(r.score_value or 0.0))
        metric_counts[str(r.score_metric or "")] += 1
        target_kind_counts[str(r.target_kind or "")] += 1

    rec = aggregate_recommended_kwargs(safe_kwargs)

    top_score = max(score_values) if score_values else 0.0
    avg_score = (sum(score_values) / float(len(score_values))) if score_values else 0.0

    hs = _get_latest_homeostasis(session, user_id=user_id, project_id=project_id)
    mood = str(hs.mood) if hs else "unknown"
    identity_hash = str(hs.identity_hash) if hs else ""

    now = datetime.utcnow()
    payload: dict[str, Any] = {
        "meta": {
            "created_at": now.isoformat() + "Z",
            "device_id": str(device_id or "local"),
            "project_id": project_id,
            "kind": "wisdom_whisper",
            "version": "1",
        },
        "homeostasis": {
            "mood": mood,
            # Identity hash is already exported by the identity_backup mechanism.
            "identity_hash": identity_hash,
        },
        "wisdom": {
            "recommended_kwargs": rec,
            "evidence": {
                "n_entries_considered": int(len(rows)),
                "n_entries_with_safe_kwargs": int(len(safe_kwargs)),
                "score_metric_counts": dict(metric_counts),
                "target_kind_counts": dict(target_kind_counts),
                "top_score_value": float(top_score),
                "avg_score_value": float(avg_score),
            },
        },
    }

    return WisdomWhisper(payload=payload)


def queue_outbox_message(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    device_id: str,
    kind: str,
    payload: dict[str, Any],
) -> int:
    redacted, _changed = redact_hive_payload(payload or {})
    if not isinstance(redacted, dict):
        redacted = {}

    if _changed:
        meta = redacted.get("meta")
        if isinstance(meta, dict):
            privacy = meta.get("privacy")
            if not isinstance(privacy, dict):
                privacy = {}
            privacy["redacted"] = True
            meta["privacy"] = privacy
            redacted["meta"] = meta

    membrane = check_hive_payload(redacted)
    if not membrane.ok:
        raise ValueError(f"privacy_membrane_blocked:{str(kind or '')[:64]}:{'|'.join(membrane.reasons[:4])}")

    row = HiveOutboxMessage(
        created_by_user_id=int(user_id),
        project_id=int(project_id) if project_id is not None else None,
        device_id=str(device_id or "local"),
        kind=str(kind or "")[:64],
        payload_json=_dumps(redacted),
        submitted_at=None,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return int(row.id or 0)


def queue_wisdom_whisper(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    device_id: str,
    limit: int = 200,
) -> tuple[int | None, str | None]:
    if not bool(settings.hive_enabled):
        return None, "hive_disabled"

    policy = get_policy(session, int(user_id))
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    if not bool(privacy.get("export_enabled")):
        return None, "export_disabled_by_policy"

    whisper = build_wisdom_whisper_from_physics_ledger(
        session,
        user_id=int(user_id),
        project_id=project_id,
        device_id=str(device_id or "local"),
        limit=int(limit),
    )
    try:
        message_id = queue_outbox_message(
            session,
            user_id=int(user_id),
            project_id=project_id,
            device_id=str(device_id or "local"),
            kind="wisdom_whisper",
            payload=whisper.payload,
        )
        return message_id, None
    except ValueError:
        return None, "privacy_membrane_blocked"


def queue_homeostasis_failure(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    device_id: str,
    error_type: str,
    error_message: str | None = None,
    min_interval_minutes: int = 30,
) -> tuple[int | None, str | None]:
    """Queue a throttled homeostasis failure message ("ask parent")."""

    if not bool(settings.hive_enabled):
        return None, "hive_disabled"

    policy = get_policy(session, int(user_id))
    privacy = policy.get("privacy") if isinstance(policy.get("privacy"), dict) else {}
    if not bool(privacy.get("export_enabled")):
        return None, "export_disabled_by_policy"

    # Throttle: only one recent failure message.
    min_interval_minutes = max(1, min(int(min_interval_minutes), 24 * 60))
    since = datetime.utcnow() - timedelta(minutes=min_interval_minutes)

    q = (
        select(HiveOutboxMessage)
        .where(HiveOutboxMessage.created_by_user_id == int(user_id))
        .where(HiveOutboxMessage.kind == "homeostasis_failure")
        .where(HiveOutboxMessage.created_at >= since)
        .order_by(HiveOutboxMessage.created_at.desc())
        .limit(1)
    )
    if project_id is not None:
        q = q.where(HiveOutboxMessage.project_id == int(project_id))

    recent = session.exec(q).first()
    if recent is not None:
        return None, "throttled"

    now = datetime.utcnow()
    payload: dict[str, Any] = {
        "meta": {
            "created_at": now.isoformat() + "Z",
            "device_id": str(device_id or "local"),
            "project_id": project_id,
            "kind": "homeostasis_failure",
            "version": "1",
        },
        "failure": {
            "error_type": str(error_type or "error")[:128],
            "error_message": (str(error_message)[:500] if error_message else None),
            "action": "ask_parent",
        },
    }

    try:
        message_id = queue_outbox_message(
            session,
            user_id=int(user_id),
            project_id=project_id,
            device_id=str(device_id or "local"),
            kind="homeostasis_failure",
            payload=payload,
        )
        return message_id, None
    except ValueError:
        return None, "privacy_membrane_blocked"
