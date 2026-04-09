from __future__ import annotations

from datetime import datetime

from sqlmodel import Session, select

from mycelium_app.models import AssistantAvatarProfile, AssistantProfile, User


def _normalize_name(v: str | None) -> str:
    s = str(v or "").strip()
    return s[:64] if s else "Synapse"


def _normalize_gender(v: str | None) -> str:
    s = str(v or "").strip().lower()
    allowed = {"neutral", "female", "male", "nonbinary", "custom"}
    return s if s in allowed else "neutral"


def _normalize_voice(v: str | None) -> str:
    s = str(v or "").strip().lower()
    return s[:64] if s else "alloy"


def _normalize_avatar_url(v: str | None) -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    sl = s.lower()
    if sl.startswith("http://") or sl.startswith("https://"):
        return s[:500]
    return ""


def _get_avatar_row(session: Session, *, user_id: int, project_id: int | None) -> AssistantAvatarProfile | None:
    if project_id is not None:
        row = session.exec(
            select(AssistantAvatarProfile).where(
                AssistantAvatarProfile.user_id == int(user_id),
                AssistantAvatarProfile.project_id == int(project_id),
            )
        ).first()
        if row is not None:
            return row
    return session.exec(
        select(AssistantAvatarProfile).where(
            AssistantAvatarProfile.user_id == int(user_id),
            AssistantAvatarProfile.project_id.is_(None),
        )
    ).first()


def get_assistant_profile(session: Session, *, user_id: int, project_id: int | None = None) -> AssistantProfile | None:
    if project_id is not None:
        row = session.exec(
            select(AssistantProfile).where(
                AssistantProfile.user_id == int(user_id),
                AssistantProfile.project_id == int(project_id),
            )
        ).first()
        if row is not None:
            return row

    return session.exec(
        select(AssistantProfile).where(
            AssistantProfile.user_id == int(user_id),
            AssistantProfile.project_id.is_(None),
        )
    ).first()


def _resolve_gender_from_user(session: Session, *, user_id: int) -> str:
    """If the user has set their gender, mirror it to the companion."""
    user = session.get(User, int(user_id))
    if user and str(getattr(user, "gender", "") or "").strip():
        return _normalize_gender(user.gender)
    return "neutral"


def get_assistant_profile_effective(session: Session, *, user_id: int, project_id: int | None = None) -> dict[str, object]:
    row = get_assistant_profile(session, user_id=user_id, project_id=project_id)
    avatar_row = _get_avatar_row(session, user_id=user_id, project_id=project_id)
    avatar_url = ""
    if avatar_row is not None:
        avatar_url = _normalize_avatar_url(avatar_row.assistant_avatar_url)

    # Mirror user gender when no explicit companion gender has been set
    mirrored_gender = _resolve_gender_from_user(session, user_id=user_id)

    if row is None:
        return {
            "project_id": project_id,
            "given_name": "Synapse",
            "gender_identity": mirrored_gender,
            "vocal_preset": "alloy",
            "assistant_avatar_url": avatar_url,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_default": True,
        }

    effective_gender = _normalize_gender(row.gender_identity)
    if effective_gender == "neutral" and mirrored_gender != "neutral":
        effective_gender = mirrored_gender

    return {
        "project_id": row.project_id,
        "given_name": _normalize_name(row.given_name),
        "gender_identity": effective_gender,
        "vocal_preset": _normalize_voice(row.vocal_preset),
        "assistant_avatar_url": avatar_url,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "is_default": False,
    }


def set_assistant_profile(
    session: Session,
    *,
    user_id: int,
    project_id: int | None,
    given_name: str,
    gender_identity: str,
    vocal_preset: str,
    assistant_avatar_url: str | None = None,
) -> AssistantProfile:
    now = datetime.utcnow()

    row = session.exec(
        select(AssistantProfile).where(
            AssistantProfile.user_id == int(user_id),
            AssistantProfile.project_id == (None if project_id is None else int(project_id)),
        )
    ).first()

    if row is None:
        row = AssistantProfile(
            user_id=int(user_id),
            project_id=(None if project_id is None else int(project_id)),
            created_at=now,
            updated_at=now,
            given_name=_normalize_name(given_name),
            gender_identity=_normalize_gender(gender_identity),
            vocal_preset=_normalize_voice(vocal_preset),
        )
    else:
        row.updated_at = now
        row.given_name = _normalize_name(given_name)
        row.gender_identity = _normalize_gender(gender_identity)
        row.vocal_preset = _normalize_voice(vocal_preset)

    session.add(row)

    avatar_row = session.exec(
        select(AssistantAvatarProfile).where(
            AssistantAvatarProfile.user_id == int(user_id),
            AssistantAvatarProfile.project_id == (None if project_id is None else int(project_id)),
        )
    ).first()
    avatar_url = _normalize_avatar_url(assistant_avatar_url)
    if avatar_row is None:
        avatar_row = AssistantAvatarProfile(
            user_id=int(user_id),
            project_id=(None if project_id is None else int(project_id)),
            created_at=now,
            updated_at=now,
            assistant_avatar_url=avatar_url,
        )
    else:
        avatar_row.updated_at = now
        avatar_row.assistant_avatar_url = avatar_url
    session.add(avatar_row)

    session.commit()
    session.refresh(row)
    return row
