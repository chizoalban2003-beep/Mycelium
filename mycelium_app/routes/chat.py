from __future__ import annotations

import json
from datetime import datetime
from urllib import parse, request

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlmodel import Session, select

from mycelium_app.assistant_profile import get_assistant_profile_effective
from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import ConversationMessage, NexusNudge, NexusPolicy, ProjectMember, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import ChatHistoryResponse, ChatMessagePublic, ChatSendRequest, ChatSendResponse
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/nexus/chat", tags=["chat"])


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _ensure_project_access(session: Session, user_id: int, project_id: int | None) -> None:
    if project_id is None:
        return
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == int(project_id), ProjectMember.user_id == int(user_id))
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")


def _to_public(row: ConversationMessage) -> ChatMessagePublic:
    return ChatMessagePublic(
        id=int(row.id or 0),
        created_at=row.created_at,
        project_id=row.project_id,
        conversation_key=str(row.conversation_key or "default"),
        channel=str(row.channel or "app"),
        role=str(row.role or "assistant"),
        content=str(row.content or ""),
        metadata=_loads_dict(row.metadata_json),
    )


def _send_telegram(*, bot_token: str, chat_id: str, text: str) -> bool:
    token = str(bot_token or "").strip()
    cid = str(chat_id or "").strip()
    if not token or not cid:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = parse.urlencode(
        {
            "chat_id": cid,
            "text": text[:4096],
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = request.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with request.urlopen(req, timeout=10) as resp:  # nosec B310
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except Exception:
        return False


def _resolve_user_id_from_telegram_chat_id(session: Session, chat_id: str) -> int | None:
    cid = str(chat_id or "").strip()
    if not cid:
        return None

    rows = session.exec(select(NexusPolicy)).all()
    for row in rows:
        policy = get_policy(session, int(row.user_id))
        notif = policy.get("notifications") if isinstance(policy.get("notifications"), dict) else {}
        if not bool(notif.get("telegram_enabled", False)):
            continue
        mapped_chat_id = str(notif.get("telegram_chat_id") or "").strip()
        if mapped_chat_id and mapped_chat_id == cid:
            return int(row.user_id)
    return None


@router.post("/send", response_model=ChatSendResponse)
def send_chat(
    payload: ChatSendRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, payload.project_id)

    msg = str(payload.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    if len(msg) > 5000:
        raise HTTPException(status_code=413, detail="message too large")

    channel = str(payload.channel or "app").strip().lower()
    if channel not in {"app", "telegram"}:
        channel = "app"

    conv_key = str(payload.conversation_key or "default").strip()[:64] or "default"

    user_row = ConversationMessage(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        conversation_key=conv_key,
        channel=channel,
        role="user",
        content=msg,
        metadata_json=_dumps({}),
    )
    session.add(user_row)
    session.flush()

    ap = get_assistant_profile_effective(session, user_id=user_id, project_id=payload.project_id)
    assistant_name = str(ap.get("given_name", "Synapse")).strip() or "Synapse"

    # Deterministic lightweight assistant response (tool-free skeleton).
    assistant_text = (
        f"I’m {assistant_name}. I received your message. "
        "I can help with focus workflows, telemetry insights, and task replicas. "
        "If you want, ask me to bootstrap or verify your next work session."
    )

    assistant_row = ConversationMessage(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        conversation_key=conv_key,
        channel=channel,
        role="assistant",
        content=assistant_text,
        metadata_json=_dumps({"assistant_name": assistant_name}),
    )
    session.add(assistant_row)
    session.flush()

    delivered_external = False
    if channel == "telegram":
        policy = get_policy(session, user_id)
        notif = policy.get("notifications") if isinstance(policy.get("notifications"), dict) else {}
        if bool(getattr(settings, "notifications_bridge_enabled", False)) and bool(notif.get("telegram_enabled", False)):
            delivered_external = _send_telegram(
                bot_token=str(getattr(settings, "notifications_telegram_bot_token", "") or ""),
                chat_id=str(notif.get("telegram_chat_id") or ""),
                text=assistant_text,
            )

    n = NexusNudge(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        kind="chat_message",
        title=f"Message from {assistant_name}",
        message=assistant_text,
        payload_json=_dumps({"conversation_key": conv_key, "channel": channel}),
    )
    session.add(n)

    now = datetime.utcnow()
    user_row.delivered_at = now
    assistant_row.delivered_at = now
    session.add(user_row)
    session.add(assistant_row)
    session.commit()
    session.refresh(user_row)
    session.refresh(assistant_row)

    return ChatSendResponse(
        ok=True,
        user_message=_to_public(user_row),
        assistant_message=_to_public(assistant_row),
        delivered_external=bool(delivered_external),
    )


@router.get("/history", response_model=ChatHistoryResponse)
def history(
    limit: int = 50,
    project_id: int | None = None,
    conversation_key: str = "default",
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_id = int(current_user.id or 0)
    _ensure_project_access(session, user_id, project_id)

    lim = max(1, min(int(limit), 500))
    conv_key = str(conversation_key or "default").strip()[:64] or "default"

    q = select(ConversationMessage).where(ConversationMessage.created_by_user_id == user_id)
    q = q.where(ConversationMessage.conversation_key == conv_key)
    if project_id is None:
        q = q.where(ConversationMessage.project_id.is_(None))
    else:
        q = q.where(ConversationMessage.project_id == int(project_id))
    q = q.order_by(ConversationMessage.created_at.desc()).limit(lim)

    rows = list(reversed(session.exec(q).all()))
    return ChatHistoryResponse(messages=[_to_public(r) for r in rows])


@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
    session: Session = Depends(get_session),
):
    expected = str(getattr(settings, "notifications_telegram_webhook_secret", "") or "").strip()
    if expected:
        got = str(x_telegram_bot_api_secret_token or "").strip()
        if got != expected:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    try:
        payload = await request.json()
    except Exception:
        return {"ok": True, "ignored": "invalid_json"}

    if not isinstance(payload, dict):
        return {"ok": True, "ignored": "invalid_payload"}

    msg = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    text = str(msg.get("text") or "").strip()
    chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
    chat_id = str(chat.get("id") or "").strip()
    if not text or not chat_id:
        return {"ok": True, "ignored": "unsupported_message"}

    user_id = _resolve_user_id_from_telegram_chat_id(session, chat_id)
    if not user_id:
        return {"ok": True, "ignored": "unmapped_chat"}

    conv_key = f"telegram:{chat_id}"[:64]

    user_row = ConversationMessage(
        created_by_user_id=int(user_id),
        project_id=None,
        conversation_key=conv_key,
        channel="telegram",
        role="user",
        content=text[:5000],
        metadata_json=_dumps({"source": "telegram_webhook"}),
    )
    session.add(user_row)
    session.flush()

    ap = get_assistant_profile_effective(session, user_id=int(user_id), project_id=None)
    assistant_name = str(ap.get("given_name", "Synapse")).strip() or "Synapse"

    lower = text.lower()
    if "bootstrap" in lower:
        assistant_text = (
            f"I’m {assistant_name}. I can bootstrap a focus session. "
            "Use /api/nexus/tasks/bootstrap/work-session in the app to start it with your preferred duration."
        )
    elif "verify" in lower:
        assistant_text = (
            f"I’m {assistant_name}. I can verify outcomes after execution. "
            "Use /api/nexus/tasks/replicas/{id}/verify with adherence and outcome details."
        )
    else:
        assistant_text = (
            f"I’m {assistant_name}. Message received. "
            "I can help with work-session planning, task replicas, and telemetry insight."
        )

    assistant_row = ConversationMessage(
        created_by_user_id=int(user_id),
        project_id=None,
        conversation_key=conv_key,
        channel="telegram",
        role="assistant",
        content=assistant_text,
        metadata_json=_dumps({"assistant_name": assistant_name, "source": "telegram_webhook"}),
    )
    session.add(assistant_row)
    session.flush()

    delivered = False
    if bool(getattr(settings, "notifications_bridge_enabled", False)):
        delivered = _send_telegram(
            bot_token=str(getattr(settings, "notifications_telegram_bot_token", "") or ""),
            chat_id=chat_id,
            text=assistant_text,
        )

    now = datetime.utcnow()
    user_row.delivered_at = now
    assistant_row.delivered_at = now if delivered else None
    session.add(user_row)
    session.add(assistant_row)
    session.commit()

    return {"ok": True, "delivered": bool(delivered)}
