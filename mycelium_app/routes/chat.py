from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from urllib import parse, request

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlmodel import Session, select

from mycelium_app.assistant_profile import get_assistant_profile_effective
from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import (
    ConversationMessage,
    HiveOutboxMessage,
    HiveDevice,
    NexusNudge,
    NexusPolicy,
    ProjectMember,
    SignalLedgerEvent,
    User,
)
from mycelium_app.parental_policy import get_policy, set_policy
from mycelium_app.routes.hybrid import auto_handoff_confirm, auto_handoff_launch
from mycelium_app.routes.live import prune_mission_log
from mycelium_app.stimulus import record_stimulus_event
from mycelium_app.schemas import (
    AutoHandoffConfirmRequest,
    AutoHandoffLaunchRequest,
    ChatHistoryResponse,
    ChatMessagePublic,
    ChatSendRequest,
    ChatSendResponse,
    ChatToneCheckResponse,
    MissionLogPruneRequest,
)
from mycelium_app.settings import settings
from mycelium_app.self_reflection import compute_daily_consolidation
from mycelium_app.viscosity import calculate_live_viscosity


router = APIRouter(prefix="/api/nexus/chat", tags=["chat"])


_PERSONA_MARKERS: dict[str, list[str]] = {
    "coach": ["let's", "next step", "you can", "momentum", "goal"],
    "calm": ["steady", "breathe", "calm", "pace", "gentle"],
    "briefing": ["status", "summary", "risk", "next", "decision"],
}


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


def _parse_mission_log_command(text: str) -> tuple[str, int | None, int | None] | None:
    raw = str(text or '').strip().lower()
    if not raw.startswith('/'):
        return None

    parts = [part for part in raw.split() if part]
    if not parts:
        return None

    command = parts[0].split('@', 1)[0]
    if command not in {'/clear_logs', '/prune'}:
        return None

    project_id: int | None = None
    hours: int | None = None
    saw_explicit_argument = False
    saw_recognized_value = False

    index = 1
    while index < len(parts):
        token = parts[index]
        if token in {'project', 'project_id'} and index + 1 < len(parts):
            saw_explicit_argument = True
            candidate = ''.join(ch for ch in parts[index + 1] if ch.isdigit())
            if candidate:
                try:
                    project_id = int(candidate)
                    saw_recognized_value = True
                except Exception:
                    project_id = None
            index += 2
            continue

        candidate = ''.join(ch for ch in token if ch.isdigit())
        if candidate:
            saw_explicit_argument = True
            try:
                value = int(candidate)
            except Exception:
                value = None
            if value is not None:
                saw_recognized_value = True
                if command == '/prune' and hours is None:
                    hours = value
                elif project_id is None:
                    project_id = value
        elif token:
            saw_explicit_argument = True
        index += 1

    if command == '/clear_logs':
        if saw_explicit_argument and not saw_recognized_value:
            return None
        return 'clear', project_id, None

    if saw_explicit_argument and not saw_recognized_value and project_id is None and hours is None:
        return None

    return 'prune', project_id, (hours if hours is not None else 24)


def _build_status_message(session: Session, *, user_id: int, assistant_name: str) -> str:
    wm = max(15, min(int(getattr(settings, "hybrid_predictor_window_minutes", 120) or 120), 24 * 60))
    since = datetime.utcnow() - timedelta(minutes=wm)
    rows = session.exec(
        select(SignalLedgerEvent).where(
            SignalLedgerEvent.created_by_user_id == int(user_id),
            SignalLedgerEvent.created_at >= since,
        )
    ).all()
    vis = calculate_live_viscosity(rows)
    node_count = session.exec(
        select(HiveDevice).where(HiveDevice.last_seen_at >= since)
    ).all()
    policy = get_policy(session, int(user_id))
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    permission_tier = str(actions_cfg.get("default_permission_tier", "execute")).strip().lower() or "execute"
    return (
        f"I’m {assistant_name}. η: {float(vis.score):.2f} ({str(vis.band).title()}). "
        f"Nodes: {len(node_count)}, "
        f"Tier: {permission_tier.title()}, "
        f"Mode: {str(vis.prediction_state).title()}, "
        f"battery={('n/a' if vis.battery_level is None else str(round(float(vis.battery_level), 1)) + '%')}, "
        f"cpu={('n/a' if vis.cpu_temp_c is None else str(round(float(vis.cpu_temp_c), 1)) + '°C')}, "
        f"interruptions={int(vis.recent_interruptions)}."
    )


def _build_daily_summary_message(session: Session, *, user_id: int, assistant_name: str) -> str:
    out = compute_daily_consolidation(session, user_id=int(user_id), project_id=None, window_hours=24)
    return f"I’m {assistant_name}. {str(out.get('summary_text') or '').strip()}"


def _persona_mode_from_policy(session: Session, *, user_id: int) -> str:
    policy = get_policy(session, int(user_id))
    assistant = policy.get("assistant") if isinstance(policy.get("assistant"), dict) else {}
    mode = str(assistant.get("persona_mode", "calm")).strip().lower()
    if mode not in {"coach", "calm", "briefing"}:
        mode = "calm"
    return mode


def _apply_persona_tone(text: str, *, mode: str) -> str:
    base = str(text or "").strip()
    m = str(mode or "calm").strip().lower()
    if m == "coach":
        return f"Coach mode: {base} Next step: execute one focused action now."
    if m == "briefing":
        return f"Briefing mode: {base} Decision: proceed if confidence and policy gates are green."
    return f"Calm mode: {base} Keep a steady, low-friction pace."


def _tone_consistency_score(*, mode: str, text: str) -> float:
    markers = _PERSONA_MARKERS.get(str(mode), _PERSONA_MARKERS["calm"])
    t = str(text or "").lower()
    if not markers:
        return 0.0
    hits = sum(1 for m in markers if m in t)
    return float(hits) / float(len(markers))


def _set_kill_switch(
    session: Session,
    *,
    user_id: int,
    enabled: bool,
    clear_pending: bool,
) -> tuple[bool, int]:
    policy = get_policy(session, int(user_id))
    actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
    updated_policy = {
        **policy,
        "actions": {
            **actions_cfg,
            "kill_switch": bool(enabled),
        },
    }
    set_policy(session, int(user_id), updated_policy)

    cleared = 0
    if bool(clear_pending):
        q = (
            select(HiveOutboxMessage)
            .where(HiveOutboxMessage.created_by_user_id == int(user_id))
            .where(HiveOutboxMessage.kind == "device_action")
            .where(HiveOutboxMessage.submitted_at.is_(None))
            .order_by(HiveOutboxMessage.created_at.asc())
            .limit(2000)
        )
        rows = session.exec(q).all()
        now = datetime.utcnow()
        for m in rows:
            p = _loads_dict(m.payload_json)
            p["ack"] = {
                "status": "rejected",
                "notes": "cleared_by_telegram_freeze",
                "acked_at": now.isoformat() + "Z",
            }
            m.payload_json = _dumps(p)
            m.submitted_at = now
            session.add(m)
            cleared += 1
        session.commit()
    return bool(enabled), int(cleared)


def _build_launch_now_message(
    session: Session,
    *,
    current_user: User,
    assistant_name: str,
) -> str:
    try:
        launch = auto_handoff_launch(
            AutoHandoffLaunchRequest(
                project_id=None,
                window_minutes=int(getattr(settings, "hybrid_predictor_window_minutes", 120) or 120),
                base_duration_minutes=45,
                current_device_id="phone",
                candidate_device_ids=["phone", "laptop", "desktop"],
                focus_app="mycelium",
            ),
            current_user=current_user,
            session=session,
        )
    except Exception as e:
        return f"I’m {assistant_name}. I could not initialize launch right now ({type(e).__name__})."
    if str(launch.launch_mode) == "recovery" or int(launch.suggested_duration_minutes or 0) <= 0:
        return f"I’m {assistant_name}. Launch deferred: all nodes are gated. Recovery mode is recommended right now."

    rid = int(launch.replica_id or 0)
    if rid <= 0:
        return f"I’m {assistant_name}. I analyzed your nodes but could not create a launch replica."

    if str(launch.launch_mode) == "approved" and int(launch.queued_device_action_id or 0) > 0:
        return (
            f"I’m {assistant_name}. Launch initialized on {launch.recommended_device_id or 'best node'} "
            f"for {int(launch.suggested_duration_minutes or 0)} minutes. "
            f"Queued action #{int(launch.queued_device_action_id or 0)}."
        )

    try:
        confirmed = auto_handoff_confirm(
            AutoHandoffConfirmRequest(replica_id=rid, device_id=launch.recommended_device_id),
            current_user=current_user,
            session=session,
        )
    except Exception as e:
        return (
            f"I’m {assistant_name}. I prepared launch proposal #{rid}, "
            f"but confirmation failed ({type(e).__name__})."
        )
    return (
        f"I’m {assistant_name}. Launch initialized on {launch.recommended_device_id or 'best node'} "
        f"for {int(launch.suggested_duration_minutes or 0)} minutes. "
        f"Queued action #{int(confirmed.queued_device_action_id)}."
    )


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

    try:
        record_stimulus_event(
            session,
            user_id=user_id,
            project_id=payload.project_id,
            device_id=str(settings.nexus_device_id or "local"),
            source="chat",
            modality="text",
            signal_type=f"chat_{channel}",
            stimulus={
                "conversation_key": conv_key,
                "channel": channel,
                "role": "user",
                "message_len": len(msg),
                "message_digest": hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16],
            },
            occurred_at=user_row.created_at,
        )
    except Exception:
        pass

    ap = get_assistant_profile_effective(session, user_id=user_id, project_id=payload.project_id)
    assistant_name = str(ap.get("given_name", "Synapse")).strip() or "Synapse"

    # Deterministic lightweight assistant response (tool-free skeleton).
    lower = msg.lower()
    if "daily summary" in lower or "consolidation" in lower or "end of day" in lower:
        assistant_text = _build_daily_summary_message(session, user_id=user_id, assistant_name=assistant_name)
    elif "launch now" in lower or "start session now" in lower or "initialize desktop" in lower:
        assistant_text = _build_launch_now_message(session, current_user=current_user, assistant_name=assistant_name)
    elif "how are you" in lower or "status" in lower or "viscosity" in lower:
        assistant_text = _build_status_message(session, user_id=user_id, assistant_name=assistant_name)
    else:
        assistant_text = (
            f"I’m {assistant_name}. I received your message. "
            "I can help with focus workflows, telemetry insights, and task replicas. "
            "If you want, ask me to bootstrap or verify your next work session."
        )

    persona_mode = _persona_mode_from_policy(session, user_id=user_id)
    assistant_text = _apply_persona_tone(assistant_text, mode=persona_mode)

    assistant_row = ConversationMessage(
        created_by_user_id=user_id,
        project_id=payload.project_id,
        conversation_key=conv_key,
        channel=channel,
        role="assistant",
        content=assistant_text,
        metadata_json=_dumps({"assistant_name": assistant_name, "persona_mode": persona_mode}),
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


@router.get("/tone/check", response_model=ChatToneCheckResponse)
def tone_check(
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
    mode = _persona_mode_from_policy(session, user_id=user_id)

    q = select(ConversationMessage).where(ConversationMessage.created_by_user_id == int(user_id))
    q = q.where(ConversationMessage.conversation_key == conv_key)
    q = q.where(ConversationMessage.role == "assistant")
    if project_id is None:
        q = q.where(ConversationMessage.project_id.is_(None))
    else:
        q = q.where(ConversationMessage.project_id == int(project_id))
    rows = session.exec(q.order_by(ConversationMessage.created_at.desc()).limit(lim)).all()

    scores = [_tone_consistency_score(mode=mode, text=str(r.content or "")) for r in rows]
    mean = float(sum(scores) / float(len(scores))) if scores else 0.0

    return ChatToneCheckResponse(
        ok=True,
        persona_mode=mode,
        n_messages=int(len(rows)),
        tone_consistency=float(round(mean, 6)),
        markers=list(_PERSONA_MARKERS.get(mode, [])),
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
    mapped_user = session.exec(select(User).where(User.id == int(user_id))).first()

    def _send_telegram_reply(
        *,
        assistant_text: str,
        command_name: str | None = None,
        project_id: int | None = None,
        result: object | None = None,
    ) -> dict[str, object]:
        persona_mode = _persona_mode_from_policy(session, user_id=int(user_id))
        assistant_text_local = _apply_persona_tone(assistant_text, mode=persona_mode)
        metadata: dict[str, object] = {
            "assistant_name": assistant_name,
            "source": "telegram_webhook",
            "persona_mode": persona_mode,
        }
        if command_name is not None:
            metadata["command"] = command_name
        if project_id is not None:
            metadata["project_id"] = project_id
        if result is not None:
            metadata["pruned_count"] = int(getattr(result, "pruned_count", 0) or 0)
            metadata["remaining_count"] = int(getattr(result, "remaining_count", 0) or 0)

        assistant_row = ConversationMessage(
            created_by_user_id=int(user_id),
            project_id=None,
            conversation_key=conv_key,
            channel="telegram",
            role="assistant",
            content=assistant_text_local,
            metadata_json=_dumps(metadata),
        )
        session.add(assistant_row)
        session.flush()

        delivered = False
        if bool(getattr(settings, "notifications_bridge_enabled", False)):
            delivered = _send_telegram(
                bot_token=str(getattr(settings, "notifications_telegram_bot_token", "") or ""),
                chat_id=chat_id,
                text=assistant_text_local,
            )

        now = datetime.utcnow()
        user_row.delivered_at = now
        assistant_row.delivered_at = now if delivered else None
        session.add(user_row)
        session.add(assistant_row)
        session.commit()

        return {"ok": True, "delivered": bool(delivered), "command": command_name or "message"}

    command = _parse_mission_log_command(text)
    if command and mapped_user is not None:
        kind, project_id, hours = command
        if kind == 'clear':
            try:
                result = prune_mission_log(
                    payload=MissionLogPruneRequest(project_id=project_id, older_than_hours=None, clear_all=True),
                    current_user=mapped_user,
                    session=session,
                )
            except HTTPException as exc:
                if int(getattr(exc, 'status_code', 500)) == 403:
                    assistant_text = f"I’m {assistant_name}. Access denied. Identity mismatch."
                else:
                    raise
            else:
                scope_text = f" for project {int(project_id)}" if project_id is not None else ""
                assistant_text = (
                    f"I’m {assistant_name}. Memory reset. HUD baseline established{scope_text}. "
                    f"Pruned {int(result.pruned_count)} traces."
                )
                return _send_telegram_reply(assistant_text=assistant_text, command_name=command[0], project_id=project_id, result=result)
        else:
            prune_hours = int(hours or 24)
            try:
                result = prune_mission_log(
                    payload=MissionLogPruneRequest(project_id=project_id, older_than_hours=prune_hours, clear_all=False),
                    current_user=mapped_user,
                    session=session,
                )
            except HTTPException as exc:
                if int(getattr(exc, 'status_code', 500)) == 403:
                    assistant_text = f"I’m {assistant_name}. Access denied. Identity mismatch."
                else:
                    raise
            else:
                scope_text = f" for project {int(project_id)}" if project_id is not None else ""
                assistant_text = (
                    f"I’m {assistant_name}. Stale traces removed. Metabolic health: 100%{scope_text}. "
                    f"Remaining traces: {int(result.remaining_count)}."
                )
                return _send_telegram_reply(assistant_text=assistant_text, command_name=command[0], project_id=project_id, result=result)

    lower = text.lower()
    if lower.startswith("/clear_logs") or lower.startswith("/prune"):
        assistant_text = (
            f"I’m {assistant_name}. I couldn’t parse that command. "
            "Use /clear_logs or /prune [hours], optionally /prune project <id> [hours]."
        )
        return _send_telegram_reply(assistant_text=assistant_text, command_name="prune" if lower.startswith("/prune") else "clear")

    if lower in {"/freeze", "/killswitch on", "/freeze now"}:
        if mapped_user is None:
            assistant_text = f"I’m {assistant_name}. I could not resolve your account for freeze command."
        else:
            enabled, cleared = _set_kill_switch(
                session,
                user_id=int(user_id),
                enabled=True,
                clear_pending=True,
            )
            assistant_text = (
                f"I’m {assistant_name}. Emergency brake engaged (kill-switch={str(enabled).lower()}). "
                f"Cleared {int(cleared)} pending device actions."
            )
    elif lower in {"/unfreeze", "/thaw", "/killswitch off"}:
        if mapped_user is None:
            assistant_text = f"I’m {assistant_name}. I could not resolve your account for unfreeze command."
        else:
            enabled, _ = _set_kill_switch(
                session,
                user_id=int(user_id),
                enabled=False,
                clear_pending=False,
            )
            assistant_text = f"I’m {assistant_name}. Emergency brake released (kill-switch={str(enabled).lower()})."
    elif lower in {"/freeze status", "/killswitch", "/safety"}:
        policy = get_policy(session, int(user_id))
        actions_cfg = policy.get("actions") if isinstance(policy.get("actions"), dict) else {}
        assistant_text = (
            f"I’m {assistant_name}. kill-switch={str(bool(actions_cfg.get('kill_switch', False))).lower()}, "
            f"autonomy_mode={str(actions_cfg.get('autonomy_mode', 'strict'))}, "
            f"require_confirm={str(bool(actions_cfg.get('require_confirm', True))).lower()}."
        )
    elif lower.startswith("/status") or "how are you" in lower or "status" in lower or "viscosity" in lower:
        assistant_text = _build_status_message(session, user_id=int(user_id), assistant_name=assistant_name)
    elif "bootstrap" in lower:
        assistant_text = (
            f"I’m {assistant_name}. I can bootstrap a focus session. "
            "Use /api/nexus/tasks/bootstrap/work-session in the app to start it with your preferred duration."
        )
    elif "launch now" in lower or "start session now" in lower or "initialize desktop" in lower:
        if mapped_user is None:
            assistant_text = f"I’m {assistant_name}. I could not resolve your account for launch."
        else:
            assistant_text = _build_launch_now_message(session, current_user=mapped_user, assistant_name=assistant_name)
    elif "daily summary" in lower or "consolidation" in lower or "end of day" in lower:
        assistant_text = _build_daily_summary_message(session, user_id=int(user_id), assistant_name=assistant_name)
    elif "how are you" in lower or "status" in lower or "viscosity" in lower:
        assistant_text = _build_status_message(session, user_id=int(user_id), assistant_name=assistant_name)
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

    persona_mode = _persona_mode_from_policy(session, user_id=int(user_id))
    assistant_text = _apply_persona_tone(assistant_text, mode=persona_mode)

    assistant_row = ConversationMessage(
        created_by_user_id=int(user_id),
        project_id=None,
        conversation_key=conv_key,
        channel="telegram",
        role="assistant",
        content=assistant_text,
        metadata_json=_dumps(
            {"assistant_name": assistant_name, "source": "telegram_webhook", "persona_mode": persona_mode}
        ),
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
