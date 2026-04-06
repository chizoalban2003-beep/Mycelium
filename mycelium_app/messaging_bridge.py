from __future__ import annotations

import json
from datetime import datetime, timedelta
from urllib import parse, request

from sqlmodel import Session, select

from mycelium_app.models import NexusNudge
from mycelium_app.parental_policy import get_policy
from mycelium_app.settings import settings


def _loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_nudge_url(nudge_id: int) -> str | None:
    base = str(getattr(settings, "app_public_base_url", "") or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/projects?nudge_id={int(nudge_id)}"


def _send_telegram(*, bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    token = (bot_token or "").strip()
    chat = (chat_id or "").strip()
    if not token or not chat:
        return False, "telegram credentials missing"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = parse.urlencode(
        {
            "chat_id": chat,
            "text": text[:4096],
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    req = request.Request(url=url, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with request.urlopen(req, timeout=10) as resp:  # nosec B310 - Telegram API URL is fixed
            body = resp.read().decode("utf-8", errors="replace")
            if 200 <= int(getattr(resp, "status", 200)) < 300 and '"ok":true' in body:
                return True, ""
            return False, "telegram api non-ok"
    except Exception as e:
        return False, str(e)[:300]


def dispatch_pending_nudges(session: Session, *, max_items: int | None = None) -> int:
    """Dispatch unseen nudges to external channels based on user policy.

    Current channel: Telegram.
    """

    if not bool(getattr(settings, "notifications_bridge_enabled", False)):
        return 0

    bot_token = str(getattr(settings, "notifications_telegram_bot_token", "") or "").strip()
    if not bot_token:
        return 0

    lookback_h = max(1, min(int(getattr(settings, "notifications_dispatch_lookback_hours", 24) or 24), 7 * 24))
    since = datetime.utcnow() - timedelta(hours=lookback_h)
    limit = max(1, min(int(max_items or getattr(settings, "notifications_dispatch_max_per_tick", 50) or 50), 500))

    q = (
        select(NexusNudge)
        .where(NexusNudge.created_at >= since)
        .where(NexusNudge.seen_at.is_(None))
        .order_by(NexusNudge.created_at.asc())
        .limit(limit)
    )
    rows = session.exec(q).all()

    sent = 0
    for n in rows:
        payload = _loads_dict(n.payload_json)
        bridge = payload.get("_bridge") if isinstance(payload.get("_bridge"), dict) else {}
        t_state = bridge.get("telegram") if isinstance(bridge.get("telegram"), dict) else {}
        if str(t_state.get("dispatched_at") or "").strip():
            continue

        policy = get_policy(session, int(n.created_by_user_id or 0))
        notif = policy.get("notifications") if isinstance(policy.get("notifications"), dict) else {}
        if not bool(notif.get("enabled", False)):
            continue
        if not bool(notif.get("telegram_enabled", False)):
            continue

        chat_id = str(notif.get("telegram_chat_id") or "").strip()[:64]
        if not chat_id:
            continue

        kinds = notif.get("telegram_nudge_kinds") if isinstance(notif.get("telegram_nudge_kinds"), list) else []
        allowed_kinds = {str(k).strip().lower() for k in kinds if str(k).strip()}
        nudge_kind = str(n.kind or "").strip().lower()
        if allowed_kinds and nudge_kind not in allowed_kinds:
            continue

        url = _build_nudge_url(int(n.id or 0))
        msg = f"SynapseHive • {str(n.title or 'Nudge').strip()}\n{str(n.message or '').strip()}"
        if url:
            msg = f"{msg}\n\nOpen: {url}"

        ok, err = _send_telegram(bot_token=bot_token, chat_id=chat_id, text=msg)

        bridge = dict(bridge)
        t_state = dict(t_state)
        t_state["attempts"] = int(t_state.get("attempts", 0) or 0) + 1
        t_state["last_attempt_at"] = datetime.utcnow().isoformat() + "Z"
        if ok:
            t_state["dispatched_at"] = datetime.utcnow().isoformat() + "Z"
            t_state["last_error"] = ""
            sent += 1
        else:
            t_state["last_error"] = str(err or "dispatch_failed")[:300]

        bridge["telegram"] = t_state
        payload["_bridge"] = bridge
        n.payload_json = _dumps(payload)
        session.add(n)

    if rows:
        session.commit()
    return sent
