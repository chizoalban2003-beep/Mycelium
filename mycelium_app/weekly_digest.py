"""Weekly ecosystem digest — generates and optionally emails a summary.

Every week (or on demand), compiles the force field state, behavioral
patterns, and growth stage into a human-readable digest. Can deliver
via email (SMTP bridge) or as a nudge.
"""

from __future__ import annotations

import json
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from sqlmodel import Session, select

from mycelium_app.models import NexusNudge, SignalLedgerEvent, User
from mycelium_app.settings import settings


def _build_email_html(digest: dict[str, str], agent_name: str, stage: str) -> str:
    """Build an HTML email from digest content."""
    headline = digest.get("headline", "Weekly Update")
    body = digest.get("body", "")
    highlights = digest.get("highlights", "")

    return f"""
    <div style="font-family:-apple-system,sans-serif;max-width:560px;margin:0 auto;padding:24px;background:#020617;color:#e2e8f0;border-radius:12px">
      <div style="text-align:center;margin-bottom:20px">
        <div style="font-size:28px">🌱</div>
        <div style="font-size:20px;font-weight:bold;color:#22d3ee">{agent_name}</div>
        <div style="font-size:11px;color:#64748b;letter-spacing:0.15em;text-transform:uppercase">GROW WITH DATA</div>
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:16px;margin-bottom:16px">
        <div style="font-size:16px;font-weight:bold;color:#e2e8f0">{headline}</div>
        <div style="font-size:13px;color:#94a3b8;margin-top:8px;line-height:1.6">{body}</div>
      </div>
      {f'<div style="background:#022c22;border:1px solid #064e3b;border-radius:8px;padding:12px;margin-bottom:16px;font-size:13px;color:#6ee7b7">{highlights}</div>' if highlights else ''}
      <div style="text-align:center;font-size:11px;color:#475569;margin-top:16px">
        Stage: {stage} · Sent by {agent_name} · myco.local
      </div>
    </div>
    """


def send_weekly_digest_email(
    session: Session,
    *,
    user_id: int,
    digest: dict[str, str],
    agent_name: str = "Myco",
    stage: str = "infant",
) -> bool:
    """Send the weekly digest via SMTP. Returns True if sent."""
    if not bool(settings.mail_enabled):
        return False

    user = session.get(User, int(user_id))
    if not user or not user.email:
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{agent_name} — {digest.get('headline', 'Weekly Digest')}"
        msg["From"] = str(settings.mail_from_address)
        msg["To"] = str(user.email)

        text_body = f"{digest.get('headline', '')}\n\n{digest.get('body', '')}\n\n{digest.get('highlights', '')}"
        html_body = _build_email_html(digest, agent_name, stage)

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        host = str(settings.mail_smtp_host).strip()
        port = int(settings.mail_smtp_port)
        if not host:
            return False

        if settings.mail_smtp_use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=int(settings.mail_smtp_timeout_seconds))
        else:
            server = smtplib.SMTP(host, port, timeout=int(settings.mail_smtp_timeout_seconds))
            if settings.mail_smtp_use_tls:
                server.starttls()

        username = str(settings.mail_smtp_username).strip()
        password = str(settings.mail_smtp_password).strip()
        if username and password:
            server.login(username, password)

        server.sendmail(str(settings.mail_from_address), [str(user.email)], msg.as_string())
        server.quit()
        return True

    except Exception:
        return False


def generate_and_deliver_digest(
    session: Session,
    *,
    user_id: int,
) -> dict[str, Any]:
    """Generate weekly digest and deliver via nudge + optional email."""
    from mycelium_app.force_field import compute_force_field
    from mycelium_app.unified_field import generate_weekly_digest
    from mycelium_app.pattern_engine import analyze_patterns
    from mycelium_app.assistant_profile import get_assistant_profile_effective
    from mycelium_app.growth import compute_growth_stage

    since = datetime.utcnow() - timedelta(hours=168)
    rows = session.exec(
        select(SignalLedgerEvent)
        .where(SignalLedgerEvent.created_by_user_id == int(user_id), SignalLedgerEvent.created_at >= since)
        .order_by(SignalLedgerEvent.created_at)
    ).all()

    signals = []
    for r in rows:
        try:
            payload = json.loads(r.payload_json or "{}")
        except Exception:
            payload = {}
        surface = payload.get("surface") or payload.get("stimulus") or payload
        signals.append({
            "signal_type": str(r.signal_type or ""),
            "app_name": str(surface.get("app_name", r.signal_type or "")),
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "payload": surface,
        })

    ff = compute_force_field(signals, window_hours=168, n_iterations=20)
    patterns = analyze_patterns(session, user_id=user_id, window_hours=168)
    profile = get_assistant_profile_effective(session, user_id=user_id, project_id=None)
    stage, _, _ = compute_growth_stage(session, user_id=user_id)

    agent_name = str(profile.get("given_name", "Myco"))
    digest = generate_weekly_digest(ff, patterns, agent_name=agent_name)

    # Create nudge
    nudge = NexusNudge(
        created_by_user_id=int(user_id),
        project_id=None,
        kind="weekly_digest",
        title=str(digest.get("headline", "Weekly Digest")),
        message=str(digest.get("body", "")),
        payload_json=json.dumps({
            "digest": digest,
            "stage": stage,
            "n_signals": len(signals),
        }, separators=(",", ":")),
    )
    session.add(nudge)
    session.commit()

    # Try email
    email_sent = send_weekly_digest_email(
        session, user_id=user_id, digest=digest,
        agent_name=agent_name, stage=stage,
    )

    return {
        "ok": True,
        "digest": digest,
        "email_sent": email_sent,
        "nudge_created": True,
        "n_signals": len(signals),
        "stage": stage,
    }
