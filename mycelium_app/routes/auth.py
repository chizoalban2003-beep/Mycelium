from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, delete, select

from mycelium_app.db import get_session
from mycelium_app.models import PasswordResetToken, User
from mycelium_app.parental_policy import get_policy
from mycelium_app.schemas import (
    Message,
    PasswordResetConfirm,
    PasswordResetConfirmResponse,
    PasswordResetRequest,
    PasswordResetRequestResponse,
    Token,
    UserCreate,
    UserPublic,
)
from mycelium_app.security import (
    create_access_token,
    create_password_reset_token,
    hash_password,
    hash_password_reset_token,
    verify_password,
    verify_password_reset_token,
)
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/auth", tags=["auth"])


def _send_telegram_message(*, bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    try:
        import json
        from urllib import parse, request as urlrequest

        token = str(bot_token or "").strip()
        chat = str(chat_id or "").strip()
        if not token or not chat:
            return False, "missing telegram configuration"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = parse.urlencode({"chat_id": chat, "text": text[:4096], "disable_web_page_preview": "true"}).encode(
            "utf-8"
        )
        req = urlrequest.Request(url=url, data=payload, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urlrequest.urlopen(req, timeout=10) as resp:  # nosec B310 - fixed Telegram API URL
            body = resp.read().decode("utf-8", errors="replace")
            if 200 <= int(getattr(resp, "status", 200)) < 300 and '"ok":true' in body:
                return True, ""
            return False, "telegram api non-ok"
    except Exception as e:
        return False, str(e)[:300]


def create_password_reset_request_link(
    session: Session,
    *,
    email: str,
    base_url: str,
) -> tuple[bool, str]:
    normalized_email = str(email or "").strip().lower()
    user = session.exec(select(User).where(User.email == normalized_email)).first()
    if not user:
        return False, "If the account exists, a recovery link was prepared."

    session.exec(delete(PasswordResetToken).where(PasswordResetToken.user_id == user.id, PasswordResetToken.used_at.is_(None)))
    session.commit()

    token, token_hash = create_password_reset_token()
    now = datetime.utcnow()
    row = PasswordResetToken(
        user_id=int(user.id),
        token_hash=token_hash,
        created_at=now,
        expires_at=now + timedelta(minutes=30),
        used_at=None,
    )
    session.add(row)
    session.commit()

    policy = get_policy(session, int(user.id))
    notif = policy.get("notifications") if isinstance(policy.get("notifications"), dict) else {}
    telegram_enabled = bool(notif.get("telegram_enabled", False))
    telegram_chat_id = str(notif.get("telegram_chat_id") or "").strip()
    if telegram_enabled and telegram_chat_id and bool(getattr(settings, "notifications_bridge_enabled", False)):
        bot_token = str(getattr(settings, "notifications_telegram_bot_token", "") or "").strip()
        reset_url = f"{str(base_url).rstrip('/')}/reset-password/{token}"
        _send_telegram_message(
            bot_token=bot_token,
            chat_id=telegram_chat_id,
            text=(
                "Mycelium recovery link\n"
                f"Open this once to set a new password:\n{reset_url}\n\n"
                "If you did not request this, ignore it."
            ),
        )
    return True, "If the account exists, a recovery link was prepared."


def consume_password_reset_token(session: Session, *, token: str, new_password: str) -> tuple[bool, str]:
    token_hash = hash_password_reset_token(str(token or ""))
    row = session.exec(select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)).first()
    if not row:
        return False, "Invalid or expired recovery link"
    now = datetime.utcnow()
    if row.used_at is not None or row.expires_at <= now:
        return False, "Invalid or expired recovery link"

    user = session.get(User, row.user_id)
    if not user:
        return False, "Invalid or expired recovery link"

    user.hashed_password = hash_password(str(new_password or ""))
    row.used_at = now
    session.add(user)
    session.add(row)
    session.commit()
    return True, "Password updated. You can sign in now."


@router.post("/register", response_model=UserPublic)
def register(payload: UserCreate, session: Session = Depends(get_session)):
    email = str(payload.email or "").strip().lower()
    full_name = str(payload.full_name or "").strip()
    password = str(payload.password or "")
    existing = session.exec(select(User).where(User.email == email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=email, full_name=full_name, hashed_password=hash_password(password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return UserPublic(id=user.id, email=user.email, full_name=user.full_name, created_at=user.created_at)


@router.post("/login", response_model=Token)
def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    username = str(form_data.username or "").strip().lower()
    user = session.exec(select(User).where(User.email == username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    token = create_access_token(subject=str(user.id))
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=settings.access_token_expire_minutes * 60,
    )
    return Token(access_token=token)


@router.post("/logout", response_model=Message)
def logout(response: Response):
    response.delete_cookie(settings.cookie_name)
    return Message(message="Logged out")


@router.post("/password-reset/request", response_model=PasswordResetRequestResponse)
def request_password_reset(
    payload: PasswordResetRequest,
    request: Request,
    session: Session = Depends(get_session),
):
    base_url = str(getattr(settings, "app_public_base_url", "") or str(request.base_url)).rstrip("/")
    _, message = create_password_reset_request_link(session, email=payload.email, base_url=base_url)
    return PasswordResetRequestResponse(ok=True, message=message)


@router.post("/password-reset/confirm", response_model=PasswordResetConfirmResponse)
def confirm_password_reset(payload: PasswordResetConfirm, token: str, session: Session = Depends(get_session)):
    ok, message = consume_password_reset_token(session, token=token, new_password=payload.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return PasswordResetConfirmResponse(ok=True, message=message)
