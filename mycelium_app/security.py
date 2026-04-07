from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import jwt
from passlib.context import CryptContext

from mycelium_app.settings import settings


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(*, subject: str, expires_minutes: Optional[int] = None, extra: dict[str, Any] | None = None) -> str:
    expire_minutes = expires_minutes or settings.access_token_expire_minutes
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    to_encode: dict[str, Any] = {"sub": subject, "exp": expire}
    if extra:
        to_encode.update(extra)
    return jwt.encode(to_encode, settings.secret_key, algorithm="HS256")


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.secret_key, algorithms=["HS256"])


def create_password_reset_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    return token, hash_password_reset_token(token)


def hash_password_reset_token(token: str) -> str:
    key = str(getattr(settings, "secret_key", "") or "dev-secret-change-me").encode("utf-8")
    return hmac.new(key, token.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_password_reset_token(token: str, token_hash: str) -> bool:
    return hmac.compare_digest(hash_password_reset_token(token), str(token_hash or ""))
