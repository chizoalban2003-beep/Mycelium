from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.models import ProjectMember, ProjectRole, User
from mycelium_app.security import decode_token
from mycelium_app.settings import settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def _extract_token_from_cookie(request: Request) -> Optional[str]:
    return request.cookies.get(settings.cookie_name)


def get_current_user(
    request: Request,
    session: Session = Depends(get_session),
    token: str | None = Depends(oauth2_scheme),
) -> User:
    # Prefer Authorization header if provided, else fallback to cookie.
    raw_token = token or _extract_token_from_cookie(request) or ""
    try:
        payload = decode_token(raw_token)
        subject = payload.get("sub")
        if not subject:
            raise ValueError("Missing subject")
        user_id = int(subject)
    except Exception:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = session.exec(select(User).where(User.id == user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_project_role(project_id: int, minimum: ProjectRole):
    role_order = {ProjectRole.viewer: 1, ProjectRole.editor: 2, ProjectRole.owner: 3}

    def _checker(
        current_user: User = Depends(get_current_user),
        session: Session = Depends(get_session),
    ) -> User:
        member = session.exec(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == current_user.id,
            )
        ).first()
        if not member:
            raise HTTPException(status_code=403, detail="Not a project member")
        if role_order[member.role] < role_order[minimum]:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return current_user

    return _checker
