from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.models import User
from mycelium_app.schemas import Message, Token, UserCreate, UserPublic
from mycelium_app.security import create_access_token, hash_password, verify_password
from mycelium_app.settings import settings


router = APIRouter(prefix="/api/auth", tags=["auth"])


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
