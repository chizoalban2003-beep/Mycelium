from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.models import Project, ProjectMember, ProjectRole, User
from mycelium_app.schemas import MemberAdd, Message, ProjectCreate, ProjectInviteRequest, ProjectInviteResponse, ProjectPublic
from mycelium_app.security import hash_password
from mycelium_app.stimulus import record_stimulus_event


router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectPublic])
def list_projects(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    memberships = session.exec(select(ProjectMember).where(ProjectMember.user_id == current_user.id)).all()
    if not memberships:
        return []
    project_ids = [m.project_id for m in memberships]
    projects = session.exec(select(Project).where(Project.id.in_(project_ids))).all()
    return [
        ProjectPublic(
            id=p.id,
            name=p.name,
            description=p.description,
            created_at=p.created_at,
            created_by_user_id=p.created_by_user_id,
        )
        for p in projects
    ]


@router.post("", response_model=ProjectPublic)
def create_project(payload: ProjectCreate, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    project = Project(name=payload.name, description=payload.description, created_by_user_id=current_user.id)
    session.add(project)
    session.commit()
    session.refresh(project)

    session.add(ProjectMember(project_id=project.id, user_id=current_user.id, role=ProjectRole.owner))
    session.commit()

    try:
        record_stimulus_event(
            session,
            user_id=int(current_user.id or 0),
            project_id=int(project.id or 0),
            device_id="local",
            source="project_api",
            modality="workspace",
            signal_type="project_create",
            stimulus={
                "project_name_len": len(str(payload.name or "")),
                "has_description": bool(str(payload.description or "").strip()),
                "role": "owner",
            },
            occurred_at=project.created_at,
        )
    except Exception:
        pass

    return ProjectPublic(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        created_by_user_id=project.created_by_user_id,
    )


@router.get("/{project_id}", response_model=ProjectPublic)
def get_project(project_id: int, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == current_user.id)
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a project member")
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectPublic(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        created_by_user_id=project.created_by_user_id,
    )


@router.post("/{project_id}/members", response_model=Message)
def add_member(
    project_id: int,
    payload: MemberAdd,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # MVP: only owners can add members; users must already exist.
    owner = session.exec(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.id,
            ProjectMember.role == ProjectRole.owner,
        )
    ).first()
    if not owner:
        raise HTTPException(status_code=403, detail="Owner role required")
    user = session.exec(select(User).where(User.email == payload.email)).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found (register first)")
    existing = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user.id)
    ).first()
    if existing:
        existing.role = payload.role
        session.add(existing)
    else:
        session.add(ProjectMember(project_id=project_id, user_id=user.id, role=payload.role))
    session.commit()

    try:
        record_stimulus_event(
            session,
            user_id=int(current_user.id or 0),
            project_id=project_id,
            device_id="local",
            source="project_api",
            modality="workspace",
            signal_type="member_add",
            stimulus={"role": str(payload.role), "has_existing_member": bool(existing), "user_found": True},
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return Message(message="Member added")


@router.post("/{project_id}/invite", response_model=ProjectInviteResponse)
def invite_member(
    project_id: int,
    payload: ProjectInviteRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # Owner-only onboarding path: create user (or update password optionally), then add membership.
    owner = session.exec(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.id,
            ProjectMember.role == ProjectRole.owner,
        )
    ).first()
    if not owner:
        raise HTTPException(status_code=403, detail="Owner role required")

    created_user = False
    updated_password = False
    email = str(payload.email).strip().lower()

    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(
            email=email,
            full_name=str(payload.full_name or "").strip(),
            hashed_password=hash_password(payload.password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        created_user = True
    else:
        if bool(payload.reset_password_if_exists):
            user.hashed_password = hash_password(payload.password)
            if str(payload.full_name or "").strip():
                user.full_name = str(payload.full_name).strip()
            session.add(user)
            session.commit()
            updated_password = True

    existing = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id, ProjectMember.user_id == user.id)
    ).first()
    added_member = False
    if existing:
        existing.role = payload.role
        session.add(existing)
    else:
        session.add(ProjectMember(project_id=project_id, user_id=user.id, role=payload.role))
        added_member = True
    session.commit()

    try:
        record_stimulus_event(
            session,
            user_id=int(current_user.id or 0),
            project_id=project_id,
            device_id="local",
            source="project_api",
            modality="workspace",
            signal_type="member_invite",
            stimulus={
                "role": str(payload.role),
                "created_user": bool(created_user),
                "updated_password": bool(updated_password),
                "added_member": bool(added_member),
            },
            occurred_at=datetime.utcnow(),
        )
    except Exception:
        pass

    return ProjectInviteResponse(
        ok=True,
        message="Invite processed",
        created_user=bool(created_user),
        updated_password=bool(updated_password),
        added_member=bool(added_member),
    )
