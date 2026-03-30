from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr

from mycelium_app.models import NodeRunStatus, NodeType, ProjectRole


class Message(BaseModel):
    message: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str = ""


class UserPublic(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectPublic(BaseModel):
    id: int
    name: str
    description: str
    created_at: datetime
    created_by_user_id: int


class MemberAdd(BaseModel):
    email: EmailStr
    role: ProjectRole = ProjectRole.viewer


class TreeNodeCreate(BaseModel):
    parent_id: Optional[int] = None
    name: str
    node_type: NodeType
    config_json: str = "{}"


class TreeNodePublic(BaseModel):
    id: int
    project_id: int
    parent_id: Optional[int]
    name: str
    node_type: NodeType
    config_json: str
    created_by_user_id: int
    created_at: datetime


class NodeRunPublic(BaseModel):
    id: int
    node_id: int
    status: NodeRunStatus
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    logs: str
