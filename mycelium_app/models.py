from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ProjectRole(str, Enum):
    owner = "owner"
    editor = "editor"
    viewer = "viewer"


class NodeType(str, Enum):
    etl = "etl"
    eda = "eda"
    stat_test = "stat_test"
    feature_engineering = "feature_engineering"
    ml_model = "ml_model"
    dashboard = "dashboard"
    prediction_service = "prediction_service"


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: str = ""
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str = ""
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectMember(SQLModel, table=True):
    project_id: int = Field(foreign_key="project.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    role: ProjectRole = Field(default=ProjectRole.viewer)
    added_at: datetime = Field(default_factory=datetime.utcnow)


class TreeNode(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    parent_id: Optional[int] = Field(default=None, foreign_key="treenode.id", index=True)
    name: str
    node_type: NodeType = Field(index=True)
    config_json: str = "{}"  # persisted JSON blob (validated later per node_type)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NodeRunStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class NodeRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: int = Field(foreign_key="treenode.id", index=True)
    status: NodeRunStatus = Field(default=NodeRunStatus.queued, index=True)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    logs: str = ""
