from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

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


class NexusIngestTextRequest(BaseModel):
    text: str
    project_id: int | None = None
    modality: str = "auto"  # auto, finance, style, grammar
    source: str = "text"
    tags: list[str] = Field(default_factory=list)
    physics_used: dict[str, object] = Field(default_factory=dict)
    feedback: str | None = None


class NexusEntryPublic(BaseModel):
    entry_uuid: str
    created_at: datetime
    project_id: int | None
    device_id: str
    source: str
    modality: str
    raw_text: str
    extracted: dict[str, object]
    physics_used: dict[str, object]
    confidence: float | None
    feedback: str
    tags: list[str]


class NexusIngestTextResponse(BaseModel):
    ok: bool = True
    entry: NexusEntryPublic


class NexusListResponse(BaseModel):
    entries: list[NexusEntryPublic]


class NexusExportResponse(BaseModel):
    device_id: str
    exported_at: datetime
    entries: list[NexusEntryPublic]


class NexusImportRequest(BaseModel):
    entries: list[NexusEntryPublic]


class NexusImportResponse(BaseModel):
    ok: bool = True
    imported: int
    skipped: int


class NexusPolicyPublic(BaseModel):
    policy: dict[str, object]


class NexusPolicyUpdateRequest(BaseModel):
    policy: dict[str, object]


class NexusIntroResponse(BaseModel):
    mode: str
    observe_hours: int
    message: str


class HiveReportBuildRequest(BaseModel):
    project_id: int | None = None
    since: datetime | None = None
    limit: int = 500


class HiveReportPublic(BaseModel):
    created_at: datetime
    device_id: str
    project_id: int | None
    report: dict[str, object]


class HiveReportBuildResponse(BaseModel):
    ok: bool = True
    report: HiveReportPublic


class HiveOutboxStoreResponse(BaseModel):
    ok: bool = True
    outbox_id: int


class HiveOutboxListResponse(BaseModel):
    reports: list[HiveReportPublic]


class HiveGlobalUpdatePublic(BaseModel):
    update_uuid: str
    created_at: datetime
    source: str
    version: str
    update: dict[str, object]


class HiveGlobalUpdateImportRequest(BaseModel):
    update_uuid: str | None = None
    source: str = "manual_import"
    version: str = ""
    update: dict[str, object]


class HiveGlobalUpdateImportResponse(BaseModel):
    ok: bool = True
    update_uuid: str


class HiveGlobalUpdateListResponse(BaseModel):
    updates: list[HiveGlobalUpdatePublic]


class TelemetryIngestRequest(BaseModel):
    project_id: int | None = None
    device_id: str | None = None
    signal_type: str
    payload: dict[str, object] = Field(default_factory=dict)
    occurred_at: datetime | None = None


class TelemetryIngestResponse(BaseModel):
    ok: bool = True
    event_id: int


class TelemetrySummaryResponse(BaseModel):
    ok: bool = True
    window_hours: int
    n_events: int
    signal_counts: dict[str, int]
    confidence: float
    patterns: list[dict[str, object]]
    first_word: str | None = None


class TelemetryDeepFreezeSweepRequest(BaseModel):
    project_id: int | None = None
    device_id: str | None = None
    window_hours: int = 24
    accept_r2_threshold: float = 0.90
    min_pairs: int = 30


class TelemetryDeepFreezeSweepResponse(BaseModel):
    ok: bool = True
    entry_id: int
    domain: str
    metric: str
    r2: float
    accuracy: float
    n_pairs: int
    accepted: bool


class GrowthSweepRecordRequest(BaseModel):
    project_id: int | None = None
    device_id: str | None = None
    domain: str
    metric: str
    score: float
    accepted: bool = False
    proposal: dict[str, object] = Field(default_factory=dict)
    outcome: dict[str, object] = Field(default_factory=dict)
    notes: str = ""


class GrowthSweepPublic(BaseModel):
    created_at: datetime
    device_id: str
    project_id: int | None
    domain: str
    metric: str
    score: float
    accepted: bool
    notes: str
    proposal: dict[str, object]
    outcome: dict[str, object]


class GrowthSweepRecordResponse(BaseModel):
    ok: bool = True
    entry_id: int


class GrowthStatusResponse(BaseModel):
    ok: bool = True
    stage: str
    unlocked_features: list[str]
    stats: dict[str, object]
    motto: str


class GrowthRecentResponse(BaseModel):
    entries: list[GrowthSweepPublic]
