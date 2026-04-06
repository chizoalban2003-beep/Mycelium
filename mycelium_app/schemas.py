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


class NexusPrivacyExportStatus(BaseModel):
    hive_enabled: bool
    export_enabled: bool


class NexusPrivacyExportUpdateRequest(BaseModel):
    export_enabled: bool


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


class HiveOutboxMessagePublic(BaseModel):
    created_at: datetime
    device_id: str
    project_id: int | None
    kind: str
    payload: dict[str, object]


class HiveOutboxMessageStoreResponse(BaseModel):
    ok: bool = True
    message_id: int


class HiveOutboxMessageListResponse(BaseModel):
    messages: list[HiveOutboxMessagePublic]


class HiveWhisperImportRequest(BaseModel):
    update_uuid: str | None = None
    source: str = "hive_empathy"
    version: str = "whisper_v1"
    whisper: dict[str, object]


class HiveWhisperImportResponse(BaseModel):
    ok: bool = True
    update_uuid: str
    imported: bool = True


class HiveCuriosityFeedbackImportRequest(BaseModel):
    update_uuid: str | None = None
    source: str = "active_curiosity"
    version: str = "curiosity_v1"
    feedback: dict[str, object]


class HiveCuriosityFeedbackImportResponse(BaseModel):
    ok: bool = True
    update_uuid: str
    imported: bool = True


class HiveCuriosityConceptImportRequest(BaseModel):
    update_uuid: str | None = None
    source: str = "user_feedback_ionizer"
    version: str = "concept_v1"
    concept: dict[str, object]


class HiveCuriosityConceptImportResponse(BaseModel):
    ok: bool = True
    update_uuid: str
    imported: bool = True


class HiveWisdomLatestResponse(BaseModel):
    ok: bool = True
    as_of: datetime | None = None
    project_id: int | None = None
    n_updates_considered: int = 0
    n_whispers_used: int = 0
    recommended_kwargs: dict[str, object] = Field(default_factory=dict)
    evidence: dict[str, object] = Field(default_factory=dict)


class HiveHealthPoint(BaseModel):
    date: str  # YYYY-MM-DD
    n_global_updates: int = 0
    n_wisdom_whispers: int = 0
    n_curiosity_concepts: int = 0


class HiveMetricTrendPoint(BaseModel):
    date: str  # YYYY-MM-DD
    n: int = 0
    avg: float = 0.0


class HiveMetricTrend(BaseModel):
    metric_name: str
    points: list[HiveMetricTrendPoint] = Field(default_factory=list)


class HiveHealthResponse(BaseModel):
    ok: bool = True
    as_of: datetime
    window_days: int = 30
    totals: dict[str, int] = Field(default_factory=dict)
    messages_by_kind: dict[str, int] = Field(default_factory=dict)
    growth_curve: list[HiveHealthPoint] = Field(default_factory=list)
    metric_trends: list[HiveMetricTrend] = Field(default_factory=list)


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


class SelfReflectionResponse(BaseModel):
    ok: bool = True
    mood: str
    mood_signal: dict[str, float]
    identity_hash: str
    top_preferences: list[dict[str, object]]
    causal_hints: list[str]
    stats: dict[str, object]


class HomeostasisTickResponse(BaseModel):
    ok: bool = True
    mood: str
    identity_hash: str
    actions: list[str]


class HomeostasisStatusResponse(BaseModel):
    ok: bool = True
    state: dict[str, object] | None = None


class IdentityPresentationResponse(BaseModel):
    ok: bool = True
    identity_hash: str
    mood: str
    display_name: str
    tagline: str
    palette: dict[str, str] = Field(default_factory=dict)


class NexusNudgePublic(BaseModel):
    id: int
    created_at: datetime
    project_id: int | None
    kind: str
    title: str
    message: str
    payload: dict[str, object] = Field(default_factory=dict)
    seen_at: datetime | None = None


class NexusNudgeListResponse(BaseModel):
    nudges: list[NexusNudgePublic]


class NexusNudgeAckRequest(BaseModel):
    nudge_id: int


class NexusNudgeAckResponse(BaseModel):
    ok: bool = True


class CuriosityCasePublic(BaseModel):
    id: int
    created_at: datetime
    project_id: int | None = None
    status: str
    dataset_digest: str
    target_col: str
    target_kind: str
    row_index: int | None = None
    error_kind: str
    error_value: float
    predicted: object | None = None
    actual: object | None = None
    excerpt: dict[str, object] = Field(default_factory=dict)
    question: str
    answered_at: datetime | None = None
    dismissed_at: datetime | None = None


class CuriosityCaseListResponse(BaseModel):
    cases: list[CuriosityCasePublic]


class CuriosityAnswerRequest(BaseModel):
    project_id: int | None = None
    case_id: int
    answer_text: str
    corrected_target: object | None = None
    tags: list[str] = Field(default_factory=list)
    export_to_hive: bool = True


class CuriosityAnswerResponse(BaseModel):
    ok: bool = True
    answer_id: int


class CuriosityDismissRequest(BaseModel):
    case_id: int


class CuriosityDismissResponse(BaseModel):
    ok: bool = True


class CuriosityExportSummaryResponse(BaseModel):
    ok: bool = True
    summary: dict[str, object] = Field(default_factory=dict)


class NexusFeedbackIonizeRequest(BaseModel):
    project_id: int | None = None
    nudge_id: int | None = None
    hint_tag: str
    action: str = "confirm"  # confirm|correct
    concept_text: str
    export_to_hive: bool = False


class NexusFeedbackIonizeResponse(BaseModel):
    ok: bool = True
    entry_uuid: str
    entry_id: int
    digest: str
    exported_to_hive: bool = False
    export_redacted: bool = False
    export_reason: str | None = None


class NexusKnowledgeAuditResponse(BaseModel):
    ok: bool = True
    as_of: datetime | None = None
    project_id: int | None = None
    local: dict[str, object] = Field(default_factory=dict)
    hive: dict[str, object] = Field(default_factory=dict)
    validation: dict[str, object] = Field(default_factory=dict)
