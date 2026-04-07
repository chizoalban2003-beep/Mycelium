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


class ProjectInviteRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str = ""
    role: ProjectRole = ProjectRole.viewer
    reset_password_if_exists: bool = False


class ProjectInviteResponse(BaseModel):
    ok: bool = True
    message: str
    created_user: bool = False
    updated_password: bool = False
    added_member: bool = False


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


class HiveHealthSmoothedPoint(BaseModel):
    date: str  # YYYY-MM-DD
    global_updates_ma: float = 0.0
    wisdom_whispers_ma: float = 0.0
    curiosity_concepts_ma: float = 0.0


class HiveMetricTrendPoint(BaseModel):
    date: str  # YYYY-MM-DD
    n: int = 0
    avg: float = 0.0


class HiveMetricTrend(BaseModel):
    metric_name: str
    points: list[HiveMetricTrendPoint] = Field(default_factory=list)


class HiveRegressionAlert(BaseModel):
    metric_name: str
    date: str  # last day evaluated
    direction: str  # higher_better|lower_better
    baseline_days: int
    baseline_n: int
    baseline_avg: float
    last_n: int
    last_avg: float
    delta: float  # last_avg - baseline_avg
    delta_pct: float  # (last-baseline)/abs(baseline)
    severity: str = "warn"  # warn|critical


class HiveBroadcastImpactEvent(BaseModel):
    broadcast_date: str
    metric_name: str
    pre_days: int
    post_days: int
    pre_n: int
    post_n: int
    pre_avg: float
    post_avg: float
    delta: float
    delta_pct: float


class HiveHealthResponse(BaseModel):
    ok: bool = True
    as_of: datetime
    window_days: int = 30
    totals: dict[str, int] = Field(default_factory=dict)
    messages_by_kind: dict[str, int] = Field(default_factory=dict)
    growth_curve: list[HiveHealthPoint] = Field(default_factory=list)
    growth_curve_smoothed: list[HiveHealthSmoothedPoint] = Field(default_factory=list)
    metric_trends: list[HiveMetricTrend] = Field(default_factory=list)
    regression_alerts: list[HiveRegressionAlert] = Field(default_factory=list)
    broadcast_impact: list[HiveBroadcastImpactEvent] = Field(default_factory=list)


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


class TelemetryAssistantTickResponse(BaseModel):
    ok: bool = True
    created: bool = False


class TelemetryAssistantActionRequest(BaseModel):
    nudge_id: int
    action_id: str
    decision: str = "approve"  # approve|reject


class TelemetryAssistantActionResponse(BaseModel):
    ok: bool = True
    executed: bool = False
    action_id: str
    decision: str
    detail: str = ""
    sweep_entry_id: int | None = None
    queued_device_action_id: int | None = None


class TelemetryDeviceActionPublic(BaseModel):
    message_id: int
    created_at: datetime
    device_id: str
    project_id: int | None
    action_id: str
    confidence: float
    command: dict[str, object] = Field(default_factory=dict)


class TelemetryDeviceActionPendingResponse(BaseModel):
    ok: bool = True
    actions: list[TelemetryDeviceActionPublic]


class TelemetryDeviceActionAckRequest(BaseModel):
    device_id: str | None = None
    status: str = "executed"  # executed|failed|rejected
    notes: str = ""


class TelemetryDeviceActionAckResponse(BaseModel):
    ok: bool = True
    message_id: int
    status: str
    executed: bool = False


class TaskTrajectoryRecordRequest(BaseModel):
    project_id: int | None = None
    device_id: str | None = None
    sequence: list[str] = Field(default_factory=list)
    app_state: dict[str, object] = Field(default_factory=dict)
    input_vector: dict[str, object] = Field(default_factory=dict)
    trajectory_key: str | None = None
    confidence: float = 0.5
    support_count: int = 1


class TaskTrajectoryRecordResponse(BaseModel):
    ok: bool = True
    trajectory_id: int
    trajectory_key: str


class TaskReplicaProposeRequest(BaseModel):
    project_id: int | None = None
    device_id: str | None = None
    title: str
    trajectory_key: str
    capability: str
    command: dict[str, object] = Field(default_factory=dict)
    consensus_fraction: float = 0.6
    species_confidence: float = 0.8
    notes: str = ""


class TaskReplicaPublic(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    project_id: int | None
    device_id: str
    title: str
    trajectory_key: str
    consensus_fraction: float
    species_confidence: float
    capability: str
    status: str
    command: dict[str, object] = Field(default_factory=dict)
    notes: str = ""


class TaskReplicaProposeResponse(BaseModel):
    ok: bool = True
    replica: TaskReplicaPublic


class TaskReplicaListResponse(BaseModel):
    replicas: list[TaskReplicaPublic]


class TaskReplicaDecisionRequest(BaseModel):
    device_id: str | None = None
    decision: str = "approve"  # approve|reject


class TaskReplicaDecisionResponse(BaseModel):
    ok: bool = True
    replica_id: int
    decision: str
    queued_device_action_id: int | None = None
    detail: str = ""


class TaskReplicaAckRequest(BaseModel):
    status: str = "executed"  # executed|failed
    notes: str = ""


class TaskReplicaAckResponse(BaseModel):
    ok: bool = True
    replica_id: int
    status: str


class TaskReplicaVerifyRequest(BaseModel):
    planned_minutes: int = 45
    focused_minutes: int = 0
    completed: bool = False
    closed_early: bool = False
    interruption_count: int = 0
    feedback_labels: list[str] = Field(default_factory=list)
    notes: str = ""


class TaskReplicaVerifyResponse(BaseModel):
    ok: bool = True
    replica_id: int
    adherence: float
    accepted: bool
    reward_delta: float
    updated_species_confidence: float
    feedback_labels: list[str] = Field(default_factory=list)
    growth_entry_id: int | None = None


class TaskReplicaFeedbackSummaryResponse(BaseModel):
    ok: bool = True
    window_hours: int = 168
    total_verified: int = 0
    label_counts: dict[str, int] = Field(default_factory=dict)
    label_acceptance: dict[str, float] = Field(default_factory=dict)


class TaskReplicaExplainResponse(BaseModel):
    ok: bool = True
    replica_id: int
    status: str
    capability: str
    species_confidence: float
    policy_min_confidence: float
    permission_tier: str
    kill_switch: bool
    autonomy: dict[str, object] = Field(default_factory=dict)
    gates: list[str] = Field(default_factory=list)
    recommended_decision: str = "reject"


class TaskActionKillSwitchRequest(BaseModel):
    enabled: bool = True
    clear_pending: bool = False
    project_id: int | None = None


class TaskActionKillSwitchResponse(BaseModel):
    ok: bool = True
    enabled: bool
    cleared_pending: int = 0


class TaskActionReplayRequest(BaseModel):
    device_id: str | None = None
    reason: str = "manual_replay"


class TaskActionReplayResponse(BaseModel):
    ok: bool = True
    original_message_id: int
    replay_message_id: int
    detail: str = ""


class TaskBootstrapWorkSessionRequest(BaseModel):
    project_id: int | None = None
    device_id: str | None = None
    focus_app: str = "mycelium"
    duration_minutes: int = 45
    consensus_fraction: float = 0.70
    species_confidence: float = 0.95


class TaskBootstrapWorkSessionResponse(BaseModel):
    ok: bool = True
    trajectory_id: int
    trajectory_key: str
    replica: TaskReplicaPublic


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


class DailyConsolidationResponse(BaseModel):
    ok: bool = True
    window_hours: int
    n_total: int
    n_accepted: int
    acceptance_rate: float
    adherence_mean: float | None = None
    top_domains: list[dict[str, object]] = Field(default_factory=list)
    summary_text: str = ""


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


class AssistantProfilePublic(BaseModel):
    ok: bool = True
    project_id: int | None = None
    given_name: str = "Synapse"
    gender_identity: str = "neutral"
    vocal_preset: str = "alloy"
    assistant_avatar_url: str = ""
    created_at: datetime
    updated_at: datetime
    is_default: bool = True


class AssistantProfileUpdateRequest(BaseModel):
    project_id: int | None = None
    given_name: str = "Synapse"
    gender_identity: str = "neutral"
    vocal_preset: str = "alloy"
    assistant_avatar_url: str = ""


class ChatSendRequest(BaseModel):
    project_id: int | None = None
    conversation_key: str = "default"
    channel: str = "app"  # app|telegram
    message: str


class ChatMessagePublic(BaseModel):
    id: int
    created_at: datetime
    project_id: int | None = None
    conversation_key: str
    channel: str
    role: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class ChatSendResponse(BaseModel):
    ok: bool = True
    user_message: ChatMessagePublic
    assistant_message: ChatMessagePublic
    delivered_external: bool = False


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessagePublic]


class LiveHiveNode(BaseModel):
    id: str
    kind: str
    label: str
    weight: float = 0.0


class LiveHiveEdge(BaseModel):
    source: str
    target: str
    flow: float = 0.0
    kind: str = "signal"


class LiveViscositySnapshot(BaseModel):
    score: float = 0.0
    band: str = "medium"  # low|medium|high
    prediction_state: str = "observe"  # flow|observe|gated
    battery_factor: float = 0.0
    thermal_factor: float = 0.0
    interruption_factor: float = 0.0
    battery_level: float | None = None
    cpu_temp_c: float | None = None
    recent_interruptions: int = 0


class LiveHiveStateResponse(BaseModel):
    ok: bool = True
    as_of: datetime
    window_minutes: int
    counters: dict[str, int] = Field(default_factory=dict)
    nodes: list[LiveHiveNode] = Field(default_factory=list)
    edges: list[LiveHiveEdge] = Field(default_factory=list)
    viscosity: LiveViscositySnapshot = Field(default_factory=LiveViscositySnapshot)


class HybridWorkSessionPredictRequest(BaseModel):
    project_id: int | None = None
    window_minutes: int = 120


class HybridWorkSessionPredictResponse(BaseModel):
    ok: bool = True
    project_id: int | None = None
    recommend: bool = False
    timing_score: float = 0.0
    governor_ok: bool = False
    governor_confidence: float = 0.0
    confidence_floor: float = 0.90
    n_signals: int = 0
    reasons: list[str] = Field(default_factory=list)
    suggested_minutes: int = 45


class AdaptiveDirectiveRequest(BaseModel):
    project_id: int | None = None
    window_minutes: int = 120
    base_duration_minutes: int = 45


class AdaptiveDirectiveResponse(BaseModel):
    ok: bool = True
    project_id: int | None = None
    base_duration_minutes: int = 45
    suggested_duration_minutes: int = 45
    strategy: str = "hold"  # hold|shorten|normalize
    reason: str = ""
    hybrid: HybridWorkSessionPredictResponse
    viscosity: LiveViscositySnapshot


class AdaptiveNodeRecommendation(BaseModel):
    device_id: str
    n_signals: int = 0
    suggested_duration_minutes: int = 45
    strategy: str = "hold"
    reason: str = ""
    viscosity: LiveViscositySnapshot


class AdaptiveMultiNodeDirectiveRequest(BaseModel):
    project_id: int | None = None
    window_minutes: int = 120
    base_duration_minutes: int = 45
    current_device_id: str | None = None
    candidate_device_ids: list[str] = Field(default_factory=list)


class AdaptiveMultiNodeDirectiveResponse(BaseModel):
    ok: bool = True
    project_id: int | None = None
    current_device_id: str | None = None
    recommended_device_id: str | None = None
    handoff_recommended: bool = False
    reason: str = ""
    hybrid: HybridWorkSessionPredictResponse
    recommendations: list[AdaptiveNodeRecommendation] = Field(default_factory=list)


class AutoHandoffLaunchRequest(BaseModel):
    project_id: int | None = None
    window_minutes: int = 120
    base_duration_minutes: int = 45
    current_device_id: str | None = None
    candidate_device_ids: list[str] = Field(default_factory=list)
    focus_app: str = "mycelium"


class AutoHandoffLaunchResponse(BaseModel):
    ok: bool = True
    project_id: int | None = None
    handoff_recommended: bool = False
    recommended_device_id: str | None = None
    launch_mode: str = "recovery"  # recovery|proposed|approved
    suggested_duration_minutes: int = 0
    reason: str = ""
    trajectory_id: int | None = None
    replica_id: int | None = None
    queued_device_action_id: int | None = None
    hybrid: HybridWorkSessionPredictResponse
    recommendations: list[AdaptiveNodeRecommendation] = Field(default_factory=list)


class AutoHandoffConfirmRequest(BaseModel):
    replica_id: int
    device_id: str | None = None


class AutoHandoffConfirmResponse(BaseModel):
    ok: bool = True
    replica_id: int
    queued_device_action_id: int
    detail: str = ""


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


class AdaptiveMemoryEntryPublic(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    project_id: int | None = None
    device_id: str
    lane: str
    memory_key: str
    source: str
    content: dict[str, object] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    strength: float
    decay_half_life_hours: float
    last_reinforced_at: datetime | None = None
    last_accessed_at: datetime | None = None


class AdaptiveMemoryUpsertRequest(BaseModel):
    project_id: int | None = None
    lane: str = "episodic"  # episodic|semantic|procedural
    memory_key: str
    content: dict[str, object] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"
    strength_delta: float = 0.10
    decay_half_life_hours: float = 168.0


class AdaptiveMemoryUpsertResponse(BaseModel):
    ok: bool = True
    memory: AdaptiveMemoryEntryPublic


class AdaptiveMemoryListResponse(BaseModel):
    memories: list[AdaptiveMemoryEntryPublic]


class AdaptiveMemoryReinforceRequest(BaseModel):
    delta: float = 0.10


class AdaptiveMemoryReinforceResponse(BaseModel):
    ok: bool = True
    memory: AdaptiveMemoryEntryPublic


class AdaptiveMemoryDecayRunRequest(BaseModel):
    project_id: int | None = None
    lane: str | None = None
    min_elapsed_hours: float = 1.0


class AdaptiveMemoryDecayRunResponse(BaseModel):
    ok: bool = True
    updated: int
    mean_strength_before: float
    mean_strength_after: float


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


class DeployVersionResponse(BaseModel):
    ok: bool = True
    app_name: str
    app_version: str
    git_sha: str
    build_id: str
    railway_environment: str


class HandoffSessionPublic(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    project_id: int | None = None
    current_device_id: str
    target_device_id: str
    replica_id: int | None = None
    status: str
    launch_mode: str
    attempt_count: int
    max_attempts: int
    timeout_at: datetime | None = None
    next_retry_at: datetime | None = None
    last_error: str = ""
    details: dict[str, object] = Field(default_factory=dict)


class HandoffSessionStartRequest(BaseModel):
    project_id: int | None = None
    window_minutes: int = 120
    base_duration_minutes: int = 45
    current_device_id: str | None = None
    candidate_device_ids: list[str] = Field(default_factory=list)
    focus_app: str = "mycelium"
    max_attempts: int = 3
    timeout_seconds: int = 300


class HandoffSessionStartResponse(BaseModel):
    ok: bool = True
    session: HandoffSessionPublic


class HandoffSessionTickRequest(BaseModel):
    retry_wait_seconds: int = 20


class HandoffSessionTickResponse(BaseModel):
    ok: bool = True
    session: HandoffSessionPublic


class TaskActionAuditItem(BaseModel):
    message_id: int
    created_at: datetime
    project_id: int | None = None
    device_id: str
    action_id: str
    capability: str
    confidence: float
    permission_tier: str
    kill_switch: bool
    min_confidence: float
    status: str
    gates: list[str] = Field(default_factory=list)
    would_pass_now: bool


class TaskActionAuditTimelineResponse(BaseModel):
    ok: bool = True
    items: list[TaskActionAuditItem] = Field(default_factory=list)
