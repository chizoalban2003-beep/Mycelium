from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

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
    gender: str = Field(default="")  # neutral|female|male|nonbinary|custom — mirrors to companion


class PasswordResetToken(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    token_hash: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    expires_at: datetime = Field(index=True)
    used_at: Optional[datetime] = Field(default=None, index=True)


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


class PhysicsLedgerEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    target_kind: str = Field(index=True)
    target_col: str = Field(default="", index=True)

    feature_cols_json: str = "[]"
    dtypes_json: str = "{}"

    preset_name: str | None = Field(default=None, index=True)
    preset_display: str | None = None

    applied_kwargs_json: str = "{}"
    score_metric: str = Field(default="", index=True)
    score_value: float = 0.0


class ExperienceBufferEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    # Stable identifier for export/import across devices.
    entry_uuid: str = Field(default_factory=lambda: uuid4().hex, index=True, unique=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    device_id: str = Field(default="", index=True)
    source: str = Field(default="text", index=True)  # e.g. text,file,api
    modality: str = Field(default="auto", index=True)  # e.g. finance,style,grammar

    raw_text: str = ""
    extracted_json: str = "{}"  # JSON payload (ionized atoms / style profile / etc.)
    physics_used_json: str = "{}"  # optional: physics knobs used during an action

    confidence: float | None = Field(default=None, index=True)
    feedback: str = ""  # optional human feedback (thumbs-up/down notes)
    tags_json: str = "[]"


class NexusPolicy(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    policy_json: str = "{}"


class AssistantProfile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    given_name: str = Field(default="Synapse", index=True)
    gender_identity: str = Field(default="neutral", index=True)
    vocal_preset: str = Field(default="alloy", index=True)


class AssistantAvatarProfile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    assistant_avatar_url: str = Field(default="", index=False)


class HiveOutboxReport(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)
    device_id: str = Field(default="", index=True)
    report_json: str = "{}"  # anonymized aggregates only
    submitted_at: Optional[datetime] = Field(default=None, index=True)


class HiveGlobalUpdate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    update_uuid: str = Field(default_factory=lambda: uuid4().hex, index=True, unique=True)
    source: str = Field(default="manual_import", index=True)
    version: str = Field(default="", index=True)
    update_json: str = "{}"  # e.g. recommended knobs, safe allowlisted fields


class HiveDevice(SQLModel, table=True):
    """Track Hive child devices that have reported in.

    Used to generate a one-time "first connect" operator nudge without scanning
    the full HiveGlobalUpdate JSON history.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    device_id: str = Field(default="", index=True, unique=True)

    first_seen_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    last_seen_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    last_source: str = Field(default="", index=True)



class HiveOutboxMessage(SQLModel, table=True):
    """Generic Hive outbox message.

    This is intentionally separate from HiveOutboxReport so we can add new
    message kinds without changing existing table schemas.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)
    device_id: str = Field(default="", index=True)

    kind: str = Field(default="", index=True)  # e.g. wisdom_whisper, homeostasis_failure
    payload_json: str = "{}"  # JSON dict; must remain non-sensitive by policy
    submitted_at: Optional[datetime] = Field(default=None, index=True)


class SignalLedgerEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    device_id: str = Field(default="", index=True)
    signal_type: str = Field(default="", index=True)  # e.g. screen,onoff,app,network,text_sample
    payload_json: str = "{}"  # JSON dict; must never contain raw secrets by policy


class MissionLogLedgerEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    device_id: str = Field(default="", index=True)
    source_kind: str = Field(default="", index=True)  # signal|growth|nudge|diagnostic
    source_ref: str = Field(default="", index=True, unique=True)

    mode: str = Field(default="", index=True)
    tier: str = Field(default="", index=True)
    title: str = Field(default="", index=True)
    detail: str = Field(default="")
    delta: float | None = Field(default=None, index=True)
    delta_text: str = Field(default="")


class GrowthLedgerEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    device_id: str = Field(default="", index=True)

    # What was being optimized.
    domain: str = Field(default="", index=True)  # e.g. telemetry_next_app, grammar_rewrite
    metric: str = Field(default="", index=True)  # e.g. r2, acceptance_rate, mae
    score: float = Field(default=0.0, index=True)

    # Whether the sweep outcome was accepted / considered a 'best sweep'.
    accepted: bool = Field(default=False, index=True)

    # Optional structured payloads (must remain non-sensitive).
    proposal_json: str = "{}"  # what it tried
    outcome_json: str = "{}"  # what happened
    notes: str = ""


class HomeostasisState(SQLModel, table=True):
    """Persisted homeostasis snapshot.

    This is the Nexus "body" state: mood broadcast + resource health + last
    self-repair / pruning actions.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    mood: str = Field(default="curious", index=True)
    mood_signal_json: str = "{}"
    identity_hash: str = Field(default="", index=True)

    agitated_cycles: int = 0
    last_deep_breath_at: Optional[datetime] = Field(default=None, index=True)
    last_identity_backup_at: Optional[datetime] = Field(default=None, index=True)

    disk_total_bytes: int = 0
    disk_free_bytes: int = 0

    venv_present: bool = False
    notes: str = ""


class NexusNudge(SQLModel, table=True):
    """A small user-facing notification (the system's 'voice')."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    kind: str = Field(default="", index=True)  # e.g. wisdom_update
    title: str = ""
    message: str = ""
    payload_json: str = "{}"

    seen_at: Optional[datetime] = Field(default=None, index=True)


class WisdomIntegrationState(SQLModel, table=True):
    """Track the last broadcasted wisdom a child has integrated.

    Stored separately to avoid altering HomeostasisState schema.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    last_wisdom_digest: str = Field(default="", index=True)
    last_wisdom_kwargs_json: str = "{}"
    last_nudge_at: Optional[datetime] = Field(default=None, index=True)


class TaskTrajectory(SQLModel, table=True):
    """Observed user action sequence used for behavioral mirroring."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    device_id: str = Field(default="", index=True)
    trajectory_key: str = Field(default="", index=True)

    # JSON blobs (privacy-safe, no raw secrets).
    sequence_json: str = "[]"  # e.g. ["open_spotify", "search_deep_focus", "set_volume_40"]
    app_state_json: str = "{}"
    input_vector_json: str = "{}"

    confidence: float = Field(default=0.0, index=True)
    support_count: int = Field(default=1, index=True)


class TaskReplica(SQLModel, table=True):
    """Executable action proposal derived from trajectories + hive consensus."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    device_id: str = Field(default="", index=True)
    title: str = Field(default="", index=True)
    trajectory_key: str = Field(default="", index=True)

    consensus_fraction: float = Field(default=0.0, index=True)
    species_confidence: float = Field(default=0.0, index=True)
    capability: str = Field(default="", index=True)
    command_json: str = "{}"

    # proposed | approved | rejected | executed | failed
    status: str = Field(default="proposed", index=True)
    approved_at: Optional[datetime] = Field(default=None, index=True)
    executed_at: Optional[datetime] = Field(default=None, index=True)
    notes: str = ""


class AdaptiveMemoryEntry(SQLModel, table=True):
    """Adaptive memory lane entry.

    Lanes:
    - episodic: session/event memories
    - semantic: stable facts/preferences
    - procedural: routines/scripts that should become automatic
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)
    device_id: str = Field(default="", index=True)

    lane: str = Field(default="episodic", index=True)  # episodic|semantic|procedural
    memory_key: str = Field(default="", index=True)
    source: str = Field(default="manual", index=True)

    content_json: str = "{}"
    tags_json: str = "[]"

    strength: float = Field(default=0.5, index=True)  # [0,1]
    decay_half_life_hours: float = Field(default=168.0)  # one week default

    last_reinforced_at: Optional[datetime] = Field(default=None, index=True)
    last_accessed_at: Optional[datetime] = Field(default=None, index=True)


class MetricSnapshot(SQLModel, table=True):
    """Persisted metric measurement for validation-shadow honesty."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    # What was measured
    dataset_digest: str = Field(default="", index=True)
    wisdom_digest: str = Field(default="", index=True)
    phase: str = Field(default="", index=True)  # baseline|trial

    target_col: str = ""
    target_kind: str = Field(default="", index=True)
    metric_name: str = Field(default="", index=True)  # mae|rmse|accuracy
    metric_value: float = 0.0

    # Repro context
    kwargs_json: str = "{}"
    notes: str = ""


class ConversationMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    conversation_key: str = Field(default="default", index=True)
    channel: str = Field(default="app", index=True)  # app|telegram|sms|email
    role: str = Field(default="user", index=True)  # user|assistant|system
    content: str = ""
    metadata_json: str = "{}"

    delivered_at: Optional[datetime] = Field(default=None, index=True)


class MetricCausalTrace(SQLModel, table=True):
    """Persisted explanation artifact for Validation Shadow.

    Stored in its own table so existing SQLite DBs don't require ALTER TABLE.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    baseline_snapshot_id: int = Field(foreign_key="metricsnapshot.id", index=True)
    trial_snapshot_id: int = Field(foreign_key="metricsnapshot.id", index=True)

    dataset_digest: str = Field(default="", index=True)
    wisdom_digest: str = Field(default="", index=True)

    metric_name: str = Field(default="", index=True)
    improvement_frac: float | None = None

    method: str = Field(default="weights_shift", index=True)
    narrative: str = ""
    top_shifts_json: str = "[]"  # JSON list of top feature shifts


class CuriosityCase(SQLModel, table=True):
    """A high-error sample that requests human ground truth.

    Privacy stance:
    - stores a minimal excerpt (allowlisted columns only)
    - can be answered without exporting raw row data
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    case_uuid: str = Field(default="", index=True, unique=True)
    dataset_digest: str = Field(default="", index=True)
    target_col: str = Field(default="", index=True)
    target_kind: str = Field(default="", index=True)

    row_index: int | None = Field(default=None, index=True)
    row_fingerprint: str = Field(default="", index=True)

    predicted_json: str = "null"
    actual_json: str = "null"
    error_value: float = 0.0
    error_kind: str = Field(default="abs_error", index=True)  # abs_error|miss

    excerpt_json: str = "{}"  # allowlisted columns only
    question: str = ""

    status: str = Field(default="pending", index=True)  # pending|answered|dismissed
    answered_at: Optional[datetime] = Field(default=None, index=True)
    dismissed_at: Optional[datetime] = Field(default=None, index=True)


class CuriosityAnswer(SQLModel, table=True):
    """User-provided ground truth/explanation for a CuriosityCase."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    case_id: int = Field(foreign_key="curiositycase.id", index=True)
    answer_text: str = ""
    corrected_target_json: str = "null"  # optional corrected label/value
    tags_json: str = "[]"

    exported_to_hive_at: Optional[datetime] = Field(default=None, index=True)


class HandoffSession(SQLModel, table=True):
    """Deterministic handoff execution session with retries and timeout state."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)

    current_device_id: str = Field(default="", index=True)
    target_device_id: str = Field(default="", index=True)

    replica_id: Optional[int] = Field(default=None, foreign_key="taskreplica.id", index=True)

    # launched|proposed|queued|waiting_retry|completed|failed|timed_out|recovery
    status: str = Field(default="launched", index=True)
    launch_mode: str = Field(default="proposed", index=True)

    attempt_count: int = Field(default=0, index=True)
    max_attempts: int = Field(default=3)

    timeout_at: Optional[datetime] = Field(default=None, index=True)
    next_retry_at: Optional[datetime] = Field(default=None, index=True)
    last_error: str = ""

    details_json: str = "{}"
