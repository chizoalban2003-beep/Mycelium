"""PhysML — Physics-inspired Machine Learning for tabular data.

The core engine models tabular features as charged particles undergoing
gel electrophoresis.  Feature "charges" (statistical associations with
the target) drive migration through a viscous medium whose resistance
is modulated by feature collinearity, distribution shape, and an
iterative PCR-style amplification step.

Public API
----------
run_physics_prediction   Low-level functional interface.
PhysicsPredictor         scikit-learn compatible estimator (base class).
PhysicsRegressor         PhysicsPredictor with regression-optimised defaults.
PhysicsClassifier        PhysicsPredictor with classification-optimised defaults.
CompetitiveEnsemblePredictor  Stage 36 stacking ensemble (fast, competitive).
PhysicsPlane             Enum: solid | liquid | gas (medium preset).
PredictionResult         Rich result dataclass returned by the engine.
PredictorRuntimeState    Mutable state object for multi-run homeostasis.
NeuralPhysicsEngine      MLP + feature-attention backend (Stage 1–3).
PhysicsAgent             Autonomous observe/reward/adapt loop (Stage 4).
DataStream               Mini-batch streaming for big data (Stage 5).
PhysicsAgentSession      User-facing stateful session API (Stage 7).
MultiTaskPhysicsEngine   Shared-trunk multi-task engine (Stage 9).
MyceliumAgent            Flagship autonomous agent — the project's top-level
                         branded class combining Stages 8–11.
myco                     Short alias for MyceliumAgent (``from physml import myco``).
DriftDetector            Online concept-drift detector (Stage 17).
FederatedMyceliumAgent   Federated learning coordinator (Stage 19).
benchmark_agent          Evaluation harness returning BenchmarkResult (Stage 14).
ModelRegistry            Lightweight JSONL model registry (Stage 29).
Featurizer               Raw-input → float32 vector converter (Stage 30).
Tool                     Named callable for agentic tool-use (Stage 31).
ToolRegistry             Registry for Tool objects (Stage 31).
AutonomousLoop           Agent + tools agentic loop (Stage 31).
GoalPlanner              Multi-step goal decomposition (Stage 32).
SubTask                  Single sub-goal dataclass (Stage 32).
EpisodicMemory           kNN episodic memory store (Stage 33).
pretrain_neural_engine   Masked-feature pretraining function (Stage 34).
pretrain_mycelium        Convenience pretraining wrapper (Stage 34).
ParallelDataStream       Concurrent mini-batch processor (Stage 35).
CompetitiveEnsemblePredictor  Stage 36 stacking ensemble (fast, competitive).
run_goal                 Stage 37 — goal-driven closed autonomous loop (method on MyceliumAgent).
attach_memory            Stage 38 — attach EpisodicMemory for auto episode recording (method on MyceliumAgent).
self_evaluate            Stage 39 — held-out accuracy/calibration self-evaluation (method on MyceliumAgent).
self_improve             Stage 40 — auto-tune threshold based on self-eval (method on MyceliumAgent).
introspect               Stage 41 — rich internal-state summary (method on MyceliumAgent).
Stage 42 — bug fixes: O(1) memory eviction, no double inference, real partial_fit in self_improve.
LifelongLearner          Stage 69 — continuous self-improvement loop (chunk-based streaming,
                         auto self-improve when accuracy dips, competitive report on demand).
HyperTuner               Stage 70 — autonomous hyperparameter self-tuning (AutoML into self-improve
                         cycle; best configs stored in KnowledgeGraph).
SelfHealer               Stage 71 — self-healing / recovery (AnomalyGuard + AgentCheckpoint rollback
                         + curriculum reset on anomaly or model collapse).
EvalScheduler            Stage 73 — autonomous evaluation & reporting (scheduled CompetitiveReport
                         runs, KnowledgeGraph logging, alert on rank drop).
SelfPlay                 Stage 74 — multi-agent adversarial self-play (two AutonomousAgent instances
                         compete in CompetitiveArena, exchange experience via FederatedMyceliumAgent).
ToolSpec                 Stage 44 — JSON-schema tool specification.
ToolCall                 Stage 44 — structured tool call result.
ToolPlanner              Stage 44 — embedding + memory-based tool selection.
FeedbackBuffer           Stage 45 — bounded feedback buffer for online RLHF.
FeedbackItem             Stage 45 — single labelled feedback example.
OnlineRLHF               Stage 45 — online RLHF loop (partial_fit on feedback).
Specialist               Stage 46 — specialist agent descriptor.
OrchestratorResult       Stage 46 — routing result from AgentOrchestrator.
AgentOrchestrator        Stage 46 — multi-specialist routing coordinator.
AutoMLOptimizer          Stage 47 — successive-halving hyperparameter search.
ConformalClassifier      Stage 48 — split-conformal classifier (valid prediction sets).
ConformalRegressor       Stage 48 — split-conformal regressor (valid prediction intervals).
Explainer                Stage 49 — permutation-importance feature attribution.
explain_agent            Stage 49 — convenience: fit Explainer from an agent.
AgentCheckpoint          Stage 50 — joblib-based full-agent save/load.
MetaLearner              Stage 51 — strategy selector via cross-task performance history.
"""

from physml.predictor import (
    PhysicsPlane,
    PredictionMetrics,
    PredictionResult,
    PredictorError,
    PredictorRuntimeState,
    WeightInfo,
    MigrationInfo,
    BondInfo,
    IterationInfo,
    EquilibriumZone,
    infer_target_kind,
    infer_feature_kind,
    run_physics_prediction,
    serialize_predictor_state,
    deserialize_predictor_state,
    save_predictor_state,
    load_predictor_state,
    prune_predictor_state,
    update_predictor_state_from_result,
    serialize_metrics,
    clean_tabular_dataframe,
)
from physml.estimator import PhysicsPredictor, PhysicsRegressor, PhysicsClassifier
from physml.neural_engine import NeuralPhysicsEngine, run_neural_prediction
from physml.ensemble_predictor import CompetitiveEnsemblePredictor
from physml.agent import AgentAction, DataStream, PhysicsAgent
from physml.agent_api import PhysicsAgentSession
from physml.multitask_engine import MultiTaskPhysicsEngine
from physml.mycelium_agent import MyceliumAgent

#: Short alias — ``myco`` is identical to :class:`MyceliumAgent`.
myco = MyceliumAgent

from physml.drift import DriftDetector
from physml.federated import FederatedMyceliumAgent
from physml.evaluation import benchmark_agent, BenchmarkResult
from physml.registry import ModelRegistry

# Stage 30 — Featurizer
from physml.featurizer import Featurizer

# Stage 31 — Tool-calling support
from physml.tools import Tool, ToolRegistry, AutonomousLoop

# Stage 32 — Goal planner
from physml.planner import GoalPlanner, SubTask

# Stage 33 — Episodic memory
from physml.memory import EpisodicMemory

# Stage 34 — Pretraining
from physml.pretrain import pretrain_neural_engine, pretrain_mycelium

# Stage 35 — Parallel data stream
from physml.stream_worker import ParallelDataStream

# Stage 44 — Structured tool-calling protocol
from physml.tool_planner import ToolSpec, ToolCall, ToolPlanner

# Stage 45 — FeedbackBuffer + online RLHF
from physml.feedback import FeedbackBuffer, FeedbackItem, OnlineRLHF

# Stage 46 — AgentOrchestrator
from physml.orchestrator import Specialist, OrchestratorResult, AgentOrchestrator

# Stage 47 — AutoMLOptimizer
from physml.automl import AutoMLOptimizer

# Stage 48 — Conformal Prediction
from physml.conformal import ConformalClassifier, ConformalRegressor

# Stage 49 — Explainability
from physml.explainability import Explainer, explain_agent

# Stage 50 — AgentCheckpoint
from physml.checkpoint import AgentCheckpoint

# Stage 51 — MetaLearner
from physml.meta_learner import MetaLearner

# Stage 52 — Prioritized Replay Buffer
from physml.replay_buffer import ReplayBuffer, PrioritizedReplay, Transition

# Stage 53 — HyperScheduler
from physml.scheduler import (
    StepSchedule,
    CosineSchedule,
    ExponentialSchedule,
    LinearSchedule,
    HyperScheduler,
)

# Stage 54 — AnomalyGuard
from physml.anomaly import AnomalyGuard, AnomalyResult

# Stage 55 — MultiObjectiveOptimizer
from physml.multiobjective import MultiObjectiveOptimizer, Solution

# Stage 56 — AgentProfiler
from physml.profiler import AgentProfiler, ProfileEntry

# Stage 57 — KnowledgeGraph
from physml.knowledge_graph import KnowledgeGraph, KnowledgeNode

# Stage 58 — RewardShaper
from physml.reward_shaper import RewardShaper

# Stage 59 — CurriculumScheduler
from physml.curriculum import CurriculumScheduler

# Stage 60 — SyntheticDataGenerator
from physml.synthetic_data import SyntheticDataGenerator

# Stage 61 — UncertaintyEstimator
from physml.uncertainty import UncertaintyEstimator

# Stage 62 — WorldModel
from physml.world_model import WorldModel

# Stage 63 — IntrinsicMotivation
from physml.intrinsic import IntrinsicMotivation

# Stage 64 — CompetitiveArena
from physml.arena import CompetitiveArena, ArenaResult

# Stage 65 — GoalConditionedPolicy
from physml.goal_policy import GoalSpec, GoalConditionedPolicy

# Stage 66 — SafetyMonitor
from physml.safety import SafetyConstraint, SafetyViolation, SafetyMonitor

# Stage 67 — AutonomousAgent (full integration)
from physml.autonomous_agent import AutonomousAgent

# Stage 68 — CompetitiveReport
from physml.competitive_report import CompetitiveReport

# Stage 69 — LifelongLearner (continuous self-improvement loop)
from physml.lifelong import LifelongLearner, RoundResult

# Stage 70 — HyperTuner (autonomous hyperparameter self-tuning)
from physml.hyper_tuner import HyperTuner, TuneResult

# Stage 71 — SelfHealer (anomaly-triggered checkpoint rollback)
from physml.self_healer import SelfHealer, HealingIncident

# Stage 73 — EvalScheduler (autonomous evaluation & reporting)
from physml.eval_scheduler import EvalScheduler, ScheduledReport

# Stage 74 — SelfPlay (multi-agent adversarial self-play)
from physml.self_play import SelfPlay, PlayRound

# Stage 75 — CausalGraph (correlation-based causal discovery)
from physml.causal_graph import CausalGraph, CausalEdge

# Stage 76 — PrivacyEngine (differential-privacy wrapper)
from physml.privacy_engine import PrivacyEngine, PrivacyBudget

# Stage 77 — TimeSeriesAdapter (time-series → tabular features)
from physml.timeseries_adapter import TimeSeriesAdapter, AdapterResult

# Stage 78 — ExperimentTracker (lightweight ML experiment tracking)
from physml.experiment_tracker import ExperimentTracker, Run

# Stage 79 — ModelDistillery (knowledge distillation)
from physml.model_distillery import ModelDistillery, DistillationResult

__all__ = [
    "PhysicsPlane",
    "PredictionMetrics",
    "PredictionResult",
    "PredictorError",
    "PredictorRuntimeState",
    "WeightInfo",
    "MigrationInfo",
    "BondInfo",
    "IterationInfo",
    "EquilibriumZone",
    "infer_target_kind",
    "infer_feature_kind",
    "run_physics_prediction",
    "serialize_predictor_state",
    "deserialize_predictor_state",
    "save_predictor_state",
    "load_predictor_state",
    "prune_predictor_state",
    "update_predictor_state_from_result",
    "serialize_metrics",
    "clean_tabular_dataframe",
    "PhysicsPredictor",
    "PhysicsRegressor",
    "PhysicsClassifier",
    "CompetitiveEnsemblePredictor",
    "NeuralPhysicsEngine",
    "run_neural_prediction",
    # Stage 4 + 5
    "AgentAction",
    "PhysicsAgent",
    "DataStream",
    # Stage 7
    "PhysicsAgentSession",
    # Stage 9
    "MultiTaskPhysicsEngine",
    # Stage 11 — flagship class
    "MyceliumAgent",
    "myco",  # shorthand alias for MyceliumAgent
    # Stage 14 — evaluation harness
    "benchmark_agent",
    "BenchmarkResult",
    # Stage 17 — drift detection
    "DriftDetector",
    # Stage 19 — federated learning
    "FederatedMyceliumAgent",
    # Stage 29 — model registry
    "ModelRegistry",
    # Stage 30 — featurizer
    "Featurizer",
    # Stage 31 — tool-calling support
    "Tool",
    "ToolRegistry",
    "AutonomousLoop",
    # Stage 32 — goal planner
    "GoalPlanner",
    "SubTask",
    # Stage 33 — episodic memory
    "EpisodicMemory",
    # Stage 34 — pretraining
    "pretrain_neural_engine",
    "pretrain_mycelium",
    # Stage 35 — parallel data stream
    "ParallelDataStream",
    # Stage 44 — structured tool-calling protocol
    "ToolSpec",
    "ToolCall",
    "ToolPlanner",
    # Stage 45 — FeedbackBuffer + online RLHF
    "FeedbackBuffer",
    "FeedbackItem",
    "OnlineRLHF",
    # Stage 46 — AgentOrchestrator
    "Specialist",
    "OrchestratorResult",
    "AgentOrchestrator",
    # Stage 47 — AutoMLOptimizer
    "AutoMLOptimizer",
    # Stage 48 — Conformal Prediction
    "ConformalClassifier",
    "ConformalRegressor",
    # Stage 49 — Explainability
    "Explainer",
    "explain_agent",
    # Stage 50 — AgentCheckpoint
    "AgentCheckpoint",
    # Stage 51 — MetaLearner
    "MetaLearner",
    # Stage 52 — Prioritized Replay Buffer
    "ReplayBuffer",
    "PrioritizedReplay",
    "Transition",
    # Stage 53 — HyperScheduler
    "StepSchedule",
    "CosineSchedule",
    "ExponentialSchedule",
    "LinearSchedule",
    "HyperScheduler",
    # Stage 54 — AnomalyGuard
    "AnomalyGuard",
    "AnomalyResult",
    # Stage 55 — MultiObjectiveOptimizer
    "MultiObjectiveOptimizer",
    "Solution",
    # Stage 56 — AgentProfiler
    "AgentProfiler",
    "ProfileEntry",
    # Stage 57 — KnowledgeGraph
    "KnowledgeGraph",
    "KnowledgeNode",
    # Stage 58 — RewardShaper
    "RewardShaper",
    # Stage 59 — CurriculumScheduler
    "CurriculumScheduler",
    # Stage 60 — SyntheticDataGenerator
    "SyntheticDataGenerator",
    # Stage 61 — UncertaintyEstimator
    "UncertaintyEstimator",
    # Stage 62 — WorldModel
    "WorldModel",
    # Stage 63 — IntrinsicMotivation
    "IntrinsicMotivation",
    # Stage 64 — CompetitiveArena
    "CompetitiveArena",
    "ArenaResult",
    # Stage 65 — GoalConditionedPolicy
    "GoalSpec",
    "GoalConditionedPolicy",
    # Stage 66 — SafetyMonitor
    "SafetyConstraint",
    "SafetyViolation",
    "SafetyMonitor",
    # Stage 67 — AutonomousAgent
    "AutonomousAgent",
    # Stage 68 — CompetitiveReport
    "CompetitiveReport",
    # Stage 69 — LifelongLearner
    "LifelongLearner",
    "RoundResult",
    # Stage 70 — HyperTuner
    "HyperTuner",
    "TuneResult",
    # Stage 71 — SelfHealer
    "SelfHealer",
    "HealingIncident",
    # Stage 73 — EvalScheduler
    "EvalScheduler",
    "ScheduledReport",
    # Stage 74 — SelfPlay
    "SelfPlay",
    "PlayRound",
    # Stage 75 — CausalGraph
    "CausalGraph",
    "CausalEdge",
    # Stage 76 — PrivacyEngine
    "PrivacyEngine",
    "PrivacyBudget",
    # Stage 77 — TimeSeriesAdapter
    "TimeSeriesAdapter",
    "AdapterResult",
    # Stage 78 — ExperimentTracker
    "ExperimentTracker",
    "Run",
    # Stage 79 — ModelDistillery
    "ModelDistillery",
    "DistillationResult",
]
