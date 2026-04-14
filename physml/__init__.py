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
ToolSpec                 Stage 44 — JSON-schema tool specification.
ToolCall                 Stage 44 — structured tool call result.
ToolPlanner              Stage 44 — embedding + memory-based tool selection.
FeedbackBuffer           Stage 45 — bounded feedback buffer for online RLHF.
FeedbackItem             Stage 45 — single labelled feedback example.
OnlineRLHF               Stage 45 — online RLHF loop (partial_fit on feedback).
Specialist               Stage 46 — specialist agent descriptor.
OrchestratorResult       Stage 46 — routing result from AgentOrchestrator.
AgentOrchestrator        Stage 46 — multi-specialist routing coordinator.
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
]
