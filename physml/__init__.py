"""PhysML — Physics-inspired Machine Learning for tabular data.

The core engine models tabular features as charged particles undergoing
gel electrophoresis.  Feature "charges" (statistical associations with
the target) drive migration through a viscous medium whose resistance
is modulated by feature collinearity, distribution shape, and an
iterative PCR-style amplification step.

Public API
----------
run_physics_prediction   Low-level functional interface.
PhysicsPredictor         scikit-learn compatible estimator.
PhysicsPlane             Enum: solid | liquid | gas (medium preset).
PredictionResult         Rich result dataclass returned by the engine.
PredictorRuntimeState    Mutable state object for multi-run homeostasis.
NeuralPhysicsEngine      MLP + feature-attention backend (Stage 1–3).
PhysicsAgent             Autonomous observe/reward/adapt loop (Stage 4).
DataStream               Mini-batch streaming for big data (Stage 5).
PhysicsAgentSession      User-facing stateful session API (Stage 7).
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
from physml.estimator import PhysicsPredictor
from physml.neural_engine import NeuralPhysicsEngine, run_neural_prediction
from physml.agent import AgentAction, DataStream, PhysicsAgent
from physml.agent_api import PhysicsAgentSession

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
    "NeuralPhysicsEngine",
    "run_neural_prediction",
    # Stage 4 + 5
    "AgentAction",
    "PhysicsAgent",
    "DataStream",
    # Stage 7
    "PhysicsAgentSession",
]
