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
]


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
from physml.agent import AgentAction, DataStream, PhysicsAgent
from physml.agent_api import PhysicsAgentSession
from physml.multitask_engine import MultiTaskPhysicsEngine
from physml.mycelium_agent import MyceliumAgent

#: Short alias — ``myco`` is identical to :class:`MyceliumAgent`.
myco = MyceliumAgent

from physml.drift import DriftDetector
from physml.federated import FederatedMyceliumAgent
from physml.evaluation import benchmark_agent, BenchmarkResult

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
]
