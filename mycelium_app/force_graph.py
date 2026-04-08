"""Force-directed graph data builder for the physics predictor ecosystem.

Transforms a PredictionResult's migration map and bonding map into a
graph structure suitable for rendering with a force-directed layout
(D3.js / canvas). Also builds a standalone graph from SedimentationResult.

Nodes = features, sized by mass / density, colored by layer / state.
Links = bonds (affinity + collinearity), weighted by affinity strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mycelium_app.physics_predictor import PredictionResult
from mycelium_app.sedimentation import SedimentationResult


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    label: str
    group: str
    mass: float
    charge: float
    viscosity: float
    velocity: float
    state: str
    layer: str
    depth: float
    entropy: float
    complex_id: int | None


@dataclass(frozen=True, slots=True)
class GraphLink:
    source: str
    target: str
    weight: float
    bond_type: str


def build_prediction_graph(result: PredictionResult) -> dict[str, Any]:
    """Build a force-directed graph from a supervised PredictionResult."""
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    seen_features: set[str] = set()

    for m in result.migration_map:
        if m.feature in seen_features:
            continue
        seen_features.add(m.feature)

        # Map state to layer name for color grouping
        if m.state == "trapped":
            layer = "turbulent"
        elif m.state == "dampened":
            layer = "suspension"
        else:
            layer = "bedrock"

        nodes.append({
            "id": m.feature,
            "label": m.feature,
            "group": str(m.feature_kind),
            "mass": round(float(m.mass), 4),
            "charge": round(float(m.charge), 4),
            "viscosity": round(float(m.viscosity), 4),
            "velocity": round(float(m.terminal_velocity), 4),
            "state": m.state,
            "layer": layer,
            "depth": round(abs(float(m.terminal_velocity)), 4),
            "entropy": round(float(m.entropy), 4),
            "complex_id": m.complex_id,
        })

    for b in result.bonding_map:
        if b.feature_a in seen_features and b.feature_b in seen_features:
            links.append({
                "source": b.feature_a,
                "target": b.feature_b,
                "weight": round(float(b.affinity), 4),
                "bond_type": getattr(b, "bond_type", "affinity"),
            })

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "target": result.target,
            "target_kind": str(result.target_kind),
            "plane": result.plane.value,
            "n_nodes": len(nodes),
            "n_links": len(links),
        },
    }


def build_sedimentation_graph(result: SedimentationResult) -> dict[str, Any]:
    """Build a force-directed graph from an unsupervised SedimentationResult."""
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    feature_set = {f.feature for f in result.features}

    for f in result.features:
        nodes.append({
            "id": f.feature,
            "label": f.feature,
            "group": f.layer,
            "mass": round(float(f.density), 4),
            "charge": 0.0,
            "viscosity": round(float(f.viscosity), 4),
            "velocity": round(float(f.settling_velocity), 4),
            "state": f.layer,
            "layer": f.layer,
            "depth": round(float(f.depth), 4),
            "entropy": round(float(f.entropy), 4),
            "complex_id": f.complex_id,
            "vif": f.vif,
        })

    # Links from the correlation matrix for features within the same complex
    corr = result.correlation_matrix
    for cx in result.complexes:
        feats = list(cx.features)
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                fa, fb = feats[i], feats[j]
                if fa in corr and fb in corr.get(fa, {}):
                    w = abs(float(corr[fa].get(fb, 0.0)))
                else:
                    w = cx.internal_cohesion
                links.append({
                    "source": fa,
                    "target": fb,
                    "weight": round(w, 4),
                    "bond_type": "flocculation",
                })

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "mode": "sedimentation",
            "n_rows": result.n_rows,
            "n_features": result.n_features,
            "n_nodes": len(nodes),
            "n_links": len(links),
            "n_complexes": len(result.complexes),
            "digest": result.digest,
        },
    }
