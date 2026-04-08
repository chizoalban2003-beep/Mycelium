"""API routes for unsupervised sedimentation and force-directed graph visualization."""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlmodel import Session

import pandas as pd

from mycelium_app.db import get_session
from mycelium_app.deps import get_current_user
from mycelium_app.force_graph import build_prediction_graph, build_sedimentation_graph
from mycelium_app.models import User
from mycelium_app.sedimentation import run_sedimentation


router = APIRouter(prefix="/api/sedimentation", tags=["sedimentation"])


@router.post("/settle")
async def settle(
    file: UploadFile = File(...),
    max_features: int = Form(200),
    flocculation_threshold: float = Form(0.7),
    n_iterations: int = Form(10),
    gravity: float = Form(1.0),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Run unsupervised gravitational sedimentation on uploaded CSV."""
    raw = await file.read()
    if not raw:
        return {"ok": False, "error": "Empty file"}

    max_features = max(2, min(int(max_features), 500))
    flocculation_threshold = max(0.1, min(float(flocculation_threshold), 0.99))
    n_iterations = max(1, min(int(n_iterations), 50))
    gravity = max(0.1, min(float(gravity), 10.0))

    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=50_000)
    except Exception as e:
        return {"ok": False, "error": f"CSV parse error: {e}"}

    result = run_sedimentation(
        df,
        max_features=max_features,
        flocculation_threshold=flocculation_threshold,
        n_iterations=n_iterations,
        gravity=gravity,
    )

    graph = build_sedimentation_graph(result)

    return {
        "ok": True,
        "n_rows": result.n_rows,
        "n_features": result.n_features,
        "digest": result.digest,
        "layers": result.layer_summary,
        "features": [
            {
                "feature": f.feature,
                "density": f.density,
                "viscosity": f.viscosity,
                "settling_velocity": f.settling_velocity,
                "depth": f.depth,
                "layer": f.layer,
                "complex_id": f.complex_id,
                "complex_size": f.complex_size,
                "entropy": f.entropy,
                "variance": f.variance,
                "correlation_sum": f.correlation_sum,
                "vif": f.vif,
            }
            for f in result.features
        ],
        "complexes": [
            {
                "complex_id": c.complex_id,
                "features": list(c.features),
                "combined_density": c.combined_density,
                "mean_settling_velocity": c.mean_settling_velocity,
                "mean_depth": c.mean_depth,
                "layer": c.layer,
                "internal_cohesion": c.internal_cohesion,
            }
            for c in result.complexes
        ],
        "graph": graph,
    }
