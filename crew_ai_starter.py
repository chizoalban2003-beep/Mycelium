"""Minimal CrewAI starter wired to the physics prediction engine.

Run:
  /home/chizoalban2003/Mycelium/.venv/bin/python crew_ai_starter.py
"""

from __future__ import annotations

import os
import json

import pandas as pd
from crewai import Agent, Crew, Task

from mycelium_app.physics_predictor import PhysicsPlane, run_physics_prediction


def build_demo_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_hours": [1.0, 2.5, 3.5, 4.0, 5.5, 6.0],
            "sleep_hours": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
            "target": [52, 58, 63, 68, 74, 79],
        }
    )


def summarize_prediction(result) -> str:
    metrics = result.metrics
    payload = {
        "target": result.target,
        "target_kind": result.target_kind,
        "plane": result.plane.value,
        "rows": metrics.n_rows,
        "features_used": metrics.n_features_used,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "accuracy": metrics.accuracy,
        "preview_rows": result.preview_rows[:3],
    }
    return json.dumps(payload, indent=2, default=str)


def main() -> None:
    df = build_demo_dataframe()
    prediction = run_physics_prediction(
        df,
        target_col="target",
        plane=PhysicsPlane.solid,
        train_fraction=0.67,
        random_seed=42,
        n_cycles=5,
        max_preview_rows=3,
    )

    if prediction is None:
        print("\nPrediction abstained because the model was too uncertain.")
        return

    print("\n=== Prediction Summary ===")
    print(summarize_prediction(prediction))

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "\nCrewAI is installed, but no OPENAI_API_KEY is set yet. "
            "Set one, then rerun this script to enable the agent step."
        )
        return

    analyst = Agent(
        role="Prediction Analyst",
        goal="Turn the physics predictor output into a practical next step",
        backstory="You explain model output clearly for a human operator who wants simple guidance.",
        verbose=False,
    )

    task = Task(
        description=(
            "You are reviewing a fresh prediction-engine run.\n\n"
            f"Prediction summary:\n{summarize_prediction(prediction)}\n\n"
            "Give a short interpretation, mention whether the run looks healthy, and suggest one next step."
        ),
        expected_output="A concise plain-English summary with one recommendation.",
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=False)
    result = crew.kickoff()

    print("\n=== CrewAI Output ===")
    print(result)


if __name__ == "__main__":
    main()