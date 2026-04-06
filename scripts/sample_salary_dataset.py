#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_salary_dataset(nrows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n = int(nrows)
    if n < 10:
        raise ValueError("nrows must be >= 10")

    titles = np.array(
        [
            "Data Analyst",
            "Data Scientist",
            "ML Engineer",
            "Backend Engineer",
            "Product Analyst",
            "Research Assistant",
            "DevOps Engineer",
            "Business Analyst",
        ],
        dtype=object,
    )
    locations = np.array(["NY", "SF", "LA", "Austin", "Remote", "Berlin", "London", "Toronto"], dtype=object)
    edu = np.array(["high_school", "bachelors", "masters", "phd"], dtype=object)

    years = rng.integers(0, 21, size=n)
    remote = rng.random(size=n) < 0.42
    company_size = rng.choice([10, 25, 50, 100, 250, 500, 1000, 5000], size=n, replace=True)

    title = rng.choice(titles, size=n, replace=True)
    location = rng.choice(locations, size=n, replace=True)
    education = rng.choice(edu, size=n, replace=True, p=[0.20, 0.50, 0.25, 0.05])

    edu_bonus = pd.Series(education).map({"high_school": 0, "bachelors": 6000, "masters": 12000, "phd": 22000}).to_numpy()
    title_bonus = pd.Series(title).map(
        {
            "Data Analyst": 2000,
            "Business Analyst": 1500,
            "Product Analyst": 2500,
            "Data Scientist": 12000,
            "ML Engineer": 14000,
            "Backend Engineer": 9000,
            "DevOps Engineer": 11000,
            "Research Assistant": -2000,
        }
    ).to_numpy()

    loc_bonus = pd.Series(location).map({"NY": 10000, "SF": 16000, "LA": 7000, "Austin": 4000, "Remote": 5000, "Berlin": 2000, "London": 8000, "Toronto": 3000}).to_numpy()

    noise = rng.normal(0.0, 6500.0, size=n)

    salary = (
        48000
        + years * 4200
        + (remote.astype(int) * 3500)
        + np.log1p(company_size) * 2200
        + edu_bonus
        + title_bonus
        + loc_bonus
        + noise
    )
    salary = np.clip(salary, 18000, None)

    df = pd.DataFrame(
        {
            "job_title": title,
            "location": location,
            "education": education,
            "years_experience": years.astype(int),
            "company_size": company_size.astype(int),
            "remote_work": np.where(remote, "yes", "no"),
            "salary": salary.astype(float),
        }
    )

    # Inject a bit of missingness to exercise cleaning.
    for col, frac in [("education", 0.03), ("job_title", 0.02), ("location", 0.02), ("years_experience", 0.01)]:
        mask = rng.random(size=n) < float(frac)
        df.loc[mask, col] = pd.NA

    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a small synthetic salary dataset for local benchmarks")
    p.add_argument("--out", default="tmp_eval/sample_salary_dataset.csv", help="Output CSV path (default: tmp_eval/sample_salary_dataset.csv)")
    p.add_argument("--nrows", type=int, default=8000, help="Number of rows (default: 8000)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = make_sample_salary_dataset(int(args.nrows), int(args.seed))
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
