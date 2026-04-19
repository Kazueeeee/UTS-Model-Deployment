"""Shared data loading and feature engineering utilities for the student placement project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def load_datasets(features_path: str | Path, targets_path: str | Path) -> pd.DataFrame:
    """Load and merge feature and target CSV files on Student_ID."""
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    merged = features.merge(targets, on="Student_ID", how="inner")
    return merged


class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create lightweight, interpretable features from the raw student dataset.

    Added columns:
    - academic_average
    - technical_average
    - activity_score
    - lifestyle_balance
    - has_extra_curricular
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["academic_average"] = (
            X["cgpa"] * 10 + X["tenth_percentage"] + X["twelfth_percentage"]
        ) / 3.0

        X["technical_average"] = X[
            ["coding_skill_rating", "communication_skill_rating", "aptitude_skill_rating"]
        ].mean(axis=1)

        X["activity_score"] = X[
            ["projects_completed", "internships_completed", "hackathons_participated", "certifications_count"]
        ].sum(axis=1)

        X["lifestyle_balance"] = X["sleep_hours"] - X["stress_level"]

        X["has_extra_curricular"] = X["extracurricular_involvement"].map(
            {"Low": 0, "Medium": 1, "High": 2}
        )

        return X
