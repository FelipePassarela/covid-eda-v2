from pathlib import Path

import joblib
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from covid_common.feature import TARGET


@pytest.fixture()
def pipeline_classifier() -> BaseEstimator:
    return LogisticRegression(random_state=42)


@pytest.fixture()
def fitted_pipeline(
    classification_df: pd.DataFrame, pipeline_classifier: BaseEstimator
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler().set_output(transform="pandas")),
            ("clf", pipeline_classifier),
        ]
    )
    pipeline.fit(classification_df.drop(columns=TARGET), classification_df[TARGET])
    return pipeline


@pytest.fixture()
def pipeline_path(tmp_path: Path, fitted_pipeline: Pipeline) -> Path:
    path = tmp_path / "model.pkl"
    joblib.dump(fitted_pipeline, path)
    return path
