from collections.abc import Generator
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from covid.common.feature import TARGET


@pytest.fixture(autouse=True)
def silence_loguru() -> Generator[None, Any]:
    logger.remove()
    yield


@pytest.fixture()
def classification_df() -> pd.DataFrame:
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df[TARGET] = y
    return df


@pytest.fixture()
def data_path(tmp_path: Path, classification_df: pd.DataFrame) -> Path:
    path = tmp_path / "data.csv"
    classification_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def fitted_pipeline(classification_df: pd.DataFrame) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler().set_output(transform="pandas")),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(classification_df.drop(columns=TARGET), classification_df[TARGET])
    return pipeline


@pytest.fixture()
def pipeline_path(tmp_path: Path, fitted_pipeline: Pipeline) -> Path:
    path = tmp_path / "model.pkl"
    joblib.dump(fitted_pipeline, path)
    return path
