from collections.abc import Generator
from typing import Any

import pandas as pd
import pytest
from loguru import logger
from sklearn.datasets import make_classification

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
