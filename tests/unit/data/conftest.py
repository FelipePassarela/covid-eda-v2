import pandas as pd
import pytest

from covid.data import CovidDataset, DataCleaningConfig


@pytest.fixture
def dataset() -> CovidDataset:
    data = {
        "id": [1, 2, 3],
        "feature1": [0.5, 0.6, 0.7],
        "feature2": [1.0, 1.1, 1.2],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    config = DataCleaningConfig(
        target_column="target", id_column="id", sparse_threshold=0.5
    )
    return CovidDataset.from_dataframe(df, cleaning_config=config)


@pytest.fixture
def sparse_dataset() -> CovidDataset:
    data = {
        "id": [1, 2, 3],
        "feature1": [0.5, 0.6, 0.7],
        "feature2": [1.0, 1.1, 1.2],
        "sparse_feature": [None, None, 1],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    config = DataCleaningConfig(
        target_column="target", id_column="id", sparse_threshold=0.5
    )
    return CovidDataset.from_dataframe(df, cleaning_config=config)
