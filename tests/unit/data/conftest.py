import pandas as pd
import pytest

from covid.data import CovidDataset


@pytest.fixture
def dataset() -> CovidDataset:
    data = {
        "id": [1, 2, 3],
        "feature1": [0.5, 0.6, 0.7],
        "feature2": [1.0, 1.1, 1.2],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    return CovidDataset.from_dataframe(df, target="target", id_column="id")


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
    return CovidDataset.from_dataframe(
        df, target="target", id_column="id"
    ).without_sparse_columns(threshold=0.5)
