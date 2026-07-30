from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

from covid import feature


def load_data(data_path: Path) -> DataFrame:
    return pd.read_csv(data_path, dtype={feature.ID: str})


def split_features_and_target(data: DataFrame) -> tuple[DataFrame, Series]:
    X = data.drop(columns=[feature.TARGET])
    y = data[feature.TARGET]
    return X, y


def sample_data(
    data: DataFrame, n_samples: int, random_state: int | None = None
) -> DataFrame:
    return data.sample(n=n_samples, random_state=random_state)
