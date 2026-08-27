import pandas as pd
from loguru import logger

from covid.common import feature


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = data.drop(columns=[feature.TARGET])
    y = data[feature.TARGET]
    return X, y


def sample_data(
    data: pd.DataFrame, n_samples: int, random_state: int | None = None
) -> pd.DataFrame:
    data = data.sample(n=n_samples, random_state=random_state)
    logger.info(f"sampled data to shape {data.shape}")
    return data
