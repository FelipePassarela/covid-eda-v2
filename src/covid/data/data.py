from loguru import logger
from pandas import DataFrame, Series

from covid import feature


def split_features_and_target(data: DataFrame) -> tuple[DataFrame, Series]:
    X = data.drop(columns=[feature.TARGET])
    y = data[feature.TARGET]
    return X, y


def sample_data(
    data: DataFrame, n_samples: int, random_state: int | None = None
) -> DataFrame:
    data = data.sample(n=n_samples, random_state=random_state)
    logger.info(f"sampled data to shape {data.shape}")
    return data
