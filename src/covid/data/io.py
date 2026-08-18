from pathlib import Path

import pandas as pd
from loguru import logger

from covid import feature
from covid.data import split_features_and_target


def load_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path, dtype={feature.ID: str})
    logger.info("Loaded data from {} with shape {}", data_path, data.shape)
    return data


def load_and_split_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    train_data = load_data(data_path)
    X, y = split_features_and_target(train_data)
    return X, y
