from pathlib import Path

import pandas as pd
from loguru import logger
from pandas import DataFrame

from covid import feature


def load_data(data_path: Path) -> DataFrame:
    data = pd.read_csv(data_path, dtype={feature.ID: str})
    logger.info("Loaded data from {} with shape {}", data_path, data.shape)
    return data
