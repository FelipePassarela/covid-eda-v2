from pathlib import Path

import pandas as pd
from loguru import logger

from ..domain.covid_dataset import CovidDataset
from ..domain.covid_dataset_loader import CovidDatasetLoader
from ..domain.data_cleaning_config import DataCleaningConfig


class CovidDatasetCSVLoader(CovidDatasetLoader):
    def __init__(self, raw_path: Path) -> None:
        self._raw_path = raw_path

    def load_dataset(self, config: DataCleaningConfig) -> CovidDataset:
        logger.info(f"Loading dataset from {self._raw_path}")
        df = pd.read_csv(self._raw_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return CovidDataset.from_dataframe(df, cleaning_config=config)
