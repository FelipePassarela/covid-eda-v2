from abc import ABC, abstractmethod

from .covid_dataset import CovidDataset
from .data_cleaning_config import DataCleaningConfig


class CovidDatasetLoader(ABC):
    @abstractmethod
    def load_dataset(self, config: DataCleaningConfig) -> CovidDataset:
        pass
