from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .covid_dataset import CovidDataset, DataCleaningConfig


class CovidDatasetTracker(ABC):
    @abstractmethod
    def log_cleaning_config(self, cleaning_config: DataCleaningConfig) -> None:
        pass

    @abstractmethod
    def log_dataset_metrics(self, dataset: CovidDataset) -> None:
        pass

    @abstractmethod
    def log_data_path(self, path: Path) -> None:
        pass

    def compute_metrics(self, dataset: CovidDataset) -> dict[str, Any]:
        metrics = {
            "num_rows": dataset.X.shape[0],
            "num_columns": dataset.X.shape[1],
            "num_missing_values": dataset.X.isnull().sum().sum().item(),
            "target_distribution": dataset.y.value_counts().to_dict(),
        }
        return metrics
