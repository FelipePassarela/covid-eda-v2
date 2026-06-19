from dataclasses import dataclass

from covid.data import DatasetConfig
from covid.training import TrainingConfig


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    train: TrainingConfig
