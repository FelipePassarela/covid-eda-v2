from dataclasses import dataclass
from pathlib import Path

from covid.data import DataCleaningConfig
from covid.cross_val import TrainingConfig


@dataclass(frozen=True)
class RunConfig:
    raw_data_path: Path
    cleaning: DataCleaningConfig
    train: TrainingConfig
