from dataclasses import dataclass
from pathlib import Path

from covid.cross_val import TrainingConfig
from covid.data import DataCleaningConfig


@dataclass(frozen=True)
class RunConfig:
    raw_data_path: Path
    cleaning: DataCleaningConfig
    training: TrainingConfig
