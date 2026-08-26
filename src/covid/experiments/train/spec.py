from dataclasses import dataclass
from pathlib import Path

from imblearn.pipeline import Pipeline

from covid.experiments.train.tracker import TrainingTracker


@dataclass(frozen=True)
class TrainingSpec:
    model: Pipeline
    data_path: Path
    model_output_path: Path
    tracker: TrainingTracker
