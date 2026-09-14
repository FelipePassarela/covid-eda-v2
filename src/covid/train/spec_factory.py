from pathlib import Path

from omegaconf import DictConfig

from covid.common.pipeline import build_pipeline_from_config
from covid.train import TrainingSpec
from covid.train.tracker import TrainingTracker


def training_spec_from_config(
    config: DictConfig, tracker: TrainingTracker
) -> TrainingSpec:
    return TrainingSpec(
        model=build_pipeline_from_config(config),
        data_path=Path(config.train_data_path),
        model_output_path=Path(config.output_path),
        tracker=tracker,
    )
