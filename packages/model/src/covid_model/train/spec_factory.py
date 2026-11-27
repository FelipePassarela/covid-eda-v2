from pathlib import Path

from omegaconf import DictConfig

from covid_common.pipeline import build_pipeline_from_config
from covid_model.train import TrainingSpec
from covid_model.train.tracker import TrainingTracker


def training_spec_from_config(
    config: DictConfig, tracker: TrainingTracker
) -> TrainingSpec:
    return TrainingSpec(
        model=build_pipeline_from_config(config),
        data_path=Path(config.train_data_path),
        model_output_path=Path(config.output_path),
        tracker=tracker,
    )
