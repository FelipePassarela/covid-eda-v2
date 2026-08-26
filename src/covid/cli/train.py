from pathlib import Path
from typing import Any, cast

import hydra
from hydra.utils import instantiate
from imblearn.pipeline import Pipeline
from omegaconf import DictConfig, OmegaConf

from covid import paths
from covid.cli.logging import configure_logging, log_config
from covid.experiments.train import (
    TrainingSpec,
    TrainingTracker,
    WAndBTrainingTracker,
    fit,
    tune_threshold,
)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "train.log")
    train(config)


def train(config: DictConfig) -> None:
    log_config(config)

    config_for_wandb = prepare_config_for_wandb(config)
    wandb_tracker = WAndBTrainingTracker(config=config_for_wandb)

    with wandb_tracker as tracker:
        spec = create_train_spec(config, tracker)
        if should_tune_threshold(config):
            tune_threshold(spec, scoring=config.tuning_scoring)
        else:
            fit(spec)


def prepare_config_for_wandb(config: DictConfig) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )


def create_train_spec(config: DictConfig, tracker: TrainingTracker) -> TrainingSpec:
    return TrainingSpec(
        model=instantiate_model(config),
        data_path=Path(config.train_data_path),
        model_output_path=Path(config.output_path),
        tracker=tracker,
    )


def instantiate_model(config: DictConfig) -> Pipeline:
    model: Pipeline = instantiate(config.pipeline)
    model.set_output(transform="pandas")
    return model


def should_tune_threshold(config: DictConfig) -> bool:
    return config.tune_threshold


if __name__ == "__main__":
    main()
