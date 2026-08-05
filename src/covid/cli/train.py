from pathlib import Path
from typing import Any, cast

import hydra
import joblib
from hydra.utils import instantiate
from imblearn.pipeline import Pipeline
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import TunedThresholdClassifierCV

import wandb
from covid import constants
from covid.data import load_data
from covid.training import WAndBTrainingTracker, train


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(config: DictConfig) -> None:
    configure_logging()
    run_train(config)


def configure_logging() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "train.log", rotation="5 MB")


def run_train(config: DictConfig) -> None:
    logger.info("Training configuration: {}", OmegaConf.to_yaml(config))

    with wandb.init(
        project="covid",
        name=config.name,
        job_type="train",
        config=prepare_config_for_wandb(config),
    ) as run:
        model: Pipeline = instantiate(config.pipeline)
        model.set_output(transform="pandas")

        train_data_path = Path(config.train_data_path)
        train_data = load_data(train_data_path)

        tracker = WAndBTrainingTracker(run)
        trained_model: Pipeline | TunedThresholdClassifierCV = train(
            model=model,
            train_data=train_data,
            tune_threshold=config.tune_threshold,
            tuning_scoring=config.tuning_scoring,
            tracker=tracker,
        )

        save_model(trained_model, config.output_path)
        tracker.track_model(Path(config.output_path))


def prepare_config_for_wandb(config: DictConfig) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )


def save_model(model: Pipeline | TunedThresholdClassifierCV, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.success("Trained model saved to {}", output_path)


if __name__ == "__main__":
    main()
