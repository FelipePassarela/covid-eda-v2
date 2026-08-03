from pathlib import Path
from typing import Any

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
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "train.log", rotation="5 MB")
    logger.info("Training configuration: {}", OmegaConf.to_yaml(config))

    train_data_path = Path(config.train_data_path)
    output_path = Path(config.output_path)

    config_to_report = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    with wandb.init(
        project="covid",
        name=output_path.name,
        job_type="train",
        config=config_to_report,
    ) as run:
        model: Pipeline = instantiate(config.pipeline)
        model.set_output(transform="pandas")

        train_data = load_data(train_data_path)

        tracker = WAndBTrainingTracker(run)
        model: Pipeline | TunedThresholdClassifierCV = train(
            model=model,
            train_data=train_data,
            tune_threshold=config.tune_threshold,
            tuning_scoring=config.tuning_scoring,
            tracker=tracker,
        )
        save_model(model, output_path)
        tracker.track_model(output_path)


def save_model(model: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.success("Trained model saved to {}", output_path)


if __name__ == "__main__":
    main()
