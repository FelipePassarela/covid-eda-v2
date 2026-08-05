from typing import Any, cast

import hydra
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from covid import constants
from covid.data import load_data, split_features_and_target
from covid.tuning import RandomizedSearchSpec, search_hyperparameters
from covid.tuning.tracking import WAndBTuningTracker


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def main(config: DictConfig) -> None:
    configure_logging()

    X_train, y_train = prepare_data(config)
    search_spec = create_search_spec(config)
    run_tuning(X_train, y_train, search_spec, config)


def configure_logging() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "tune_models.log", rotation="5 MB")


def prepare_data(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    train_data = load_data(config.train_data_path)
    X_train, y_train = split_features_and_target(train_data)
    return X_train, y_train


def create_search_spec(config: DictConfig) -> RandomizedSearchSpec:
    pipeline = instantiate(config.pipeline)
    param_distributions = instantiate(config.param_distributions, _convert_="all")

    logger.debug("Pipeline to tune: {}", pipeline)
    logger.debug("Parameter distributions to tune: {}", param_distributions)

    search_spec = RandomizedSearchSpec(
        name=config.name,
        pipeline=pipeline,
        param_distributions=param_distributions,
        n_searches=config.n_searches,
        n_fold_repeats=config.n_fold_repeats,
        scoring=config.scoring,
    )
    logger.debug("Search specification: {}", search_spec)

    return search_spec


def run_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    spec: RandomizedSearchSpec,
    config: DictConfig,
) -> None:
    logger.info("Tuning configuration: {}", OmegaConf.to_yaml(config))

    with wandb.init(
        project="covid",
        name=spec.name,
        job_type="tuning",
        config=prepare_config_for_wandb(config),
    ) as run:
        result = search_hyperparameters(X_train, y_train, spec=spec)
        WAndBTuningTracker(run).track_search(spec, result)


def prepare_config_for_wandb(config: DictConfig) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )


if __name__ == "__main__":
    main()
