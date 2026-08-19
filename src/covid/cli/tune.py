from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from covid import constants
from covid.cli.logging import configure_logging, log_config
from covid.data import load_and_split_data
from covid.tuning import RandomizedSearchSpec, search_hyperparameters
from covid.tuning.tracking import WAndBTuningTracker


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def main(config: DictConfig) -> None:
    configure_logging(constants.LOGS_DIR / "tune.log")

    X_train, y_train = load_and_split_data(Path(config.train_data_path))
    search_spec = create_search_spec(config)
    run_tuning(X_train, y_train, search_spec, config)


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
    log_config(config)

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
