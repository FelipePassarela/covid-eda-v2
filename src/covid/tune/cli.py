from typing import Any, cast

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from covid.common import paths
from covid.common.logging import configure_logging, log_config
from covid.tune import TuningSpec, tune
from covid.tune.tracking import WAndBTuningTracker


@hydra.main(version_base=None, config_path="../conf", config_name="tune")
def main(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "tune.log")

    spec = create_spec(config)
    run_tuning(spec, config)


def create_spec(config: DictConfig) -> TuningSpec:
    pipeline = instantiate(config.pipeline)
    param_distributions = instantiate(config.param_distributions, _convert_="all")

    logger.debug("Pipeline to tune: {}", pipeline)
    logger.debug("Parameter distributions to tune: {}", param_distributions)

    return TuningSpec(
        name=config.name,
        pipeline=pipeline,
        param_distributions=param_distributions,
        n_searches=config.n_searches,
        n_fold_repeats=config.n_fold_repeats,
        scoring=config.scoring,
        data_path=config.data_path,
    )


def run_tuning(spec: TuningSpec, config: DictConfig) -> None:
    log_config(config)

    tracker = WAndBTuningTracker(
        config=prepare_config_for_wandb(config), run_name=spec.name
    )
    with tracker:
        result = tune(spec)
        tracker.track_search(spec, result)


def prepare_config_for_wandb(config: DictConfig) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )


if __name__ == "__main__":
    main()
