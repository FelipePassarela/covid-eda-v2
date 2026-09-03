from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from covid.tune import TuningSpec


def tuning_spec_from_config(config: DictConfig) -> TuningSpec:
    pipeline = instantiate(config.pipeline, _convert_="all")
    param_distributions = instantiate(config.param_distributions, _convert_="all")

    logger.debug("Pipeline to tune: {}", pipeline)
    logger.debug("Parameter distributions to tune: {}", param_distributions)

    return TuningSpec(
        name=config.name,
        pipeline=pipeline,
        param_distributions=param_distributions,
        n_searches=config.n_searches,
        n_splits=config.n_splits,
        n_fold_repeats=config.n_fold_repeats,
        scoring=config.scoring,
        data_path=config.data_path,
    )
