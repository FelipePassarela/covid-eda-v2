from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from covid.common.pipeline import build_pipeline_from_config
from covid.tune.search.tuning_spec import TuningSpec


def tuning_spec_from_config(config: DictConfig) -> TuningSpec:
    search = config.search
    pipeline = build_pipeline_from_config(search)
    param_distributions = instantiate(search.param_distributions, _convert_="all")

    logger.debug("Pipeline to tune: {}", pipeline)
    logger.debug("Parameter distributions to tune: {}", param_distributions)

    return TuningSpec(
        pipeline=pipeline,
        param_distributions=param_distributions,
        n_searches=search.n_searches,
        n_splits=search.n_splits,
        n_fold_repeats=search.n_fold_repeats,
        n_jobs=search.get("n_jobs"),
        scoring=search.scoring,
        data_path=config.dataset.path,
    )
