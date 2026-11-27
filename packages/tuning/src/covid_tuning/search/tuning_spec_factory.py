from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import RepeatedStratifiedKFold

from covid_common import constants
from covid_common.pipeline import build_pipeline_from_config
from covid_tuning.common.cross_validation import TuningCV
from covid_tuning.search.tuning_spec import TuningSpec


def tuning_spec_from_config(
    config: DictConfig, *, cv: TuningCV | None = None
) -> TuningSpec:
    search = config.search
    cv = _create_standalone_cv(search) if cv is None else cv

    pipeline = build_pipeline_from_config(search)
    param_distributions = instantiate(search.param_distributions, _convert_="all")

    logger.debug("Pipeline to tune: {}", pipeline)
    logger.debug("Parameter distributions to tune: {}", param_distributions)

    return TuningSpec(
        pipeline=pipeline,
        param_distributions=param_distributions,
        n_searches=search.n_searches,
        cv=cv,
        n_jobs=search.get("n_jobs"),
        scoring=search.scoring,
        data_path=config.dataset.path,
    )


def _create_standalone_cv(search: DictConfig) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=search.n_splits,
        n_repeats=search.n_fold_repeats,
        random_state=constants.RANDOM_STATE,
    )
