from omegaconf import DictConfig

from covid.tune.eval.tuning_evaluation_spec import TuningEvaluationSpec
from covid.tune.search.tuning_spec_factory import tuning_spec_from_config


def tuning_evaluation_spec_from_config(config: DictConfig) -> TuningEvaluationSpec:
    return TuningEvaluationSpec(
        inner=tuning_spec_from_config(config),
        n_outer_splits=config.outer_cv.n_splits,
        n_outer_repeats=config.outer_cv.n_repeats,
        n_outer_jobs=config.outer_cv.get("n_jobs", -1),
    )
