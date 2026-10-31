from omegaconf import DictConfig

from covid.tune.base.spec_factory import tuning_spec_from_config
from covid.tune.eval.tuning_evaluation_spec import TuningEvaluationSpec


def tuning_evaluation_spec_from_config(config: DictConfig) -> TuningEvaluationSpec:
    return TuningEvaluationSpec(
        inner=tuning_spec_from_config(config),
        n_outer_splits=config.outer_cv.n_splits,
        n_outer_repeats=config.outer_cv.n_repeats,
    )
