from omegaconf import DictConfig
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from covid_common import constants
from covid_tuning.eval.tuning_evaluation_spec import TuningEvaluationSpec
from covid_tuning.search.tuning_spec_factory import tuning_spec_from_config


def tuning_evaluation_spec_from_config(config: DictConfig) -> TuningEvaluationSpec:
    inner_cv = StratifiedKFold(
        n_splits=config.search.n_splits,
        shuffle=True,
        random_state=constants.RANDOM_STATE,
    )
    outer_settings = config.outer_cv
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_settings.n_splits,
        n_repeats=outer_settings.n_repeats,
        random_state=constants.RANDOM_STATE,
    )

    return TuningEvaluationSpec(
        inner=tuning_spec_from_config(config, cv=inner_cv),
        outer_cv=outer_cv,
        n_outer_jobs=outer_settings.get("n_jobs", -1),
    )
