from unittest.mock import Mock, patch

from omegaconf import DictConfig
from optuna_integration import OptunaSearchCV
from sklearn.model_selection import cross_validate

from covid_tuning import tune
from covid_tuning.eval import evaluate_tuning
from covid_tuning.eval.tuning_evaluation_spec_factory import (
    tuning_evaluation_spec_from_config,
)
from covid_tuning.search.tuning_spec_factory import tuning_spec_from_config


def test_tune_applies_configured_parallelism(
    standalone_tuning_config: DictConfig,
) -> None:
    spec = tuning_spec_from_config(standalone_tuning_config)

    with patch(
        "covid_tuning.search.tuning.OptunaSearchCV", wraps=OptunaSearchCV
    ) as optuna_search:
        tune(spec, tracker=Mock())

    optuna_search.assert_called_once()
    assert optuna_search.call_args.kwargs["n_jobs"] == 2


def test_eval_tune_applies_configured_parallelism(
    nested_tuning_config: DictConfig,
) -> None:
    evaluation_spec = tuning_evaluation_spec_from_config(nested_tuning_config)

    with patch(
        "covid_tuning.eval.tuning_evaluation.cross_validate", wraps=cross_validate
    ) as outer_cv:
        evaluate_tuning(evaluation_spec, tracker=Mock())

    outer_cv.assert_called_once()
    assert outer_cv.call_args.args[0].n_jobs == 3
    assert outer_cv.call_args.kwargs["n_jobs"] == 1
