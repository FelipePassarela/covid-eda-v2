from pathlib import Path
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_module
from omegaconf import DictConfig
from optuna_integration import OptunaSearchCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

from covid_common.constants import RANDOM_STATE
from covid_tuning import tune
from covid_tuning.eval import evaluate_tuning
from covid_tuning.eval.tracking import WandBTuningEvaluationTracker
from covid_tuning.eval.tuning_evaluation_spec_factory import (
    tuning_evaluation_spec_from_config,
)
from covid_tuning.search.tuning_spec_factory import tuning_spec_from_config


def test_tune_uses_configured_cv_and_parallelism(
    standalone_tuning_config: DictConfig,
) -> None:
    spec = tuning_spec_from_config(standalone_tuning_config)

    with patch(
        "covid_tuning.search.tuning.OptunaSearchCV", wraps=OptunaSearchCV
    ) as optuna_search:
        tune(spec, tracker=Mock())

    optuna_search.assert_called_once()
    search_options = optuna_search.call_args.kwargs
    assert search_options["n_jobs"] == 2
    assert search_options["cv"] is spec.cv
    assert spec.cv.get_n_splits() == 6


def test_eval_tune_uses_configured_cv_and_parallelism(
    nested_tuning_config: DictConfig,
) -> None:
    assert "n_fold_repeats" not in nested_tuning_config.search
    evaluation_spec = tuning_evaluation_spec_from_config(nested_tuning_config)

    with patch(
        "covid_tuning.eval.tuning_evaluation.cross_validate", wraps=cross_validate
    ) as validate:
        result = evaluate_tuning(evaluation_spec, tracker=Mock())

    validate.assert_called_once()
    inner_search = validate.call_args.args[0]
    outer_options = validate.call_args.kwargs
    assert inner_search.n_jobs == 3
    assert inner_search.cv is evaluation_spec.inner.cv
    assert inner_search.cv.get_n_splits() == 2
    assert outer_options["n_jobs"] == 1
    assert outer_options["cv"] is evaluation_spec.outer_cv
    assert len(result.cv_report) == 4


def test_factories_preserve_default_cross_validation(data_path: Path) -> None:
    with initialize_config_module(config_module="covid_tuning.conf", version_base=None):
        standalone_config = compose(config_name="tune")
        evaluation_config = compose(config_name="tuning-evaluation")
    standalone_config.dataset.path = str(data_path)
    evaluation_config.dataset.path = str(data_path)

    standalone = tuning_spec_from_config(standalone_config)
    evaluation = tuning_evaluation_spec_from_config(evaluation_config)

    assert isinstance(standalone.cv, RepeatedStratifiedKFold)
    assert standalone.cv.get_n_splits() == 25
    assert getattr(standalone.cv, "random_state", None) == RANDOM_STATE

    assert isinstance(evaluation.inner.cv, StratifiedKFold)
    assert evaluation.inner.cv.get_n_splits() == 5
    assert getattr(evaluation.inner.cv, "shuffle", False)
    assert getattr(evaluation.inner.cv, "random_state", None) == RANDOM_STATE

    assert isinstance(evaluation.outer_cv, RepeatedStratifiedKFold)
    assert evaluation.outer_cv.get_n_splits() == 30
    assert getattr(evaluation.outer_cv, "random_state", None) == RANDOM_STATE


def test_evaluation_tracker_records_effective_cross_validation(
    nested_tuning_config: DictConfig,
) -> None:
    spec = tuning_evaluation_spec_from_config(nested_tuning_config)
    run = Mock(config={})

    with (
        patch("wandb.init", return_value=run),
        WandBTuningEvaluationTracker(config={}) as tracker,
    ):
        tracker.track_spec(spec)

    serialized_inner_cv = spec.inner.as_serializable()["cv"]
    assert serialized_inner_cv == repr(spec.inner.cv)
    assert run.config["inner/search/cv"] == serialized_inner_cv
    assert run.config["inner/search/n_splits"] == 2
    assert run.config["inner/search/n_fold_repeats"] == 1
    assert run.config["outer/cv"] == repr(spec.outer_cv)
    assert run.config["outer/n_splits"] == 2
    assert run.config["outer/n_repeats"] == 2
