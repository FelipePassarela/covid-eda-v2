import sys
from collections.abc import Callable

import hydra
import typer
from omegaconf import DictConfig

from covid_common import paths, prepare_config_for_wandb
from covid_common.logging import configure_logging, log_config
from covid_tuning.eval import evaluate_tuning
from covid_tuning.eval.tracking import WandBTuningEvaluationTracker
from covid_tuning.eval.tuning_evaluation_spec_factory import (
    tuning_evaluation_spec_from_config,
)
from covid_tuning.eval.visualization.boxplot import plot_boxplot
from covid_tuning.search import tune
from covid_tuning.search.tracking import WAndBTuningTracker
from covid_tuning.search.tuning_spec_factory import tuning_spec_from_config

app = typer.Typer(
    help="Search, evaluate, and compare tuning runs.", no_args_is_help=True
)
_HYDRA_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}


@app.command(
    "tune",
    help="Search hyperparameters.",
    add_help_option=False,
    context_settings=_HYDRA_CONTEXT,
)
def tune_command(context: typer.Context) -> None:
    _run_hydra(context, _tune)


@app.command(
    "eval-tune",
    help="Evaluate tuning with nested CV.",
    add_help_option=False,
    context_settings=_HYDRA_CONTEXT,
)
def eval_tune_command(context: typer.Context) -> None:
    _run_hydra(context, _eval_tune)


app.command("boxplot", help="Plot metrics from cross-validation reports.")(plot_boxplot)


def _run_hydra(context: typer.Context, command: Callable[[], None]) -> None:
    original_argv = sys.argv
    sys.argv = [context.command_path, *context.args]
    try:
        command()
    finally:
        sys.argv = original_argv


@hydra.main(
    version_base=None, config_path="pkg://covid_tuning.conf", config_name="tune"
)
def _tune(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "tune.log")
    spec = tuning_spec_from_config(config)
    log_config(config)

    tracker = WAndBTuningTracker(config=prepare_config_for_wandb(config))
    with tracker:
        tune(spec, tracker)


@hydra.main(
    version_base=None,
    config_path="pkg://covid_tuning.conf",
    config_name="tuning-evaluation",
)
def _eval_tune(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "tuning-evaluation.log")
    spec = tuning_evaluation_spec_from_config(config)

    tracker = WandBTuningEvaluationTracker(
        config=prepare_config_for_wandb(config), run_name=config.name
    )
    with tracker:
        evaluate_tuning(spec, tracker)
