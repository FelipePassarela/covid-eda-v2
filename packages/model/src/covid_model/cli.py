import sys
from pathlib import Path

import hydra
import typer
from omegaconf import DictConfig

from covid_common import paths, prepare_config_for_wandb
from covid_common.logging import configure_logging, log_config
from covid_model.eval import WandBEvaluationTracker, evaluate
from covid_model.explain import ExplainingSpec, WandbExplainingTracker, explain
from covid_model.train import fit, tune_threshold
from covid_model.train.spec_factory import training_spec_from_config
from covid_model.train.wandb_tracker import WAndBTrainingTracker

app = typer.Typer(help="Train, evaluate, and explain models.", no_args_is_help=True)


@app.command(
    "train",
    help="Train and save a model.",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(context: typer.Context) -> None:
    original_argv = sys.argv
    sys.argv = [context.command_path, *context.args]
    try:
        _train()
    finally:
        sys.argv = original_argv


@hydra.main(
    version_base=None, config_path="pkg://covid_model.conf", config_name="train"
)
def _train(config: DictConfig) -> None:
    configure_logging(paths.LOGS_DIR / "train.log")
    log_config(config)

    config_for_wandb = prepare_config_for_wandb(config)
    with WAndBTrainingTracker(config=config_for_wandb) as tracker:
        spec = training_spec_from_config(config, tracker)
        if config.tune_threshold:
            tune_threshold(spec, scoring=config.tuning_scoring)
        else:
            fit(spec)


@app.command("eval", help="Evaluate a trained model.")
def _evaluate_model(
    model_path: Path, data_path: Path = paths.INTERIM_TEST_DATA_PATH
) -> None:
    configure_logging(paths.LOGS_DIR / "evaluate.log")
    tracker = WandBEvaluationTracker(
        model_name=model_path.name,
        config={"model_path": model_path, "data_path": data_path},
    )
    with tracker:
        evaluate(model_path, data_path, tracker=tracker)


@app.command("explain", help="Explain model predictions with SHAP.")
def _explain_model(pipeline_path: Path, data_path: Path = paths.RAW_DATA_PATH) -> None:
    tracker = WandbExplainingTracker(run_name=pipeline_path.stem)
    with tracker:
        spec = ExplainingSpec(pipeline_path=pipeline_path, data_path=data_path)
        explain(spec=spec, tracker=tracker)
