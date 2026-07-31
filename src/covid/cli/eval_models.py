from dataclasses import asdict

import pandas as pd
import typer
from loguru import logger

import wandb
from covid import constants
from covid.data import load_data, split_features_and_target
from covid.data.data import sample_data
from covid.tuning import (
    HyperparameterSearchResult,
    create_all_specs,
    create_specs,
    search_hyperparameters,
)
from covid.tuning.tracking import WAndBTracker


def main() -> None:
    logger.add(constants.LOGS_DIR / "eval_models.log", rotation="5 MB")
    typer.run(tune_models)


def tune_models(
    quick: bool = False,
    search_specs: list[str] | None = None,
    scoring: list[str] | None = None,
) -> None:
    train_data = load_data(data_path=constants.INTERIM_TRAIN_DATA_PATH)
    logger.info(f"loaded data with shape {train_data.shape}")

    if quick:
        train_data = sample_data(train_data, n_samples=15)
        logger.info(f"sampled data to shape {train_data.shape}")

    X_train, y_train = split_features_and_target(train_data)

    if scoring is None:
        scoring = ["balanced_accuracy", "recall", "f1", "precision", "roc_auc"]

    specs = (
        create_specs(search_specs, scoring, quick)
        if search_specs
        else create_all_specs(scoring, quick)
    )
    spec_names = [spec.name for spec in specs]
    logger.info(f"Running searches for {spec_names}")

    for spec in specs:
        with wandb.init(project="covid", config=asdict(spec)) as run:
            tracker = WAndBTracker(run)
            result = search_hyperparameters(X_train, y_train, spec=spec)
            tracker.track_result(result)


def log_search_result(result: HyperparameterSearchResult) -> None:
    logger.info(f"Best parameters: {result.best_params}")
    logger.info(f"Best score: {result.best_score:.3f}")
    with pd.option_context("display.max_columns", None):
        logger.info(f"Report:\n{result.report.round(2)}")


if __name__ == "__main__":
    main()
