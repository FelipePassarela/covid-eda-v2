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
    RandomizedSearchSpec,
    search_hyperparameters,
)
from covid.tuning.tracking import WAndBTracker


def main() -> None:
    typer.run(tune_models)


def tune_models(quick: bool = False) -> None:
    train_data = load_data(data_path=constants.INTERIM_TRAIN_DATA_PATH)
    logger.info(f"loaded data with shape {train_data.shape}")

    if quick:
        train_data = sample_data(train_data, n_samples=15)
        logger.info(f"sampled data to shape {train_data.shape}")

    X_train, y_train = split_features_and_target(train_data)

    scoring = ["balanced_accuracy", "recall", "f1", "precision", "roc_auc"]
    search_specs = RandomizedSearchSpec.create_specs(scoring, quick)

    for spec in search_specs:
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
