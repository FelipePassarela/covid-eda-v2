from dataclasses import asdict
from typing import Any

import pandas as pd
import typer
from loguru import logger

import wandb
from covid import constants
from covid.data import load_data, split_features_and_target
from covid.data.data import sample_data
from covid.tuning import (
    RandomizedSearchSpec,
    create_all_specs,
    create_specs,
    search_hyperparameters,
)
from covid.tuning.tracking import WAndBTracker


def main() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "tune_models.log", rotation="5 MB")

    typer.run(run_model_searches)


def run_model_searches(
    quick: bool = False,
    search_specs: list[str] | None = None,
    scoring: list[str] | None = None,
) -> None:
    train_data = load_data(data_path=constants.INTERIM_TRAIN_DATA_PATH)
    train_data = sample_data(train_data, n_samples=15) if quick else train_data
    X_train, y_train = split_features_and_target(train_data)

    scoring = resolve_scoring(scoring)
    specs = resolve_specs(search_specs, scoring, quick)

    for spec in specs:
        run_search(X_train, y_train, spec, quick)


def resolve_scoring(scoring: list[str] | None) -> list[str]:
    default_scoring = ["balanced_accuracy", "recall", "f1", "precision", "roc_auc"]
    scoring = default_scoring if scoring is None else scoring
    logger.info(f"Using scoring metrics: {scoring}")
    return scoring


def resolve_specs(
    search_specs: list[str] | None, scoring: list[str], quick: bool
) -> list[RandomizedSearchSpec]:
    specs = (
        create_specs(search_specs, scoring, quick)
        if search_specs
        else create_all_specs(scoring, quick)
    )
    spec_names = [spec.name for spec in specs]
    logger.info(f"Running searches for {spec_names}")
    return specs


def run_search(
    X_train: pd.DataFrame, y_train: pd.Series, spec: RandomizedSearchSpec, quick: bool
) -> None:
    with wandb.init(
        project="covid",
        name=spec.name,
        group="quick" if quick else "full",
        job_type="hyperparameter-search",
        config=spec.as_serializable(),
    ) as run:
        tracker = WAndBTracker(run)
        result = search_hyperparameters(X_train, y_train, spec=spec)
        tracker.track_search(spec, result)


if __name__ == "__main__":
    main()
