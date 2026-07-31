from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import RepeatedStratifiedKFold, TunedThresholdClassifierCV

from covid import constants
from covid.data import load_data, split_features_and_target
from covid.pipeline import create_default_pipeline


def main() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "train.log", rotation="5 MB")

    typer.run(train)


def train(
    train_data: Path = constants.INTERIM_TRAIN_DATA_PATH,
    output_path: Path = constants.MODELS_DIR / "final_model.joblib",
    scoring: str = "balanced_accuracy",
) -> None:
    train_data = load_data(train_data)
    X_train, y_train = split_features_and_target(train_data)

    model = train_model(X_train, y_train, scoring)

    report_scores(model, scoring)
    save_model(model, output_path)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series[Any], scoring: str
) -> TunedThresholdClassifierCV:
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=constants.RANDOM_STATE
    )
    model = TunedThresholdClassifierCV(
        estimator=create_default_pipeline(),
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        store_cv_results=True,
    )
    model.fit(X_train, y_train)
    return model


def report_scores(model: TunedThresholdClassifierCV, scoring: str) -> None:
    scoring_formatted = scoring.replace("_", " ")
    logger.info("Best cross-validated {}: {:.4f}", scoring_formatted, model.best_score_)
    logger.info("Selected decision threshold: {:.4f}", model.best_threshold_)


def save_model(model: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.success("Trained model saved to {}", output_path)


if __name__ == "__main__":
    main()
