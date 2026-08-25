from pathlib import Path

import pandas as pd
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.model_selection import RepeatedStratifiedKFold, TunedThresholdClassifierCV

from covid import constants
from covid.data import load_and_split_data
from covid.pipeline import save_pipeline
from covid.train.spec import TrainingSpec
from covid.train.tracker import TrainingTracker


def fit(spec: TrainingSpec) -> None:
    model = spec.model
    logger.info("Training model: {}", model)

    X, y = _load_data(spec.data_path, tracker=spec.tracker)
    trained_model = model.fit(X, y)
    _save_model(trained_model, output_path=spec.model_output_path, tracker=spec.tracker)


def _load_data(
    data_path: Path, tracker: TrainingTracker
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_and_split_data(data_path)
    tracker.track_data(X, y)
    return X, y


def _save_model(
    model: Pipeline | TunedThresholdClassifierCV,
    output_path: Path,
    tracker: TrainingTracker,
) -> None:
    save_pipeline(model, output_path)
    tracker.track_model(output_path)


def tune_threshold(spec: TrainingSpec, scoring: str) -> None:
    logger.info("Training model with threshold tuning: {}", spec.model)
    logger.info("Threshold tuning scoring: {}", scoring)

    X, y = _load_data(spec.data_path, tracker=spec.tracker)
    tuned_model = _tune_threshold(spec.model, X, y, scoring=scoring)

    _report_threshold_tuning_scores(tuned_model, scoring=scoring, tracker=spec.tracker)
    _save_model(tuned_model, output_path=spec.model_output_path, tracker=spec.tracker)


def _tune_threshold(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, scoring: str
) -> TunedThresholdClassifierCV:
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=constants.RANDOM_STATE
    )
    model = TunedThresholdClassifierCV(
        estimator=model, scoring=scoring, cv=cv, n_jobs=-1, store_cv_results=True
    )
    model.fit(X, y)
    return model


def _report_threshold_tuning_scores(
    model: TunedThresholdClassifierCV, scoring: str, tracker: TrainingTracker
) -> None:
    tracker.track_threshold_tuning(model, scoring)

    scoring_formatted = scoring.replace("_", " ")
    logger.info("Best cross-validated {}: {:.4f}", scoring_formatted, model.best_score_)
    logger.info("Selected decision threshold: {:.4f}", model.best_threshold_)
