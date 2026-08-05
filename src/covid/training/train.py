import pandas as pd
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.model_selection import RepeatedStratifiedKFold, TunedThresholdClassifierCV

from covid import constants
from covid.data import split_features_and_target
from covid.training.tracker import TrainingTracker


def train(
    model: Pipeline,
    train_data: pd.DataFrame,
    tune_threshold: bool,
    tuning_scoring: str,
    tracker: TrainingTracker,
) -> Pipeline | TunedThresholdClassifierCV:
    logger.info("Training model: {}", model)
    logger.info("Threshold tuning: {}", tune_threshold)

    X_train, y_train = split_features_and_target(train_data)

    tracker.track_data(X_train, y_train)

    if tune_threshold:
        model = _train_with_threshold_tuning(
            model=model, X_train=X_train, y_train=y_train, scoring=tuning_scoring
        )
        tracker.track_threshold_tuning(model, tuning_scoring)
        _report_threshold_tuning_scores(model, tuning_scoring)
    else:
        model = model.fit(X_train, y_train)

    return model


def _train_with_threshold_tuning(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, scoring: str
) -> TunedThresholdClassifierCV:
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=constants.RANDOM_STATE
    )
    model = TunedThresholdClassifierCV(
        estimator=model, scoring=scoring, cv=cv, n_jobs=-1, store_cv_results=True
    )
    model.fit(X_train, y_train)
    return model


def _report_threshold_tuning_scores(
    model: TunedThresholdClassifierCV, scoring: str
) -> None:
    scoring_formatted = scoring.replace("_", " ")
    logger.info("Best cross-validated {}: {:.4f}", scoring_formatted, model.best_score_)
    logger.info("Selected decision threshold: {:.4f}", model.best_threshold_)
