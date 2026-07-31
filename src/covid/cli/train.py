from pathlib import Path

import joblib
import typer
from loguru import logger
from sklearn import clone
from sklearn.model_selection import RepeatedStratifiedKFold, TunedThresholdClassifierCV

from covid import constants
from covid.data import load_data, split_features_and_target
from covid.pipeline import DEFAULT_PIPELINE


def main() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "train.log", rotation="5 MB")

    typer.run(train_model)


def train_model(
    train_data: Path = constants.INTERIM_TRAIN_DATA_PATH,
    output_path: Path = constants.MODELS_DIR / "final_model.joblib",
    scoring: str = "balanced_accuracy",
) -> None:
    train_data = load_data(train_data)
    X_train, y_train = split_features_and_target(train_data)

    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=constants.RANDOM_STATE
    )
    model = TunedThresholdClassifierCV(
        estimator=clone(DEFAULT_PIPELINE),
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        store_cv_results=True,
    )
    model.fit(X_train, y_train)

    scoring_formatted = scoring.replace("_", " ")
    logger.info("Best cross-validated {}: {:.4f}", scoring_formatted, model.best_score_)
    logger.info("Selected decision threshold: {:.4f}", model.best_threshold_)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    logger.success("Trained model saved to {}", output_path)


if __name__ == "__main__":
    main()
