from pathlib import Path

import joblib
import typer
from loguru import logger

from covid import constants
from covid.data import load_data, split_features_and_target
from covid.evaluation import EvaluationResult


def main() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "evaluate.log", rotation="5 MB")

    typer.run(evaluate_model)


def evaluate_model(model_path: Path) -> None:
    test_data = load_data(constants.INTERIM_TEST_DATA_PATH)
    X_test, y_test = split_features_and_target(test_data)

    model = joblib.load(model_path)
    result = EvaluationResult.from_model(model, X_test, y_test)

    scores = result.scores_to_dataframe().round(3)
    conf_matrix = result.confusion_matrix_to_dataframe()

    logger.info("Evaluation scores:\n{}", scores.to_string(index=False))
    logger.info("Confusion matrix:\n{}", conf_matrix)


if __name__ == "__main__":
    main()
