from pathlib import Path

import joblib
import typer
from loguru import logger

from covid import constants
from covid.data import load_and_split_data
from covid.evaluation import EvaluationResult


def main() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "evaluate.log", rotation="5 MB")

    typer.run(evaluate_model)


def evaluate_model(
    model_path: Path, data_path: Path = constants.INTERIM_TEST_DATA_PATH
) -> None:
    X_test, y_test = load_and_split_data(data_path)
    model = joblib.load(model_path)
    result = EvaluationResult.from_model(model, X_test, y_test)

    scores = result.scores_to_dataframe().round(3)
    conf_matrix = result.confusion_matrix_to_dataframe()

    logger.info("Evaluation scores:\n{}", scores.to_string(index=False))
    logger.info("Confusion matrix:\n{}", conf_matrix)


if __name__ == "__main__":
    main()
