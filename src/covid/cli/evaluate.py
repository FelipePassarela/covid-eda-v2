from pathlib import Path

import typer
from loguru import logger

from covid import constants
from covid.evaluation import evaluate as run_evaluation


def main() -> None:
    configure_logging()
    typer.run(evaluate)


def configure_logging() -> None:
    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(constants.LOGS_DIR / "evaluate.log", rotation="5 MB")


def evaluate(
    model_path: Path, data_path: Path = constants.INTERIM_TEST_DATA_PATH
) -> None:
    run_evaluation(model_path, data_path)


if __name__ == "__main__":
    main()
