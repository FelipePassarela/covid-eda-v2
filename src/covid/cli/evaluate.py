from pathlib import Path

import typer

from covid import constants
from covid.cli.logging import configure_logging
from covid.evaluation import WandBEvaluationTracker
from covid.evaluation import evaluate as run_evaluation


def main() -> None:
    configure_logging(constants.LOGS_DIR / "evaluate.log")
    typer.run(evaluate)


def evaluate(
    model_path: Path, data_path: Path = constants.INTERIM_TEST_DATA_PATH
) -> None:
    tracker = WandBEvaluationTracker(
        model_name=model_path.name,
        config={"model_path": model_path, "data_path": data_path},
    )
    with tracker:
        run_evaluation(model_path, data_path, tracker=tracker)


if __name__ == "__main__":
    main()
