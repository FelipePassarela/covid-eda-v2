from pathlib import Path

import typer

from covid import paths
from covid.cli.logging import configure_logging
from covid.experiments.eval import WandBEvaluationTracker
from covid.experiments.eval import evaluate as run_evaluation


def main() -> None:
    configure_logging(paths.LOGS_DIR / "evaluate.log")
    typer.run(evaluate)


def evaluate(model_path: Path, data_path: Path = paths.INTERIM_TEST_DATA_PATH) -> None:
    tracker = WandBEvaluationTracker(
        model_name=model_path.name,
        config={"model_path": model_path, "data_path": data_path},
    )
    with tracker:
        run_evaluation(model_path, data_path, tracker=tracker)


if __name__ == "__main__":
    main()
