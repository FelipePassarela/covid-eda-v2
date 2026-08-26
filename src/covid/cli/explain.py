from pathlib import Path

import typer

from covid import constants
from covid.explain import ExplainingSpec, WandbExplainingTracker
from covid.explain import explain as run_explain


def main() -> None:
    typer.run(explain)


def explain(
    pipeline_path: Path,
    train_path: Path = constants.INTERIM_TRAIN_DATA_PATH,
    test_path: Path = constants.INTERIM_TEST_DATA_PATH,
) -> None:
    tracker = WandbExplainingTracker(run_name=pipeline_path.stem)
    with tracker:
        spec = ExplainingSpec(
            pipeline_path=pipeline_path, train_path=train_path, test_path=test_path
        )
        run_explain(spec=spec, tracker=tracker)


if __name__ == "__main__":
    main()
