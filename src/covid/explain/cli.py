from pathlib import Path

import typer

from covid.common import paths
from covid.explain import ExplainingSpec, WandbExplainingTracker
from covid.explain import explain as run_explain


def main() -> None:
    typer.run(explain)


def explain(pipeline_path: Path, data_path: Path = paths.RAW_DATA_PATH) -> None:
    tracker = WandbExplainingTracker(run_name=pipeline_path.stem)
    with tracker:
        spec = ExplainingSpec(pipeline_path=pipeline_path, data_path=data_path)
        run_explain(spec=spec, tracker=tracker)


if __name__ == "__main__":
    main()
