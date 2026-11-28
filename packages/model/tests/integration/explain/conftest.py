from pathlib import Path
from unittest.mock import MagicMock

import pytest

from covid_model.explain import ExplainingSpec, explain
from covid_model.explain.result import ExplainingResult
from covid_model.explain.tracker import ExplainingTracker


@pytest.fixture()
def explaining_spec(pipeline_path: Path, data_path: Path) -> ExplainingSpec:
    return ExplainingSpec(pipeline_path, data_path, max_display=3)


@pytest.fixture()
def explaining_result(
    explaining_spec: ExplainingSpec, existing_figures: set[int]
) -> ExplainingResult:
    tracker = MagicMock(spec=ExplainingTracker)
    explain(explaining_spec, tracker)
    return tracker.track_result.call_args.args[0]
