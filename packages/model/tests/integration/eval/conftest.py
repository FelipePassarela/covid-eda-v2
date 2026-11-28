from pathlib import Path
from unittest.mock import MagicMock

import pytest

from covid_model.eval import evaluate
from covid_model.eval.tracker import EvaluationTracker


@pytest.fixture()
def evaluated_tracker(pipeline_path: Path, data_path: Path) -> MagicMock:
    tracker = MagicMock(spec=EvaluationTracker)
    evaluate(pipeline_path, data_path, tracker)
    return tracker
