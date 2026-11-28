from pathlib import Path
from unittest.mock import MagicMock

import pytest
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from covid_model.train import TrainingSpec


@pytest.fixture()
def spec(tmp_path: Path, data_path: Path) -> TrainingSpec:
    classifier = LogisticRegression(solver="saga", random_state=42)

    return TrainingSpec(
        model=Pipeline(steps=[("clf", classifier)]),
        data_path=data_path,
        model_output_path=tmp_path / "model.pkl",
        tracker=MagicMock(),
    )
