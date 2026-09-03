from pathlib import Path

import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from covid.tune import TuningSpec


@pytest.fixture()
def spec(tmp_path: Path, classification_df: pd.DataFrame) -> TuningSpec:
    data_path = tmp_path / "data.csv"
    classification_df.to_csv(data_path, index=False)

    classifier = DummyClassifier(strategy="most_frequent")

    return TuningSpec(
        name="test_tune",
        data_path=data_path,
        pipeline=Pipeline(steps=[("clf", classifier)]),
        param_distributions={"clf__strategy": ["most_frequent", "stratified"]},
        n_searches=2,
        n_fold_repeats=1,
        scoring=["accuracy"],
    )
