from pathlib import Path

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from covid.tune.base.spec import TuningSpec


def _write_data_file(tmp_path: Path) -> Path:
    data_path = tmp_path / "train.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(data_path, index=False)
    return data_path


def test_tuning_spec_accepts_dict_param_distributions(tmp_path: Path) -> None:
    spec = TuningSpec(
        pipeline=Pipeline([("clf", DummyClassifier())]),
        param_distributions={"clf__strategy": ["most_frequent", "stratified"]},
        n_searches=2,
        data_path=_write_data_file(tmp_path),
    )

    assert spec.param_distributions == {
        "clf__strategy": ["most_frequent", "stratified"]
    }
    assert spec.model_dump()["param_distributions"] == {
        "clf__strategy": "['most_frequent', 'stratified']"
    }


def test_tuning_spec_accepts_list_of_param_distributions(tmp_path: Path) -> None:
    spec = TuningSpec(
        pipeline=Pipeline([("clf", DummyClassifier())]),
        param_distributions=[
            {"clf__strategy": ["most_frequent", "stratified"]},
            {"clf__random_state": [0, 1]},
        ],
        n_searches=2,
        data_path=_write_data_file(tmp_path),
    )

    assert spec.param_distributions == [
        {"clf__strategy": ["most_frequent", "stratified"]},
        {"clf__random_state": [0, 1]},
    ]
    assert spec.model_dump()["param_distributions"] == [
        {"clf__strategy": "['most_frequent', 'stratified']"},
        {"clf__random_state": "[0, 1]"},
    ]
