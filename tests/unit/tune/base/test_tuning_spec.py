from pathlib import Path

import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from optuna.distributions import CategoricalDistribution
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

from covid.tune.search.tuning_spec import TuningSpec


def _write_data_file(tmp_path: Path) -> Path:
    data_path = tmp_path / "train.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(data_path, index=False)
    return data_path


def test_tuning_spec_accepts_optuna_distributions(tmp_path: Path) -> None:
    distribution = CategoricalDistribution(["most_frequent", "stratified"])
    spec = TuningSpec(
        pipeline=Pipeline([("clf", DummyClassifier())]),
        param_distributions={"clf__strategy": distribution},
        n_searches=2,
        data_path=_write_data_file(tmp_path),
    )

    assert spec.param_distributions == {"clf__strategy": distribution}
    assert spec.model_dump()["param_distributions"] == {
        "clf__strategy": repr(distribution)
    }


@pytest.mark.parametrize(
    "distributions",
    [
        {},
        {"clf__strategy": ["most_frequent", "stratified"]},
        [{"clf__strategy": CategoricalDistribution(["most_frequent"])}],
        {"clf__unknown": CategoricalDistribution([1])},
    ],
)
def test_tuning_spec_rejects_invalid_search_spaces(
    tmp_path: Path, distributions
) -> None:
    with pytest.raises(ValidationError):
        # noinspection bad-argument-type
        TuningSpec(
            pipeline=Pipeline([("clf", DummyClassifier())]),
            param_distributions=distributions,
            n_searches=2,
            data_path=_write_data_file(tmp_path),
        )
