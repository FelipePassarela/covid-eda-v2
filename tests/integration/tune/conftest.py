from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_module
from imblearn.pipeline import Pipeline
from omegaconf import DictConfig
from optuna.distributions import CategoricalDistribution
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from covid_common.constants import RANDOM_STATE
from covid_tuning import TuningSpec


@pytest.fixture()
def spec(tmp_path: Path, classification_df: pd.DataFrame) -> TuningSpec:
    data_path = tmp_path / "data.csv"
    classification_df.to_csv(data_path, index=False)

    classifier = DummyClassifier(strategy="most_frequent")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_STATE)

    return TuningSpec(
        data_path=data_path,
        pipeline=Pipeline(steps=[("clf", classifier)]),
        param_distributions={
            "clf__strategy": CategoricalDistribution(["most_frequent", "stratified"])
        },
        n_searches=2,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy"],
    )


@pytest.fixture()
def standalone_tuning_config(data_path: Path) -> DictConfig:
    return _compose_small_config(
        "tune", data_path, ["search.n_jobs=2", "search.n_fold_repeats=3"]
    )


@pytest.fixture()
def nested_tuning_config(data_path: Path) -> DictConfig:
    return _compose_small_config(
        "tuning-evaluation",
        data_path,
        [
            "search.n_jobs=3",
            "outer_cv.n_jobs=1",
            "outer_cv.n_splits=2",
            "outer_cv.n_repeats=2",
        ],
    )


def _compose_small_config(
    config_name: str, data_path: Path, overrides: list[str]
) -> DictConfig:
    with initialize_config_module(config_module="covid_tuning.conf", version_base=None):
        config = compose(
            config_name=config_name,
            overrides=[
                "search.n_searches=1",
                "search.n_splits=2",
                "search.scoring=[balanced_accuracy]",
                "search.pipeline.preprocessor.order=[]",
                *overrides,
            ],
        )
    config.dataset.path = str(data_path)
    return config
