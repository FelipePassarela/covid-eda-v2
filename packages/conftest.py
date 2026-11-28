from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pandas as pd

os.environ["MPLBACKEND"] = "Agg"


@pytest.fixture()
def existing_figures() -> Iterator[set[int]]:
    from matplotlib import pyplot as plt

    figures = set(plt.get_fignums())
    yield figures
    for number in set(plt.get_fignums()) - figures:
        plt.close(number)


@pytest.fixture()
def classification_df() -> pd.DataFrame:
    import pandas as pd
    from sklearn.datasets import make_classification

    from covid_common.feature import TARGET

    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df[TARGET] = y
    return df


@pytest.fixture()
def data_path(tmp_path: Path, classification_df: pd.DataFrame) -> Path:
    path = tmp_path / "data.csv"
    classification_df.to_csv(path, index=False)
    return path
