from pathlib import Path

import pandas as pd
import pytest

from covid_common.feature import ID, TARGET


@pytest.fixture()
def source_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID: [f"{number:04d}" for number in range(100)],
            "measurement": range(100),
            TARGET: [0] * 80 + [1] * 20,
        }
    )


@pytest.fixture()
def data_path(tmp_path: Path, source_data: pd.DataFrame) -> Path:
    path = tmp_path / "source.csv"
    source_data.to_csv(path, index=False)
    return path
