from pathlib import Path

import pandas as pd
from pandas import DataFrame

from covid import feature


def load_data(data_path: Path) -> DataFrame:
    return pd.read_csv(data_path, dtype={feature.ID: str})
