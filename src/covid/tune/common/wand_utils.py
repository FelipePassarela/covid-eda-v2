import json
from typing import Any

import pandas as pd

import wandb


def make_serializable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, dict):
        return {str(key): make_serializable(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [make_serializable(item) for item in value]

    return repr(value)


def to_wandb_table(dataframe: pd.DataFrame) -> wandb.Table:
    serializable = dataframe.reset_index().copy()
    serializable.drop("index", axis=1, inplace=True)

    for column in serializable.columns:
        serializable[column] = serializable[column].map(make_table_cell)

    return wandb.Table(dataframe=serializable)


def make_table_cell(value: Any) -> Any:
    value = make_serializable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value
