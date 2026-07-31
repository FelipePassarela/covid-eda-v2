import json
from typing import Any

import pandas as pd

import wandb
from covid.tuning import HyperparameterSearchResult, RandomizedSearchSpec


class WAndBTracker:
    def __init__(self, run: wandb.Run) -> None:
        self.run = run

    def track_search(
        self, spec: RandomizedSearchSpec, result: HyperparameterSearchResult
    ) -> None:
        self.run.summary["best_score"] = result.best_score
        self.run.summary["best_params"] = WAndBTracker._make_serializable(
            result.best_params
        )
        self.run.log(
            {"results": WAndBTracker._to_wandb_table(result.report, spec.name)}
        )

    def track_spec(self, spec: RandomizedSearchSpec) -> None: ...

    @classmethod
    def _to_wandb_table(cls, dataframe: pd.DataFrame, spec_name: str) -> wandb.Table:
        serializable = dataframe.reset_index().copy()
        serializable.drop("index", axis=1, inplace=True)
        serializable["name"] = spec_name

        for column in serializable.columns:
            serializable[column] = serializable[column].map(cls._make_table_cell)

        return wandb.Table(dataframe=serializable)

    @classmethod
    def _make_table_cell(cls, value: Any) -> Any:
        value = cls._make_serializable(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return value

    @staticmethod
    def _make_serializable(value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value

        if isinstance(value, dict):
            return {
                str(key): WAndBTracker._make_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, list | tuple):
            return [WAndBTracker._make_serializable(item) for item in value]

        return repr(value)
