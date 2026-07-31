from typing import Any

import pandas as pd

import wandb
from covid.tuning import HyperparameterSearchResult, RandomizedSearchSpec


class WAndBTracker:
    def __init__(self, run: wandb.Run) -> None:
        self.run = run

    def track_result(self, result: HyperparameterSearchResult) -> None:
        self.run.summary["best_score"] = result.best_score
        self.run.summary["best_params"] = WAndBTracker._make_serializable(
            result.best_params
        )
        self.run.log({"results": WAndBTracker._to_wandb_table(result.report)})

    def track_spec(self, spec: RandomizedSearchSpec) -> None: ...

    @staticmethod
    def _to_wandb_table(dataframe: pd.DataFrame) -> wandb.Table:
        serializable = dataframe.reset_index().copy()

        for column in serializable.columns:
            serializable[column] = serializable[column].map(
                WAndBTracker._make_serializable
            )

        return wandb.Table(dataframe=serializable)

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
