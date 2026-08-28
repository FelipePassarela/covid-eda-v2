import json
from typing import Any, Self

import pandas as pd

import wandb
from covid.tune import TuningResult, TuningSpec


class WAndBTuningTracker:
    def __init__(self, config: dict[str, Any], run_name: str | None = None) -> None:
        self._run_name = run_name
        self._run: wandb.Run | None = None
        self._config = config

    def __enter__(self) -> Self:
        self._run: wandb.Run = wandb.init(
            project="covid",
            name=self._run_name,
            job_type="tuning",
            config=self._config,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._run:
            self._run.finish()

    def track_search(self, spec: TuningSpec, result: TuningResult) -> None:
        if not self._run:
            raise RuntimeError("This class must be used as a context manager.")

        self._run.summary["best_score"] = result.best_score
        self._run.summary["best_params"] = WAndBTuningTracker._make_serializable(
            result.best_params
        )
        self._run.log(
            {"results": WAndBTuningTracker._to_wandb_table(result.report, spec.name)}
        )

    def track_spec(self, spec: TuningSpec) -> None: ...

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
                str(key): WAndBTuningTracker._make_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, list | tuple):
            return [WAndBTuningTracker._make_serializable(item) for item in value]

        return repr(value)
