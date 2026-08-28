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

    def track_spec(self, spec: TuningSpec) -> None:
        if not self._run:
            raise RuntimeError("This class must be used as a context manager.")

        param_dist = WAndBTuningTracker._make_serializable(spec.param_distributions)
        pipeline = WAndBTuningTracker._make_serializable(spec.pipeline.steps)

        self._run.summary.update(
            {
                "data/path": spec.data_path,
                "pipeline/steps": pipeline,
                "search/param_distributions": param_dist,
                "search/n_searches": spec.n_searches,
                "search/n_fold_repeats": spec.n_fold_repeats,
                "search/scoring": spec.scoring,
            }
        )

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

    def track_result(self, result: TuningResult) -> None:
        if not self._run:
            raise RuntimeError("This class must be used as a context manager.")

        best_params = WAndBTuningTracker._make_serializable(result.best_params)
        result_table = WAndBTuningTracker._to_wandb_table(result.report)

        self._run.summary["result/best_score"] = result.best_score
        self._run.summary["result/best_params"] = best_params
        self._run.log({"result/results": result_table})

    @classmethod
    def _to_wandb_table(cls, dataframe: pd.DataFrame) -> wandb.Table:
        serializable = dataframe.reset_index().copy()
        serializable.drop("index", axis=1, inplace=True)

        for column in serializable.columns:
            serializable[column] = serializable[column].map(cls._make_table_cell)

        return wandb.Table(dataframe=serializable)

    @classmethod
    def _make_table_cell(cls, value: Any) -> Any:
        value = cls._make_serializable(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return value
