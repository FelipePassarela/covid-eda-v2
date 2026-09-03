import json
from typing import Any, Self

import pandas as pd
import wandb

from covid.tune.base import TuningResult, TuningSpec


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

        WAndBTuningTracker.track_spec_for_run(self._run, spec)

    @staticmethod
    def track_spec_for_run(run: wandb.Run, spec: TuningSpec, group: str = "") -> None:
        param_dist = WAndBTuningTracker.make_serializable(spec.param_distributions)
        pipeline = WAndBTuningTracker.make_serializable(spec.pipeline.steps)

        run.config.update(
            {
                f"{group}/data/path": spec.data_path,
                f"{group}/pipeline/steps": pipeline,
                f"{group}/search/param_distributions": param_dist,
                f"{group}/search/n_searches": spec.n_searches,
                f"{group}/search/n_fold_repeats": spec.n_fold_repeats,
                f"{group}/search/scoring": spec.scoring,
            }
        )

    @staticmethod
    def make_serializable(value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value

        if isinstance(value, dict):
            return {
                str(key): WAndBTuningTracker.make_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, list | tuple):
            return [WAndBTuningTracker.make_serializable(item) for item in value]

        return repr(value)

    def track_result(self, result: TuningResult) -> None:
        if not self._run:
            raise RuntimeError("This class must be used as a context manager.")

        best_params = WAndBTuningTracker.make_serializable(result.best_params)
        cv_report = WAndBTuningTracker.to_wandb_table(result.cv_report)

        self._run.summary["result/best_score"] = result.best_score
        self._run.summary["result/best_params"] = best_params
        self._run.log({"result/cv_report": cv_report})

    @classmethod
    def to_wandb_table(cls, dataframe: pd.DataFrame) -> wandb.Table:
        serializable = dataframe.reset_index().copy()
        serializable.drop("index", axis=1, inplace=True)

        for column in serializable.columns:
            serializable[column] = serializable[column].map(cls._make_table_cell)

        return wandb.Table(dataframe=serializable)

    @classmethod
    def _make_table_cell(cls, value: Any) -> Any:
        value = cls.make_serializable(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return value
