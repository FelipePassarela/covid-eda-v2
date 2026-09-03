from typing import Any, Self

import wandb
from covid.tune.base.tracking import WAndBTuningTracker
from covid.tune.common.wand_utils import make_serializable, to_wandb_table
from covid.tune.nested.nested_tuning_spec import NestedTuningSpec
from covid.tune.nested.result.nested_result import NestedTuningResult


class WandBNestedTuningTracker:
    def __init__(self, config: dict[str, Any], run_name: str | None = None) -> None:
        self._run_name = run_name
        self._run: wandb.Run | None = None
        self._config = config

    def __enter__(self) -> Self:
        self._run: wandb.Run = wandb.init(
            project="covid",
            name=self._run_name,
            job_type="nested-tuning",
            config=self._config,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._run:
            self._run.finish()

    def track_spec(self, spec: NestedTuningSpec) -> None:
        if not self._run:
            raise RuntimeError("This class must be used as a context manager.")

        WAndBTuningTracker.track_spec_for_run(self._run, spec.inner, group="inner")
        self._run.config["outer/n_splits"] = spec.outer_n_splits

    def track_result(self, result: NestedTuningResult) -> None:
        if not self._run:
            raise RuntimeError("This class must be used as a context manager.")

        self._run.summary.update(
            {
                "result/mean_test_scores": result.mean_test_scores,
                "result/mean_train_scores": result.mean_train_scores,
                "result/std_test_scores": result.std_test_scores,
                "result/std_train_scores": result.std_train_scores,
                "result/params_per_fold": make_serializable(result.params_per_fold),
            }
        )
        self._run.log({"result/outer_cv_report": to_wandb_table(result.cv_report)})
