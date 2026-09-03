from typing import Any, Self

import wandb
from covid.tune.base import TuningResult, TuningSpec
from covid.tune.common.wand_utils import make_serializable, to_wandb_table


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
        param_dist = make_serializable(spec.param_distributions)
        pipeline = make_serializable(spec.pipeline.steps)

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

    def track_result(self, result: TuningResult) -> None:
        if not self._run:
            raise RuntimeError("This class must be used as a context manager.")

        best_params = make_serializable(result.best_params)
        cv_report = to_wandb_table(result.cv_report)

        self._run.summary["result/best_score"] = result.best_score
        self._run.summary["result/best_params"] = best_params
        self._run.log({"result/cv_report": cv_report})
