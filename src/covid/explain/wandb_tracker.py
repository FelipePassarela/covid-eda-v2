from typing import Self

import wandb
from covid.explain.result import ExplainingResult
from covid.explain.spec import ExplainingSpec


class WandbExplainingTracker:
    def __init__(self, run_name: str) -> None:
        self._run_name = run_name
        self._run = None

    def __enter__(self) -> Self:
        self._run = wandb.init(
            project="covid",
            name=self._run_name,
            job_type="explaining",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._run:
            self._run.finish()

    def track_spec(self, spec: ExplainingSpec) -> None:
        self._run.config.update(
            {
                "pipeline/path": str(spec.pipeline_path),
                "pipeline/name": spec.pipeline_path.stem,
                "data/train/path": str(spec.train_path),
                "data/test/path": str(spec.test_path),
                "shap/max_display": spec.max_display,
            }
        )

    def track_result(self, result: ExplainingResult) -> None:
        self._run.config.update({"pipeline/steps": str(result.pipeline.steps)})
        self._run.summary.update(
            {
                "shap/importances": wandb.Table(data=result.importances),
                "data/train/n_samples": len(result.X_train_transformed),
                "data/train/n_features": result.X_train_transformed.shape[1],
                "data/test/n_samples": len(result.X_test_transformed),
                "data/test/n_features": result.X_test_transformed.shape[1],
                "shap/beeswarm": wandb.Image(result.beeswarm_plot),
            }
        )
