from pathlib import Path
from typing import Self

import wandb
from covid.eval.context import EvaluationContext
from covid.eval.result import EvaluationResult
from covid.common.pipeline import unwrap_threshold_model_if_needed


class WandBEvaluationTracker:
    def __init__(self, model_name: str, config: dict) -> None:
        self._run = None
        self._model_name = model_name
        self._config = config

    def __enter__(self) -> Self:
        self._run = wandb.init(
            project="covid",
            name=self._model_name,
            job_type="evaluation",
            config=self._config,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._run:
            self._run.finish()

    def track_spec(self, pipeline_path: Path, data_path: Path) -> None:
        self._run.config.update(
            {
                "pipeline/path": str(pipeline_path),
                "pipeline/name": pipeline_path.stem,
                "data/path": str(data_path),
            }
        )

    def track_context(self, context: EvaluationContext) -> None:
        X, y = context.X, context.y
        pipeline = unwrap_threshold_model_if_needed(context.pipeline)
        self._run.summary.update(
            {
                "data/n_samples": len(X),
                "data/n_features": X.shape[1],
                "data/negative_samples": int((y == 0).sum()),
                "data/positive_samples": int((y == 1).sum()),
                "data/positive_rate": float(y.mean()),
                "pipeline/steps": str(pipeline.steps),
            }
        )

    def track_result(self, result: EvaluationResult) -> None:
        self._run.summary.update(
            {
                "result/balanced_accuracy": result.balanced_accuracy,
                "result/precision": result.precision,
                "result/recall": result.recall,
                "result/f1": result.f1,
                "result/roc_auc": result.roc_auc,
            }
        )

        confusion_matrix = wandb.plot.confusion_matrix(
            y_true=result.y_true,
            preds=result.y_pred,
            class_names=result.y_true.unique(),
        )
        self._run.log({"result/confusion_matrix": confusion_matrix})
