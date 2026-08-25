from pathlib import Path
from typing import Self

import pandas as pd
from sklearn.model_selection import TunedThresholdClassifierCV

import wandb


class WAndBTrainingTracker:
    def __init__(self, config: dict) -> None:
        self._config = config
        self._run = None

    def __enter__(self) -> Self:
        self._run = wandb.init(
            project="covid",
            name=self._config["name"],
            job_type="train",
            config=self._config,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._run.finish()

    def track_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._run.summary.update(
            {
                "data/n_samples": len(X_train),
                "data/n_features": X_train.shape[1],
                "data/negative_samples": int((y_train == 0).sum()),
                "data/positive_samples": int((y_train == 1).sum()),
                "data/positive_rate": float(y_train.mean()),
            }
        )

    def track_threshold_tuning(
        self, model: TunedThresholdClassifierCV, scoring: str
    ) -> None:
        self._run.summary.update(
            {
                "threshold_tuning/scoring": scoring,
                "threshold_tuning/best_score": model.best_score_,
                "threshold_tuning/best_threshold": model.best_threshold_,
            }
        )

        results = [
            [float(threshold), float(score)]
            for threshold, score in zip(
                model.cv_results_["thresholds"],
                model.cv_results_["scores"],
                strict=True,
            )
        ]
        table = wandb.Table(columns=["threshold", "score"], data=results)

        line_plot = wandb.plot.line(
            table,
            x="threshold",
            y="score",
            title=f"{scoring} by decision threshold",
        )
        self._run.log({"threshold_tuning/score_by_threshold": line_plot})

    def track_model(self, model_path: Path) -> None:
        artifact = wandb.Artifact(
            name="long-covid-classifier",
            type="model",
            metadata={"format": "joblib", "framework": "scikit-learn"},
        )
        artifact.add_file(str(model_path), name=model_path.name)
        self._run.log_artifact(artifact, aliases=["latest"])
