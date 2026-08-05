from pathlib import Path

from pandas import DataFrame, Series
from sklearn.model_selection import TunedThresholdClassifierCV

import wandb
from wandb import Run


class WAndBTrainingTracker:
    def __init__(self, run: Run) -> None:
        self.run = run

    def track_data(self, X_train: DataFrame, y_train: Series) -> None:
        self.run.summary.update(
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
        self.run.summary.update(
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
        self.run.log({"threshold_tuning/score_by_threshold": line_plot})

    def track_model(self, model_path: Path) -> None:
        artifact = wandb.Artifact(
            name="long-covid-classifier",
            type="model",
            metadata={"format": "joblib", "framework": "scikit-learn"},
        )
        artifact.add_file(str(model_path), name=model_path.name)
        self.run.log_artifact(artifact, aliases=["latest"])
