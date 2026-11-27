from typing import Any

from optuna.trial import FrozenTrial
from optuna_integration import OptunaSearchCV
from sklearn.base import BaseEstimator


class OptunaSearchCVAdapter:
    """Expose Optuna trials using the column names expected by CV reports."""

    def __init__(self, search: OptunaSearchCV, metric: str) -> None:
        self._search = search
        self._metric = metric

    @property
    def best_estimator_(self) -> BaseEstimator:
        return self._search.best_estimator_

    @property
    def best_params_(self) -> dict[str, Any]:
        return self._search.best_params_

    @property
    def best_score_(self) -> float:
        return self._search.best_score_

    @property
    def cv_results_(self) -> dict[str, Any]:
        # Build rows from trials rather than cv_results_: failed/pruned trials
        # can have missing attributes. Keep their parameters and use NaN scores.
        trials = self._search.trials_
        return {
            "params": [trial.params for trial in trials],
            **self._build_metric_columns(trials),
        }

    def _build_metric_columns(
        self, trials: list[FrozenTrial]
    ) -> dict[str, list[float]]:
        metric_columns = {}

        for split in ("test", "train"):
            if split == "train" and not self._search.return_train_score:
                continue

            for statistic in ("mean", "std"):
                trial_attr = f"{statistic}_{split}_score"
                column_name = f"{statistic}_{split}_{self._metric}"
                metric_columns[column_name] = [
                    trial.user_attrs.get(trial_attr, float("nan")) for trial in trials
                ]

        return metric_columns
