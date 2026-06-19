from typing import Callable, Mapping

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, clone
from tqdm import tqdm

from covid.training.cv_result import CVResult


class CrossValidator:
    def __init__(
        self,
        n_folds: int,
        metrics: dict[str, Callable],
        random_state: int | None,
        shuffle: bool,
    ):
        self._n_folds = n_folds
        self._metrics = metrics
        self._random_state = random_state
        self._splitter = StratifiedKFold(
            n_splits=self._n_folds, shuffle=shuffle, random_state=self._random_state
        )

    def run(
        self, models: Mapping[str, Pipeline], X: pd.DataFrame, y: pd.Series
    ) -> CVResult:
        logger.info(
            "Starting cross-validation with {n_folds} folds", n_folds=self._n_folds
        )
        metric_keys = list(self._metrics.keys())
        logger.info("Metrics being evaluated: {metrics}", metrics=metric_keys)
        logger.info("Models being evaluated: {models}", models=list(models.keys()))

        result = CVResult()

        total_steps = self._n_folds * len(models)
        with tqdm(total=total_steps, desc="Cross-validation") as pbar:
            self._run_cv(models, X, y, result, lambda: pbar.update(1))

        return result

    def _run_cv(
        self,
        models: Mapping[str, Pipeline],
        X: pd.DataFrame,
        y: pd.Series,
        result: CVResult,
        on_model_evaluated: Callable | None = None,
    ) -> None:
        for fold_idx, (train_idx, test_idx) in enumerate(
            self._splitter.split(X, y), start=1
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for name, model in models.items():
                model = clone(model)  # Ensure a fresh model for each fold
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                computed_metrics = {
                    m: func(y_test, y_pred) for m, func in self._metrics.items()
                }
                result.add_fold_metrics(
                    model_name=name,
                    fold=fold_idx,
                    metrics=computed_metrics,
                )

                if on_model_evaluated:
                    on_model_evaluated()
