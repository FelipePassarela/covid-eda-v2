from dataclasses import dataclass, field
from typing import Any

from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.feature_selection import chi2, f_classif
from xgboost import XGBClassifier

from covid import constants
from covid.pipeline import DEFAULT_PIPELINE


@dataclass
class RandomizedSearchSpec:
    name: str
    pipeline: Pipeline
    param_distributions: dict[str, Any]
    n_searches: int
    n_fold_repeats: int = 5
    scoring: list[str] = field(default_factory=lambda: ["balanced_accuracy"])

    @staticmethod
    def create_specs(
        scoring: list[str], quick: bool = False
    ) -> list[RandomizedSearchSpec]:
        return [
            RandomizedSearchSpec(
                name="xgboost",
                pipeline=DEFAULT_PIPELINE,
                param_distributions={
                    "scaler": [None],
                    "selector__score_func": [chi2, f_classif],
                    "classifier": [
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            n_jobs=1,
                            random_state=constants.RANDOM_STATE,
                        )
                    ],
                    "classifier__n_estimators": randint(100, 701),
                    "classifier__learning_rate": loguniform(1e-4, 3e-1),
                    "classifier__max_depth": randint(2, 7),
                    "classifier__subsample": uniform(0.6, 0.4),
                    "classifier__colsample_bytree": uniform(0.5, 0.5),
                    "classifier__min_child_weight": loguniform(1e-1, 1e1),
                    "classifier__reg_alpha": loguniform(1e-4, 1e1),
                    "classifier__reg_lambda": loguniform(1e-2, 1e2),
                },
                n_searches=3 if quick else 35,
                n_fold_repeats=1 if quick else 5,
                scoring=scoring,
            ),
            RandomizedSearchSpec(
                name="score_func",
                pipeline=DEFAULT_PIPELINE,
                param_distributions={
                    "selector__k": randint(5, 30),
                    "classifier__C": loguniform(0.00001, 100),
                    "classifier__solver": ["liblinear", "lbfgs", "newton-cholesky"],
                },
                n_searches=3 if quick else 50,
                n_fold_repeats=1 if quick else 5,
                scoring=scoring,
            ),
        ]
