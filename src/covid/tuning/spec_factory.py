from typing import Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from covid import constants
from covid.pipeline import create_default_pipeline
from covid.tuning import RandomizedSearchSpec


def create_specs(
    spec_names: list[str], scoring: list[str], quick: bool = False
) -> list[RandomizedSearchSpec]:
    unknown_names = set(spec_names) - SPEC_FACTORIES.keys()
    if unknown_names:
        raise ValueError(
            f"Unknown specs: {sorted(unknown_names)}. "
            f"Available specs: {sorted(SPEC_FACTORIES)}"
        )

    return [SPEC_FACTORIES[name](scoring, quick) for name in spec_names]


def create_all_specs(
    scoring: list[str], quick: bool = False
) -> list[RandomizedSearchSpec]:
    return [factory(scoring, quick) for factory in SPEC_FACTORIES.values()]


def create_xgboost_spec(scoring: list[str], quick: bool) -> RandomizedSearchSpec:
    classifier = XGBClassifier(
        objective="binary:logistic", n_jobs=1, random_state=constants.RANDOM_STATE
    )
    return RandomizedSearchSpec(
        name="xgboost",
        pipeline=create_default_pipeline(),
        param_distributions={
            "scaler": [None],
            "selector__k": randint(5, 30),
            "classifier": [classifier],
            "classifier__n_estimators": randint(50, 701),
            "classifier__learning_rate": loguniform(1e-4, 3e-1),
            "classifier__max_depth": randint(2, 5),
            "classifier__subsample": uniform(0.7, 0.3),
            "classifier__colsample_bytree": uniform(0.5, 0.5),
            "classifier__min_child_weight": loguniform(1e-1, 1e1),
            "classifier__reg_alpha": loguniform(1e-4, 1e1),
            "classifier__reg_lambda": loguniform(1e-2, 1e2),
        },
        n_searches=3 if quick else 300,
        n_fold_repeats=1 if quick else 3,
        scoring=scoring,
    )


def create_logistic_regression_spec(
    scoring: list[str], quick: bool
) -> RandomizedSearchSpec:
    return RandomizedSearchSpec(
        name="logistic_regression",
        pipeline=create_default_pipeline(),
        param_distributions={
            "selector__k": randint(5, 30),
            "classifier__C": loguniform(0.00001, 100),
            "classifier__solver": ["liblinear", "lbfgs", "newton-cholesky"],
        },
        n_searches=3 if quick else 50,
        n_fold_repeats=1 if quick else 5,
        scoring=scoring,
    )


def create_linear_svm_spec(scoring: list[str], quick: bool) -> RandomizedSearchSpec:
    return RandomizedSearchSpec(
        name="linear_svm",
        pipeline=create_default_pipeline(),
        param_distributions={
            "selector__k": randint(5, 30),
            "classifier": [LinearSVC(random_state=constants.RANDOM_STATE)],
            "classifier__C": loguniform(0.0001, 100),
            "classifier__penalty": ["l1", "l2"],
        },
        n_searches=3 if quick else 50,
        n_fold_repeats=1 if quick else 5,
        scoring=scoring,
    )


def create_svm_rbf_spec(scoring: list[str], quick: bool) -> RandomizedSearchSpec:
    return RandomizedSearchSpec(
        name="svm_rbf",
        pipeline=create_default_pipeline(),
        param_distributions={
            "selector__k": randint(5, 30),
            "classifier": [SVC(random_state=constants.RANDOM_STATE)],
            "classifier__C": loguniform(0.0001, 100),
            "classifier__gamma": loguniform(1e-4, 1),
        },
        n_searches=3 if quick else 50,
        n_fold_repeats=1 if quick else 5,
        scoring=scoring,
    )


SpecFactory = Callable[[list[str], bool], RandomizedSearchSpec]

SPEC_FACTORIES: dict[str, SpecFactory] = {
    "xgboost": create_xgboost_spec,
    "logistic_regression": create_logistic_regression_spec,
    "linear_svm": create_linear_svm_spec,
    "svm_rbf": create_svm_rbf_spec,
}
