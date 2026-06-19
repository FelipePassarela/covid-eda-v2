from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "xgboost": XGBClassifier,
    "random_forest": RandomForestClassifier,
    "dummy": DummyClassifier,
    # "lightgbm": LGBMClassifier,  # TODO: add LightGBM support
}

MODEL_ALIASES = {
    "lr": "logistic_regression",
    "svc": "svm",
    "xgb": "xgboost",
    "rf": "random_forest",
    "dmy": "dummy",
    # "lgbm": "lightgbm",  # TODO: add LightGBM support
}
