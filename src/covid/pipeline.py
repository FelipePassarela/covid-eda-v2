from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from covid import constants, feature
from covid.feature import ColumnDropper, HighMissingRateDropper


def create_default_pipeline() -> Pipeline:
    classifier = LogisticRegression(
        solver="liblinear",
        C=0.01,
        max_iter=5000,
        random_state=constants.RANDOM_STATE,
    )
    pipeline = Pipeline(
        [
            ("dropper", ColumnDropper(columns_to_drop=[feature.ID])),
            ("missing_rate_dropper", HighMissingRateDropper(missing_threshold=0.05)),
            ("imputer", SimpleImputer(strategy="mean")),
            ("selector", SelectKBest(score_func=chi2, k=14)),
            ("scaler", StandardScaler()),
            ("sampler", RandomOverSampler(random_state=constants.RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )
    return pipeline.set_output(transform="pandas")


def create_explainable_pipeline() -> Pipeline:
    classifier = XGBClassifier(random_state=constants.RANDOM_STATE)
    pipeline = Pipeline(
        [
            ("dropper", ColumnDropper(columns_to_drop=[feature.ID])),
            ("missing_rate_dropper", HighMissingRateDropper(missing_threshold=0.05)),
            ("imputer", SimpleImputer(strategy="mean")),
            ("selector", SelectKBest(score_func=chi2, k=14)),
            ("sampler", RandomOverSampler(random_state=constants.RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )
    return pipeline.set_output(transform="pandas")
