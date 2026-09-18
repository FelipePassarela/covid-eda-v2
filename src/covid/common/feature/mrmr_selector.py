from typing import Self

import pandas as pd
from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 50, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs
        self.selected_features_ = None
        self.k = k

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self.selected_features_ = mrmr_classif(
            X=X, y=y, K=self.k, n_jobs=self.n_jobs, show_progress=True
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X[self.selected_features_]

    def get_feature_names_out(self, _=None) -> list[str]:
        check_is_fitted(self, "selected_features_")
        return self.selected_features_
