from typing import Any, Self

import pandas as pd
from numpy import dtype, ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class HighMissingRateDropper(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold: float | int) -> None:
        if not 0 <= missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1")

        self.missing_threshold = float(missing_threshold)

    def fit(self, X: pd.DataFrame, _=None) -> Self:
        missing_rates = X.isna().mean()
        dense_feats = missing_rates[missing_rates <= self.missing_threshold].index
        self.selected_features_ = dense_feats.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X[self.selected_features_]

    def get_feature_names_out(self, _=None) -> ndarray[tuple[int], dtype[Any]]:
        check_is_fitted(self, "selected_features_")
        return self.selected_features_
