from typing import Any, Self

import pandas as pd
from numpy import dtype, ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from covid.eda.bivariate import association_summary


class QValueSelector(BaseEstimator, TransformerMixin):
    def __init__(self, q_value_threshold: float | int) -> None:
        self.q_value_threshold = q_value_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        association = association_summary(X, y)
        associated_feats = association.query(
            f"q_value < {self.q_value_threshold}"
        ).index
        self.selected_features_ = associated_feats.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X[self.selected_features_]

    def get_feature_names_out(self, _=None) -> ndarray[tuple[int], dtype[Any]]:
        check_is_fitted(self, "selected_features_")
        return self.selected_features_
