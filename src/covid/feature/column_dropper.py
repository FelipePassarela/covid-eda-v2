from typing import Self, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop: Sequence[str]) -> None:
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        self.feature_names_in = X.columns
        self.feature_names_out = [
            col for col in X.columns if col not in self.columns_to_drop
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out")
        return X.drop(columns=self.columns_to_drop, errors="raise")

    def get_feature_names_out(self, _=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out")
        return np.array(self.feature_names_out)
