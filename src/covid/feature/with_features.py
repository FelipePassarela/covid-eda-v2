from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WithFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str]) -> None:
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features]

    def get_feature_names_out(self) -> list[str]:
        return self.features
