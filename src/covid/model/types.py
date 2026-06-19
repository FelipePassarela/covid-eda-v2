from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PredictiveModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> PredictiveModel: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...
