import pandas as pd

from .data_cleaning_config import DataCleaningConfig


class CovidDataset:
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._X = X
        self._y = y

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, cleaning_config: DataCleaningConfig
    ) -> CovidDataset:
        target_column = cleaning_config.target_column
        id_column = cleaning_config.id_column

        X = df.drop(columns=[target_column])
        y = df[target_column]
        return (  # TODO: Make this configurable
            cls(X, y)
            ._with_basic_clean(id_column)
            ._without_sparse_columns(threshold=cleaning_config.sparse_threshold)
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat([self._X, self._y], axis=1)

    def split(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y

    def _without_sparse_columns(self, threshold: float) -> CovidDataset:
        missing_ratio = self._X.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > threshold].index
        X_cleaned = self._X.drop(columns=columns_to_drop)
        return CovidDataset(X_cleaned, self._y)

    def _as_categorical(self) -> CovidDataset:
        X = self._X.astype("category")
        return CovidDataset(X, self._y)

    @property
    def X(self) -> pd.DataFrame:
        return self._X.copy()

    @property
    def y(self) -> pd.Series:
        return self._y.copy()

    def _without_missing_target(self) -> CovidDataset:
        mask = self._y.notna()
        X_cleaned = self._X[mask]
        y_cleaned = self._y[mask]
        return CovidDataset(X_cleaned, y_cleaned)

    def _without_id(self, id_column: str) -> CovidDataset:
        if id_column in self._X.columns:
            X_cleaned = self._X.drop(columns=[id_column])
            return CovidDataset(X_cleaned, self._y)
        return self

    def _with_basic_clean(self, id_column: str) -> CovidDataset:
        y = self._y.astype(bool)
        return CovidDataset(self._X, y)._without_missing_target()._without_id(id_column)
