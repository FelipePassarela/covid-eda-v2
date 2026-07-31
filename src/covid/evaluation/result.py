from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ProbabilisticClassifier(Protocol):
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class EvaluationResult:
    balanced_accuracy: float
    recall: float
    precision: float
    f1: float
    roc_auc: float
    confusion_matrix: npt.NDArray[np.integer]

    @classmethod
    def from_model(
        cls, model: ProbabilisticClassifier, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Self:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        return cls(
            balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred),
        )

    def scores_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "balanced_accuracy": self.balanced_accuracy,
                    "recall": self.recall,
                    "precision": self.precision,
                    "f1": self.f1,
                    "roc_auc": self.roc_auc,
                }
            ]
        )

    def confusion_matrix_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.confusion_matrix,
            index=["actual_negative", "actual_positive"],
            columns=["predicted_negative", "predicted_positive"],
        )
