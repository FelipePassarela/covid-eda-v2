from dataclasses import dataclass
from typing import Self

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


@dataclass(frozen=True)
class EvaluationResult:
    balanced_accuracy: float
    recall: float
    precision: float
    f1: float
    roc_auc: float
    confusion_matrix: npt.NDArray[np.integer]
    y_pred: pd.Series
    y_true: pd.Series
    y_score: pd.Series | None = None

    @classmethod
    def from_predictions(
        cls, y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None
    ) -> Self:
        roc_auc = (
            roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
        )

        return cls(
            balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix(y_true, y_pred),
            y_pred=y_pred,
            y_true=y_true,
            y_score=y_score,
        )
