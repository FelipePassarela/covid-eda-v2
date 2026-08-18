from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass(frozen=True)
class EvaluationResult:
    balanced_accuracy: float
    recall: float
    precision: float
    f1: float
    roc_auc: float
    confusion_matrix: npt.NDArray[np.integer]
