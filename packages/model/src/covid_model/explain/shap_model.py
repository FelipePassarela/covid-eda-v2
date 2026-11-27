from collections.abc import Callable, Hashable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

type PredictionInput = pd.DataFrame | NDArray[np.float64]
type Predictor = Callable[[PredictionInput], NDArray[np.float64]]
type ShapModel = BaseEstimator | Predictor


class ClassProbabilityAdapter:
    """Expose one class probability per sample as a callable for SHAP."""

    def __init__(
        self,
        predict_proba: Predictor,
        *,
        classes: Iterable[Hashable],
        class_label: Hashable,
    ) -> None:
        self._predict_proba = predict_proba
        self._class_index = list(classes).index(class_label)

    def __call__(self, X: PredictionInput) -> NDArray[np.float64]:
        probabilities = self._predict_proba(X)
        return probabilities[:, self._class_index]


def model_for_shap(classifier: BaseEstimator, *, positive_label: Hashable) -> ShapModel:
    if isinstance(classifier, SVC):
        return ClassProbabilityAdapter(
            classifier.predict_proba,
            classes=classifier.classes_,
            class_label=positive_label,
        )
    return classifier
