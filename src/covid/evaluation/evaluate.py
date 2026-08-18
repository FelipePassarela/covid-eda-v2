from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from covid.data import load_and_split_data
from covid.evaluation.result import EvaluationResult
from covid.evaluation.result_presenter import present_evaluation_result
from covid.pipeline import load_pipeline


def evaluate(pipeline_path: Path, data_path: Path) -> None:
    X, y = load_and_split_data(data_path)
    pipeline = load_pipeline(pipeline_path)

    result = _evaluate_pipeline(pipeline, X, y)
    present_evaluation_result(result)


def _evaluate_pipeline(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> EvaluationResult:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return EvaluationResult(
        balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )
