from pathlib import Path

import pandas as pd

from covid.data import load_and_split_data
from covid.eval.context import EvaluationContext
from covid.eval.result import EvaluationResult
from covid.eval.result_presenter import present_evaluation_result
from covid.eval.tracker import EvaluationTracker
from covid.pipeline import load_pipeline


def evaluate(pipeline_path: Path, data_path: Path, tracker: EvaluationTracker) -> None:
    tracker.track_spec(pipeline_path, data_path)

    X, y = load_and_split_data(data_path)
    context = EvaluationContext(X, y, pipeline=load_pipeline(pipeline_path))
    result = _evaluate_pipeline(context)

    tracker.track_context(context)
    tracker.track_result(result)
    present_evaluation_result(result)


def _evaluate_pipeline(ctx: EvaluationContext) -> EvaluationResult:
    X, y = ctx.X, ctx.y
    pipeline = ctx.pipeline

    y_pred = pipeline.predict(X)
    y_score = pipeline.predict_proba(X)[:, 1]

    return EvaluationResult.from_predictions(
        y_true=y,
        y_pred=pd.Series(y_pred, index=y.index),
        y_score=pd.Series(y_score, index=y.index),
    )
