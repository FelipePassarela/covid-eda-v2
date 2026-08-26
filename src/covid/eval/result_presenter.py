import pandas as pd

from covid.eval.result import EvaluationResult


def present_evaluation_result(result: EvaluationResult) -> None:
    scores = _scores_to_dataframe(result)
    conf_matrix = _confusion_matrix_to_dataframe(result)

    print("Evaluation scores:\n", scores.to_string(index=False))
    print("\nConfusion matrix:\n", conf_matrix)


def _scores_to_dataframe(result: EvaluationResult) -> pd.DataFrame:
    scores_df = pd.DataFrame(
        [
            {
                "balanced_accuracy": result.balanced_accuracy,
                "recall": result.recall,
                "precision": result.precision,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        ]
    )
    return scores_df.round(3)


def _confusion_matrix_to_dataframe(result: EvaluationResult) -> pd.DataFrame:
    return pd.DataFrame(
        result.confusion_matrix,
        index=["actual_negative", "actual_positive"],
        columns=["predicted_negative", "predicted_positive"],
    )
