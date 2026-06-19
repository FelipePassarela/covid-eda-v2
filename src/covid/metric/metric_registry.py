from functools import partial

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_REGISTRY = {
    "accuracy": accuracy_score,
    "precision": partial(precision_score, average="binary", zero_division=0),
    "recall": partial(recall_score, average="binary", zero_division=0),
    "f1": partial(f1_score, average="binary", zero_division=0),
    "balanced_accuracy": partial(balanced_accuracy_score, adjusted=True),
    "roc_auc": roc_auc_score,
    "matthews": matthews_corrcoef,
}
