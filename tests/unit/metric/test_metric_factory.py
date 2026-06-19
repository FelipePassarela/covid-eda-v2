from typing import Callable

import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from covid.metric import MetricFactory


@pytest.mark.parametrize(
    "metric_name, metric_func",
    [
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
    ],
)
def test_create(factory: MetricFactory, metric_name: str, metric_func: Callable):
    metric = factory.create(metric_name)
    assert callable(metric)
    assert metric == metric_func
