import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from covid.metric import MetricFactory


@pytest.fixture
def factory() -> MetricFactory:
    return MetricFactory(
        {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        }
    )
