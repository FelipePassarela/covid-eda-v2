import numpy as np
import pytest
from shap import Explanation


@pytest.fixture()
def explanation() -> Explanation:
    return Explanation(
        values=np.array([[-1.0, 4.0, -2.0], [3.0, -2.0, 0.0]]),
        base_values=np.zeros(2),
        data=np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]]),
        feature_names=["small", "large", "tiny"],
    )
