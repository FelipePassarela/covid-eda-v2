import pytest

from covid.tune import TuningSpec
from covid.tune.nested import NestedTuningResult, NestedTuningSpec


@pytest.fixture()
def nested_spec(spec: TuningSpec) -> NestedTuningSpec:
    return NestedTuningSpec(inner=spec, n_outer_splits=3)


def assert_nested_result(result: NestedTuningResult, spec: NestedTuningSpec) -> None:
    assert result.mean_test_scores
    assert result.std_test_scores
    assert set(result.mean_test_scores) == {f"test_{s}" for s in spec.inner.scoring}
    assert set(result.std_test_scores) == {f"test_{s}" for s in spec.inner.scoring}

    assert len(result.params_per_fold) == spec.n_outer_splits
    assert not result.cv_report.empty
