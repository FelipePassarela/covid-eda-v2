from unittest.mock import MagicMock

from covid.tune.nested import NestedTuningSpec
from covid.tune.nested.nested_tune import nested_tune
from tests.integration.tune.nested.conftest import assert_nested_result


def test_nested_tune_succeeds_with_valid_spec(nested_spec: NestedTuningSpec) -> None:
    result = nested_tune(nested_spec, tracker=MagicMock())
    assert_nested_result(result, nested_spec)
