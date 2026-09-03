from pathlib import Path
from unittest.mock import MagicMock, patch

from covid.tune import TuningSpec, tune
from tests.integration.tune.base.conftest import assert_mocks_called, assert_result


def test_tune_succeeds_with_synthetic_data(spec: TuningSpec, tmp_path: Path) -> None:
    result = tune(spec, tracker=MagicMock())
    assert_result(result, spec)


def test_tune_orchestrates_successfully() -> None:
    spec = MagicMock()
    tracker = MagicMock()
    expected_result = MagicMock()

    with patch("covid.tune.base.tune._tune", return_value=expected_result) as mock_tune:
        result = tune(spec, tracker)

    assert_mocks_called(mock_tune, spec, tracker, expected_result)
    assert result == expected_result
