from unittest.mock import MagicMock, patch

from covid_tuning import tune


def test_tune_orchestrates_successfully() -> None:
    spec = MagicMock()
    tracker = MagicMock()
    expected_result = MagicMock()

    with patch(
        "covid_tuning.search.tuning._tune", return_value=expected_result
    ) as mock_tune:
        result = tune(spec, tracker)

    mock_tune.assert_called_once_with(spec)
    tracker.track_spec.assert_called_once_with(spec)
    tracker.track_result.assert_called_once_with(expected_result)
    assert result == expected_result
