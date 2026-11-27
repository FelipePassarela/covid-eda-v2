from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from covid_model.explain import ExplainingSpec, explain
from covid_model.explain.result import ExplainingResult
from covid_model.explain.tracker import ExplainingTracker


def test_explain_closes_plot_when_tracking_fails() -> None:
    spec = ExplainingSpec(Path("model.pkl"), Path("data.csv"))
    result = MagicMock(spec=ExplainingResult)
    tracker = MagicMock(spec=ExplainingTracker)
    tracker.track_result.side_effect = RuntimeError("tracking failed")

    with (
        patch("covid_model.explain.explain._explain", return_value=result),
        pytest.raises(RuntimeError, match="tracking failed"),
    ):
        explain(spec, tracker)

    tracker.track_spec.assert_called_once_with(spec)
    tracker.track_result.assert_called_once_with(result)
    result.close_plot.assert_called_once_with()
