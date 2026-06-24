from pathlib import Path

import yaml

from covid.training.cv_result import CVResults
from covid.training.cv_tracker import CVTracker


class FileCVTracker(CVTracker):
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def log_cv_results(self, cv_results: CVResults) -> None:
        output_file = self._output_dir / "cv_summary.json"
        summary = cv_results.summarize()
        summary.to_json(output_file, orient="records", lines=True)

    def log_model_params(self, model_name: str, params: dict) -> None:
        output_file = self._output_dir / f"{model_name}_params.yaml"
        with open(output_file, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
