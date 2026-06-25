from pathlib import Path

import yaml
from matplotlib.figure import Figure

from ..domain.cv_result import CVResults
from ..domain.cv_tracker import CVTracker


class FileCVTracker(CVTracker):
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def log_cv_results(self, cv_results: CVResults) -> None:
        fold_results_df = cv_results.to_dataframe()
        fold_results_df.to_csv(self._output_dir / "fold_results.csv", index=False)
        summary_df = cv_results.summarize()
        # flatten the multi-index columns for easier CSV export
        summary_df.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in summary_df.columns.values
        ]
        summary_df.to_csv(self._output_dir / "summary_results.csv")


    def log_model_params(self, model_name: str, params: dict) -> None:
        output_file = self._output_dir / f"{model_name}_params.yaml"
        with open(output_file, "w") as f:
            yaml.dump(params, f, default_flow_style=False)

    def log_cv_plot(self, fig: Figure, filename: str = "plot.png") -> None:
        output_file = self._output_dir / filename
        fig.savefig(output_file)
