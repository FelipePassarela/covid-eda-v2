from covid.cross_val.domain.cv_result import CVResults
from covid.cross_val.domain.cv_tracker import CVTracker


class NullCVTracker(CVTracker):
    def log_cv_results(self, cv_results: CVResults) -> None:
        pass

    def log_model_params(self, model_name: str, params: dict) -> None:
        pass
