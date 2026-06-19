from collections.abc import Callable


class MetricFactory:
    def __init__(self, metric_registry: dict[str, Callable]) -> None:
        self.metric_registry = metric_registry

    def create(self, metric: str) -> Callable:
        if metric not in self.metric_registry:
            raise ValueError(f"Metric '{metric}' is not registered.")
        return self.metric_registry[metric]

    def create_all(self, metrics: list[str]) -> dict[str, Callable]:
        return {m: self.create(m) for m in metrics}
