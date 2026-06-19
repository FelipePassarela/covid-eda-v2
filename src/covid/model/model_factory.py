from .types import PredictiveModel


class ModelFactory:
    def __init__(
        self,
        model_registry: dict[str, type[PredictiveModel]],
        model_aliases: dict[str, str],
    ) -> None:
        self.model_registry = model_registry
        self.model_aliases = model_aliases
        self._validate_aliases()

    def create(self, model_name: str, **kwargs) -> PredictiveModel:
        if model_name in self.model_aliases:
            model_name = self.model_aliases[model_name]

        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' is not registered.")

        model_class = self.model_registry[model_name]
        return model_class(**kwargs)

    def _validate_aliases(self) -> None:
        invalid = {
            k: v for k, v in self.model_aliases.items() if v not in self.model_registry
        }
        if invalid:
            raise ValueError(f"Aliases point to unregistered models: {invalid}")

    def registered_models(self) -> list[str]:
        return list(self.model_registry.keys())

    def model_name_from_class(self, model_class: type[PredictiveModel]) -> str | None:
        for name, cls in self.model_registry.items():
            if cls == model_class:
                return name
        return None
