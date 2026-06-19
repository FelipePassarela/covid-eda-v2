from typing import Mapping

from pynndescent.pynndescent_ import TransformerMixin


class PreprocessorFactory:
    def __init__(
        self,
        imputer_registry: Mapping[str, type[TransformerMixin]],
        scaler_registry: Mapping[str, type[TransformerMixin]],
        reducer_registry: Mapping[str, type[TransformerMixin]],
    ) -> None:
        self.imputer_reg = imputer_registry
        self.scaler_reg = scaler_registry
        self.reducer_reg = reducer_registry

    def create_imputer(self, name: str, **kwargs) -> TransformerMixin:
        if name not in self.imputer_reg:
            raise ValueError(f"Imputer '{name}' is not registered.")
        return self.imputer_reg[name](**kwargs)

    def create_scaler(self, name: str, **kwargs) -> TransformerMixin:
        if name not in self.scaler_reg:
            raise ValueError(f"Scaler '{name}' is not registered.")
        return self.scaler_reg[name](**kwargs)

    def create_reducer(self, name: str, **kwargs) -> TransformerMixin:
        if name not in self.reducer_reg:
            raise ValueError(f"Reducer '{name}' is not registered.")
        return self.reducer_reg[name](**kwargs)
