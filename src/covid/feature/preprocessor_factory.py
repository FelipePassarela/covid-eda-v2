from pynndescent.pynndescent_ import TransformerMixin

from .preprocessor_registry import PreprocessorRegistry


class PreprocessorFactory:
    def __init__(
        self,
        preprocessor_registry: PreprocessorRegistry,
    ) -> None:
        self.imputer_reg = preprocessor_registry["imputer"]
        self.scaler_reg = preprocessor_registry["scaler"]
        self.selector_reg = preprocessor_registry["selector"]
        self.reducer_reg = preprocessor_registry["reducer"]

    def create_imputer(self, name: str, **kwargs) -> TransformerMixin:
        if name not in self.imputer_reg:
            raise ValueError(f"Imputer '{name}' is not registered.")
        return self.imputer_reg[name](**kwargs)

    def create_selector(self, name: str, **kwargs) -> TransformerMixin:
        if name not in self.selector_reg:
            raise ValueError(f"Selector '{name}' is not registered.")
        return self.selector_reg[name](**kwargs)

    def create_scaler(self, name: str, **kwargs) -> TransformerMixin:
        if name not in self.scaler_reg:
            raise ValueError(f"Scaler '{name}' is not registered.")
        return self.scaler_reg[name](**kwargs)

    def create_reducer(self, name: str, **kwargs) -> TransformerMixin:
        if name not in self.reducer_reg:
            raise ValueError(f"Reducer '{name}' is not registered.")
        return self.reducer_reg[name](**kwargs)
