import pandas as pd


class BoxplotData:
    def __init__(
        self, model_names: list[str], folds_scores: list[list[float | int]], metric: str
    ) -> None:
        self.validate_params(model_names, folds_scores)

        self._models = model_names
        self._fold_scores = folds_scores
        self._metric = metric

    def to_long_dataframe(self) -> pd.DataFrame:
        data = dict(zip(self._models, self._fold_scores))
        wide_df = pd.DataFrame(data)
        return wide_df.melt(var_name="model", value_name=self._metric)

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def n_folds(self) -> int:
        return len(self._fold_scores[0])

    @classmethod
    def validate_params(
        cls, model_names: list[str], folds_scores: list[list[float | int]]
    ) -> None:
        if len(model_names) != len(folds_scores):
            raise ValueError(
                f"model_names and folds_scores must have the same length "
                f"(got {len(model_names)} models and {len(folds_scores)} score lists)"
            )

        if len(model_names) == 0:
            raise ValueError("At least one model is required to build BoxplotData")

        if cls._has_inconsistent_number_of_folds(folds_scores):
            models_scores = dict(zip(model_names, folds_scores))
            fold_counts = {
                model: len(scores) for model, scores in models_scores.items()
            }
            raise ValueError(
                f"All models must have the same number of folds, "
                f"got inconsistent fold counts: {fold_counts}"
            )

    @classmethod
    def _has_inconsistent_number_of_folds(
        cls, folds_scores: list[list[float | int]]
    ) -> bool:
        return any(len(scores) != len(folds_scores[0]) for scores in folds_scores)
