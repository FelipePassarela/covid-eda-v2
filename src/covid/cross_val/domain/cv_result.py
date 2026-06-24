import pandas as pd


class CVResults:
    def __init__(self) -> None:
        self._summary = {}

    def add_fold_metrics(
        self, model_name: str, fold: int, metrics: dict[str, float]
    ) -> None:
        if model_name not in self._summary:
            self._summary[model_name] = []
        self._summary[model_name].append({"fold": fold, **metrics})

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for model_name, folds in self._summary.items():
            for fold_metrics in folds:
                records.append({"model": model_name, **fold_metrics})
        return pd.DataFrame(records)

    def summarize(self, sort_by: str | None = None) -> pd.DataFrame:
        df = self.to_dataframe()
        stats = ["mean", "std"]
        df = df.groupby("model").agg(
            {col: stats for col in df.columns if col not in ["model", "fold"]}
        )
        if sort_by:
            df = df.sort_values(by=[(sort_by, "mean")], ascending=False)  # type: ignore
        return df
