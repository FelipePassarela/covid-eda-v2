from typing import Any, Self

from imblearn.pipeline import Pipeline
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    PositiveInt,
    field_serializer,
    model_validator,
)


class TuningSpec(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    pipeline: Pipeline
    param_distributions: dict[str, Any] | list[dict[str, Any]]
    n_searches: PositiveInt
    n_splits: PositiveInt = 5
    n_fold_repeats: PositiveInt = 5
    scoring: list[str] = Field(
        default_factory=lambda: ["balanced_accuracy"], min_length=1
    )
    data_path: FilePath

    @model_validator(mode="after")
    def validate_param_distributions(self) -> Self:
        distributions = self.param_distributions

        if isinstance(distributions, dict):
            if not distributions:
                raise ValueError("param_distributions must not be empty")
            self._validate_parameter_names(distributions)
            return self

        if not distributions:
            raise ValueError("param_distributions must not be empty")

        for distribution in distributions:
            if not distribution:
                raise ValueError("param_distributions entries must not be empty")
            self._validate_parameter_names(distribution)

        return self

    def _validate_parameter_names(self, distributions: dict[str, Any]) -> None:
        pipeline_parameters = self.pipeline.get_params(deep=True)
        unknown_parameters = distributions.keys() - pipeline_parameters.keys()

        if unknown_parameters:
            formatted = ", ".join(sorted(unknown_parameters))
            raise ValueError(f"unknown pipeline parameters: {formatted}")

    @field_serializer("pipeline")
    def serialize_pipeline(self, pipeline: Pipeline) -> str:
        return repr(pipeline)

    @field_serializer("param_distributions")
    def serialize_param_distributions(
        self, distributions: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, str] | list[dict[str, str]]:
        if isinstance(distributions, dict):
            return self._serialize_dict(distributions)

        return [
            self._serialize_dict(distribution_group)
            for distribution_group in distributions
        ]

    @staticmethod
    def _serialize_dict(distributions: dict[str, Any]) -> dict[str, str]:
        return {
            parameter: repr(distribution)
            for parameter, distribution in distributions.items()
        }

    def as_serializable(self) -> dict[str, Any]:
        return self.model_dump()
