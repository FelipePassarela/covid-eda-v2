from typing import Any, Literal, Self

from imblearn.pipeline import Pipeline
from optuna.distributions import BaseDistribution
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    PositiveInt,
    field_serializer,
    model_validator,
)

from covid_tuning.common.cross_validation import TuningCV


class TuningSpec(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    pipeline: Pipeline
    param_distributions: dict[str, BaseDistribution]
    n_searches: PositiveInt
    cv: TuningCV
    n_jobs: Literal[-1] | PositiveInt | None = None
    scoring: list[str] = Field(
        default_factory=lambda: ["balanced_accuracy"], min_length=1
    )
    data_path: FilePath

    @model_validator(mode="after")
    def validate_param_distributions(self) -> Self:
        distributions = self.param_distributions

        if not distributions:
            raise ValueError("param_distributions must not be empty")
        self._validate_parameter_names(distributions)

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

    @field_serializer("cv")
    def serialize_cv(self, cv: TuningCV) -> str:
        return repr(cv)

    @field_serializer("param_distributions")
    def serialize_param_distributions(
        self, distributions: dict[str, BaseDistribution]
    ) -> dict[str, str]:
        return self._serialize_dict(distributions)

    @staticmethod
    def _serialize_dict(distributions: dict[str, Any]) -> dict[str, str]:
        return {
            parameter: repr(distribution)
            for parameter, distribution in distributions.items()
        }

    def as_serializable(self) -> dict[str, Any]:
        return self.model_dump()
