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


class RandomizedSearchSpec(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    name: str = Field(min_length=1)
    pipeline: Pipeline
    param_distributions: dict[str, Any] = Field(min_length=1)
    n_searches: PositiveInt
    n_fold_repeats: PositiveInt = 5
    scoring: list[str] = Field(
        default_factory=lambda: ["balanced_accuracy"], min_length=1
    )
    data_path: FilePath

    @model_validator(mode="after")
    def validate_parameter_names(self) -> Self:
        pipeline_parameters = self.pipeline.get_params(deep=True)
        unknown_parameters = (
            self.param_distributions.keys() - pipeline_parameters.keys()
        )

        if unknown_parameters:
            formatted = ", ".join(sorted(unknown_parameters))
            raise ValueError(f"unknown pipeline parameters: {formatted}")

        return self

    @field_serializer("pipeline")
    def serialize_pipeline(self, pipeline: Pipeline) -> str:
        return repr(pipeline)

    @field_serializer("param_distributions")
    def serialize_param_distributions(
        self, distributions: dict[str, Any]
    ) -> dict[str, str]:
        return {
            parameter: repr(distribution)
            for parameter, distribution in distributions.items()
        }

    def as_serializable(self) -> dict[str, Any]:
        return self.model_dump()
