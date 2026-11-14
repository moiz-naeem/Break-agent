from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str
    stations: int = Field(ge=4, le=30)
    trains: int = Field(ge=1, le=100)
    mix: dict[str, float]
    factors: list[str]
    shift_ticks: int = Field(gt=0)
    event_density: float = Field(default=1.0, ge=0)
    loops: dict[int, float] = Field(default_factory=lambda: {0: 0.15, 1: 0.45, 2: 0.30, 3: 0.10})
    major_fraction: float = Field(default=0.34, ge=0, le=1)
    slack: float = Field(default=0.10, ge=0)
    rotation_fraction: float = Field(default=0.4, ge=0, le=1)
    tight_crew_fraction: float = Field(default=0.3, ge=0, le=1)

    @model_validator(mode="after")
    def validate_config(self):
        if set(self.mix) != {"express", "local", "freight"}:
            raise ValueError("mix must define express, local, and freight")
        if any(value < 0 for value in self.mix.values()) or abs(sum(self.mix.values()) - 1) > 1e-9:
            raise ValueError("mix values must be nonnegative and sum to one")
        if not self.factors or self.factors != sorted(set(self.factors)):
            raise ValueError("factors must be nonempty, unique, and sorted")
        if set(self.loops) - {0, 1, 2, 3} or any(weight < 0 for weight in self.loops.values()):
            raise ValueError("invalid loop-count distribution")
        if abs(sum(self.loops.values()) - 1) > 1e-9:
            raise ValueError("loop-count weights must sum to one")
        return self
