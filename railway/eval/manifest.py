"""Run manifests `runs/<name>.yaml` (T08 §2.6; DESIGN.md §9.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Strict):
    id: str  # label used in tables and charts
    name: str
    base_url: str | None = None  # loop only
    api_key_env: str | None = None  # loop only; the NAME of an env var, never the key
    harbor_model: str | None = None  # harbor only (-m value)


class ScenarioSpec(_Strict):
    preset: str
    set: str
    count: int = Field(gt=0)


class Manifest(_Strict):
    name: str
    harness: Literal["loop", "harbor"]
    mode: Literal["windowed", "full"] = "windowed"  # loop only
    model: ModelSpec
    temperature: float | None = None  # null = the model's default
    max_tokens: int = Field(1024, gt=0)
    dataset: str | None = None
    scenarios: ScenarioSpec | None = None
    reps: int = Field(3, gt=0)
    concurrency: int = Field(4, gt=0)
    tools_disabled: list[str] = []
    diagnostics: bool = False

    @model_validator(mode="after")
    def _check(self) -> Manifest:
        if (self.dataset is None) == (self.scenarios is None):
            raise ValueError("give exactly one of `dataset` or `scenarios`")
        if self.harness == "harbor":
            if self.dataset is None:
                raise ValueError("harness `harbor` requires `dataset`")
            if not self.model.harbor_model:
                raise ValueError("harness `harbor` requires `model.harbor_model`")
        elif not self.model.base_url:
            raise ValueError("harness `loop` requires `model.base_url`")
        return self


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a manifest must be a YAML mapping")
    manifest = Manifest.model_validate(data)
    if manifest.name != path.stem:
        raise ValueError(f"{path}: name {manifest.name!r} must equal the file stem {path.stem!r}")
    return manifest
