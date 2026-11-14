from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .constants import RULES_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Loop(StrictModel):
    id: str
    length_m: Literal[150, 300, 700]
    kind: Literal["platform", "goods"]


class Station(StrictModel):
    id: str
    major: bool
    crew_station: bool
    loops: list[Loop] = Field(max_length=3)

    @model_validator(mode="after")
    def unique_loops(self):
        if len({x.id for x in self.loops}) != len(self.loops):
            raise ValueError("duplicate loop id")
        if any(x.id not in {"L1", "L2", "L3"} for x in self.loops):
            raise ValueError("invalid loop id")
        return self


class Section(StrictModel):
    id: str
    west: str
    east: str
    length_m: int = Field(gt=0)
    line_speed_kmh: int = Field(gt=0)


class Network(StrictModel):
    yards: dict[str, str]
    stations: list[Station]
    sections: list[Section]

    @model_validator(mode="after")
    def ordered_chain(self):
        ids = ["YW", *[s.id for s in self.stations], "YE"]
        if self.yards != {"west": "YW", "east": "YE"}:
            raise ValueError("yards must be YW and YE")
        if [s.id for s in self.stations] != [f"S{i}" for i in range(1, len(self.stations) + 1)]:
            raise ValueError("stations must be ordered S1..SN")
        expected = list(zip(ids, ids[1:], strict=False))
        actual = [(s.west, s.east) for s in self.sections]
        if actual != expected or any(s.id != f"{w}-{e}" for s, (w, e) in zip(self.sections, expected, strict=False)):
            raise ValueError("sections must form the ordered contiguous chain")
        return self


class Stop(StrictModel):
    station: str
    arr: int = Field(ge=0)
    dep: int | None
    min_dwell: int = Field(ge=0, le=2)

    @model_validator(mode="after")
    def valid_times(self):
        if self.dep is not None and self.arr > self.dep:
            raise ValueError("arr must not exceed dep")
        if self.dep is None and self.min_dwell != 0:
            raise ValueError("destination dwell must be 0")
        return self


class Train(StrictModel):
    id: str
    type: Literal["express", "local", "freight", "engineering"]
    direction: Literal["east", "west"]
    pax: int = Field(ge=0)
    entry_tick: int = Field(ge=0)
    stops: list[Stop] = Field(min_length=1)
    next_service: str | None
    crew_used_at_entry: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_train(self):
        expected = {"express": "E", "local": "L", "freight": "F", "engineering": "N"}[self.type]
        if not self.id.startswith(expected) or not self.id[len(expected) :].isdigit():
            raise ValueError("train id does not match type")
        if self.type in {"freight", "engineering"} and self.pax != 0:
            raise ValueError("non-passenger train pax must be 0")
        if self.stops[-1].station != ("YE" if self.direction == "east" else "YW"):
            raise ValueError("last stop must be destination yard")
        if self.stops[-1].dep is not None or self.stops[-1].min_dwell != 0:
            raise ValueError("destination stop must have null dep and zero dwell")
        if any(not 1 <= stop.min_dwell <= 2 for stop in self.stops[:-1]):
            raise ValueError("intermediate stop dwell must be 1 or 2")
        if any(s.dep is None for s in self.stops[:-1]):
            raise ValueError("only final stop may have null dep")
        if self.type == "freight" and len(self.stops) != 1:
            raise ValueError("freight has no intermediate stops")
        return self


class Event(StrictModel):
    id: str
    kind: str
    reveal_tick: int
    model_config = ConfigDict(extra="allow")


class Scenario(StrictModel):
    schema_version: Literal[1]
    rules_version: str
    scenario_id: str
    seed: int
    preset: str
    shift_ticks: int = Field(gt=0)
    factors: list[str]
    network: Network
    trains: list[Train]
    events: list[dict]

    @model_validator(mode="after")
    def valid_scenario(self):
        if self.rules_version != RULES_VERSION:
            raise ValueError("rules_version mismatch")
        if self.factors != sorted(set(self.factors)):
            raise ValueError("factors must be unique and sorted")
        if [t.id for t in self.trains] != sorted(t.id for t in self.trains):
            raise ValueError("trains must be sorted by id")
        ids = {t.id for t in self.trains}
        if len(ids) != len(self.trains):
            raise ValueError("duplicate train id")
        for t in self.trains:
            if t.next_service and t.next_service not in ids:
                raise ValueError("next_service points to missing train")
            route = ["YW", *[s.id for s in self.network.stations], "YE"]
            if t.direction == "west":
                route.reverse()
            indices = [route.index(stop.station) if stop.station in route else -1 for stop in t.stops]
            if -1 in indices or indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("train stops must be in route order")
            if t.next_service and next(x for x in self.trains if x.id == t.next_service).direction == t.direction:
                raise ValueError("next_service must be opposite direction")
        return self


class Authorize(StrictModel):
    kind: Literal["authorize"] = "authorize"
    train: str
    station: str
    track: str


class CancelAuthority(StrictModel):
    kind: Literal["cancel_authority"] = "cancel_authority"
    train: str


class Hold(StrictModel):
    kind: Literal["hold"] = "hold"
    train: str
    until_tick: int


class RelieveCrew(StrictModel):
    kind: Literal["relieve_crew"] = "relieve_crew"
    train: str


class Reverse(StrictModel):
    kind: Literal["reverse"] = "reverse"
    train: str


Action = Annotated[Authorize | CancelAuthority | Hold | RelieveCrew | Reverse, Field(discriminator="kind")]
ActionAdapter = TypeAdapter(Action)
