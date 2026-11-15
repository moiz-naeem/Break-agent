"""The train types of docs/network-v0.3.md §3, as data the generator turns into TrainSpecs.

`priority` is published but not enforced: ARS uses it to break ties and the agent may override it, because the
reward (DESIGN.md §6.6) is what really decides who should go first. `reward_class` is the lateness multiplier
that reward uses; `load` is the typical number of passengers on board, off-peak to peak.
"""

from __future__ import annotations

from typing import Literal

from .model import NetModel, TrainSpec


class TrainType(NetModel):
    code: str
    name: str
    priority: int
    length_m: int
    max_speed_kmh: int
    restart_min: int
    electric: bool
    passenger: bool
    reward_class: Literal["express", "local", "freight", "none"]
    load: tuple[int, int] = (0, 0)

    def spec(self, train_id: str) -> TrainSpec:
        return TrainSpec(
            id=train_id,
            length_m=self.length_m,
            electric=self.electric,
            passenger=self.passenger,
            max_speed_kmh=self.max_speed_kmh,
            restart_min=self.restart_min,
        )


# Restart penalties are whole minutes, so the commuter's half minute (§3) rounds up to one. Empty stock and
# engineering trains wait for step 4: both only exist to serve disruptions and possessions.
TYPES: dict[str, TrainType] = {
    t.code: t
    for t in [
        TrainType(code="IC", name="Intercity", priority=1, length_m=260, max_speed_kmh=200, restart_min=1,
                  electric=True, passenger=True, reward_class="express", load=(400, 600)),
        TrainType(code="RE", name="Regional express", priority=2, length_m=160, max_speed_kmh=160, restart_min=1,
                  electric=True, passenger=True, reward_class="local", load=(200, 300)),
        TrainType(code="S", name="Commuter", priority=2, length_m=160, max_speed_kmh=120, restart_min=1,
                  electric=True, passenger=True, reward_class="local", load=(300, 800)),
        TrainType(code="FI", name="Intermodal freight", priority=3, length_m=600, max_speed_kmh=100, restart_min=4,
                  electric=False, passenger=False, reward_class="freight"),
        TrainType(code="FB", name="Bulk freight", priority=4, length_m=740, max_speed_kmh=80, restart_min=5,
                  electric=False, passenger=False, reward_class="freight"),
    ]
}  # fmt: skip
