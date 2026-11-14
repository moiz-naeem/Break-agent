"""Observer (T05 §2.3): what an agent sees, fetched through a client. Static info is read once."""

from __future__ import annotations

from railway.sim.observe import State, StaticInfo


class Observer:
    def __init__(self, client):
        self.client = client
        self.static = StaticInfo.from_tool_json(client.call("network", {}), client.call("timetable", {}))

    def observe(self) -> tuple[StaticInfo, State]:
        return self.static, State.model_validate_json(self.client.call("status", {}))
