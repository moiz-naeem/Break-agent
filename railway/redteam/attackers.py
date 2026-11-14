"""v0.3 red-team attackers: legal dispatchers that abuse one new mechanic each.

Every attacker plays through a `Client`, exactly as an agent does: it sees `static` and `state` as the tools
report them and never the scenario. Two families:

- **greedy-based** — a working dispatcher that deliberately abuses one v0.3 surface: misrouting to a foreign
  terminal, skipping stops by taking a bypass, parking on station mains, a balloon-loop `via`, farming crew
  timeouts, or jamming a terminal's tracks so other trains cannot enter;
- **plan-based** — the task's own best plan (the 1.0 trajectory) with one kind of call removed or rewritten.
  That answers "does skipping this pay?" against the best known plan instead of against greedy, which is the
  harder test: if dropping every crew relief from the optimal plan makes it cheaper, the relief is mispriced.

`reverse` is gone in v0.3 (docs/design-changes-v0.3.md §3.1), so v0.2's ReverseAbuse is dropped.
"""

from __future__ import annotations

from railway.policies.baselines import Greedy, Policy, ToolCall, waiting_trains
from railway.policies.routes import authorize, is_terminal
from railway.ref.reference import Reference
from railway.sim.engine import Simulator
from railway.sim.graph import AMBIGUOUS
from railway.tools.observe import Observer

PAX = ("express", "local")


# ---------------------------------------------------------------- route building on a graph


def _expand(belief: Simulator, tid: str, targets: list[str]) -> list[str] | None:
    """The explicit node chain for a walk through `targets`, or None if a leg cannot be walked.

    Returned as nodes in order, so the last one is the destination and the rest are the `via` an agent would
    name. Legs are expanded one by one with `strict=False`, so an ambiguous leg still yields a concrete route
    the interlocking accepts; a train never reverses over the section it arrived by.
    """
    train = belief._trains.get(tid)
    if train is None:
        return None
    here, came = belief._heading(train)
    nodes: list[str] = []
    for target in targets:
        hops = belief._graph.walk(here, target, came, strict=False)
        if hops is None or hops == AMBIGUOUS or not hops:
            return None
        nodes.extend(node for _, node in hops)
        here, came = target, hops[-1][0]
    return nodes


def _action(belief: Simulator, tid: str, targets: list[str], track: str):
    """An `authorize` for the walk through `targets`, with every intermediate node named in `via`."""
    nodes = _expand(belief, tid, targets)
    if not nodes:
        return None
    return authorize(tid, nodes[-1], track, nodes[:-1])


def _tracks(static, node: str) -> list[str]:
    """Every track an agent may ask for at `node`, mains included."""
    for s in static.network["stations"]:
        if s["id"] == node:
            return ["main", *sorted(loop["id"] for loop in s["loops"])]
    for t in static.network.get("terminals", []):
        if t["id"] == node:
            return sorted(track["id"] for track in t["tracks"])
    return ["main"]


def _first_ok(belief: Simulator, static, tid: str, targets: list[str], tracks: list[str] | None = None):
    for track in tracks or _tracks(static, targets[-1]):
        action = _action(belief, tid, targets, track)
        if action is not None and belief.check(action).ok:
            return action
    return None


def _terminals(static) -> list[str]:
    return [t["id"] for t in static.network.get("terminals", [])] or ["YW", "YE"]


def find_cycle(static) -> list[str]:
    """A cycle of nodes (each adjacent to the next, and the last back to the first), or [].

    A v0.3 T2 map has a bypass, so trunk + bypass form a cycle: a train can be sent round it and come back,
    the "balloon loop" move the engine allows. Depth-first over the undirected graph, edges used once.
    """
    adj: dict[str, list[tuple[str, str]]] = {}
    for section in static.network["sections"]:
        adj.setdefault(section["west"], []).append((section["id"], section["east"]))
        adj.setdefault(section["east"], []).append((section["id"], section["west"]))
    for node in adj:
        adj[node].sort()
    for start in sorted(adj):
        stack = [(start, None, [start])]
        seen = {(start, None)}
        while stack:
            node, came, path = stack.pop()
            for section, nxt in adj[node]:
                if section == came:
                    continue
                if nxt == start and len(path) > 2:
                    return path
                if (nxt, section) in seen or nxt in path:
                    continue
                seen.add((nxt, section))
                stack.append((nxt, section, [*path, nxt]))
    return []


def _ahead(static, state, tid: str) -> list[str]:
    """The nodes still ahead of `tid` on its timetabled route (its own destination last)."""
    train = next(t for t in static.trains if t["id"] == tid)
    route = list(train.get("route") or [])
    here = next((t.station for t in state.trains if t.id == tid), None)
    if here in route:
        return route[route.index(here) + 1 :]
    return route[1:]


# ---------------------------------------------------------------- greedy-based attackers


class HoldAll(Policy):
    """Hold every train to the shift end, then only advance: `hold` must never be rewarded."""

    def play(self, client) -> None:
        static, _ = Observer(client).observe()
        for t in static.trains:
            client.call("hold", {"train": t["id"], "until_tick": static.shift_ticks})
        while True:
            client.call("advance_to_next_event", {})


class PassengersOnly(Greedy):
    """Greedy that never moves freight: serve the weighted trains, abandon the cargo."""

    def choose(self, static, state, belief):
        freight = {t.id for t in state.trains if t.type == "freight"}
        for action in super().choose(static, state, belief):
            if action.train not in freight:
                yield action


class PartialFreightHold(Greedy):
    """Greedy that moves every second freight train: the "hold just enough freight" trade, which the
    undelivered-freight charge (1500) is meant to make unprofitable at any share."""

    def choose(self, static, state, belief):
        freight = sorted(t.id for t in state.trains if t.type == "freight")
        held = set(freight[::2])
        for action in super().choose(static, state, belief):
            if action.train not in held:
                yield action


class MisrouteAll(Greedy):
    """Send every train to a terminal that is not its destination: a misroute plus a wrong terminal, which
    must cost more than serving the stops (500 each) it loses."""

    def choose(self, static, state, belief):
        for train in sorted(waiting_trains(state), key=lambda t: (t.entry_tick, t.id)):
            wrong = [t for t in _terminals(static) if t != train.destination]
            action = next(filter(None, (_first_ok(belief, static, train.id, [t]) for t in wrong)), None)
            if action is None:
                action = self._first_granted(static, belief, train)
            if action is not None:
                belief.apply(action)
                yield action


class BypassSkip(Greedy):
    """Always authorize the farthest node still grantable on the route, skipping every station in between:
    fast running, missed calls. Checks that a bypassed stop is never cheaper than serving it."""

    def choose(self, static, state, belief):
        for train in sorted(waiting_trains(state), key=lambda t: (t.entry_tick, t.id)):
            action = None
            for node in reversed(_ahead(static, state, train.id)):
                loops_only = not is_terminal(static, node)
                tracks = [k for k in _tracks(static, node) if not (loops_only and k == "main")]
                action = _first_ok(belief, static, train.id, [node], tracks)
                if action is not None:
                    break
            if action is None:
                action = self._first_granted(static, belief, train)
            if action is not None:
                belief.apply(action)
                yield action


class LongAuthHold(Greedy):
    """The farthest grantable node, on the station main. A long authority is never released early and a train
    standing with an authority is not "stuck", so this is the attack on the deadlock check: provoke a
    permanent stand-off that the gridlock charge might miss, while blocking every main on the way."""

    def choose(self, static, state, belief):
        for train in sorted(waiting_trains(state), key=lambda t: (t.entry_tick, t.id)):
            action = None
            for node in reversed(_ahead(static, state, train.id)):
                tracks = _tracks(static, node) if is_terminal(static, node) else ["main"]
                action = _first_ok(belief, static, train.id, [node], tracks)
                if action is not None:
                    break
            if action is not None:
                belief.apply(action)
                yield action


class BalloonVia(Greedy):
    """Every authority goes once round the bypass cycle before carrying on (a balloon loop). The authority
    then holds the same section twice, which is where a reservation could be released too early."""

    def choose(self, static, state, belief):
        cycle = find_cycle(static)
        for train in sorted(waiting_trains(state), key=lambda t: (t.entry_tick, t.id)):
            action = None
            if cycle:
                for i in range(len(cycle)):
                    loop = [*cycle[i:], *cycle[:i], cycle[i]]
                    for node in _ahead(static, state, train.id):
                        action = _first_ok(belief, static, train.id, [*loop, node])
                        if action is not None:
                            break
                    if action is not None:
                        break
            if action is None:
                action = self._first_granted(static, belief, train)
            if action is not None:
                belief.apply(action)
                yield action


class TerminalJam(Greedy):
    """Park trains on the tracks of a terminal that is not theirs (the smallest one first), so trains whose
    origin or destination it is wait outside (ENTRY_BLOCKED) or never finish."""

    def choose(self, static, state, belief):
        terminals = sorted(
            static.network.get("terminals", []), key=lambda t: (len(t["tracks"]), t["id"])
        )
        for train in sorted(waiting_trains(state), key=lambda t: (t.entry_tick, t.id)):
            action = None
            for terminal in terminals:
                if terminal["id"] == train.destination:
                    continue
                action = _first_ok(belief, static, train.id, [terminal["id"]])
                if action is not None:
                    break
            if action is None:
                action = self._first_granted(static, belief, train)
            if action is not None:
                belief.apply(action)
                yield action


class CrewTimeoutFarm(Greedy):
    """Hold every train until its crew has timed out, then dispatch greedily: each timeout costs 40 plus a
    45-tick block, so farming them must always lose to relieving (8 ticks)."""

    def __init__(self):
        self.farmed: set[str] = set()

    def decide(self, static, state) -> list[ToolCall]:
        holds = []
        for train in waiting_trains(state):
            if train.id in self.farmed:
                continue
            if train.crew_remaining is None:  # F10 is off: nothing to farm, dispatch normally
                self.farmed.add(train.id)
                continue
            if train.crew_remaining <= 0 or train.blocked_until is not None:
                self.farmed.add(train.id)
                continue
            holds.append(ToolCall("hold", {"train": train.id, "until_tick": state.tick + train.crew_remaining + 1}))
        return [*holds, *super().decide(static, state)]

    def choose(self, static, state, belief):
        for action in super().choose(static, state, belief):
            if action.train in self.farmed:
                yield action


class NoRelief(Policy):
    """The reference with every crew relief dropped: the cheapest way to test the 40-point timeout charge
    against a dispatcher that is otherwise good."""

    def play(self, client) -> None:
        reference, observer = Reference(), Observer(client)
        while True:
            for call in reference.decide(*observer.observe()):
                if call.tool != "relieve_crew":
                    client.call(call.tool, call.args)


# ---------------------------------------------------------------- plan-based attackers


class PlanPolicy(Policy):
    """Replay a stored plan (the task's best, which scores 1.0) with one mutation.

    `drop(static, tool, args)` removes calls; `rewrite(static, tool, args)` changes them; `churn` cancels and
    re-grants every authority that was granted, which must cost nothing.

    A churn run records every re-grant the interlocking refused in `regrant_refusals`. That is not a cost
    finding: an authority the plan was granted can stop being grantable the moment it is cancelled (route
    resolution depends on state the agent cannot see), so the run desynchronises from the plan and the
    cost-neutrality check cannot be read at all. The battery reports those runs as inconclusive.
    """

    def __init__(self, plan: list[dict], *, drop=None, rewrite=None, churn: bool = False):
        self.plan, self.drop, self.rewrite, self.churn = plan, drop, rewrite, churn
        self.regrant_refusals: list[str] = []

    def play(self, client) -> None:
        static, _ = Observer(client).observe()
        for call in self.plan:
            tool, args = call["tool"], dict(call["args"])
            if self.drop and self.drop(static, tool, args):
                continue
            if self.rewrite:
                tool, args = self.rewrite(static, tool, args)
            out = client.call(tool, args)
            if self.churn and tool == "authorize" and out.startswith("GRANTED"):
                client.call("cancel_authority", {"train": args["train"]})
                again = client.call(tool, args)
                if not again.startswith("GRANTED"):
                    self.regrant_refusals.append(f"{args} -> {again.strip().splitlines()[0]}")
        client.call("end_shift", {})

def _types(static) -> dict[str, str]:
    return {t["id"]: t["type"] for t in static.trains}


def _is_freight(static, args) -> bool:
    return _types(static).get(args.get("train")) == "freight"


def plan_attackers(plan: list[dict]) -> dict[str, PlanPolicy]:
    """The best plan, mutated. Each must cost at least what the plan costs (reward at most 1)."""

    def drop_relief(_static, tool, _args):
        return tool == "relieve_crew"

    def drop_freight(static, tool, args):
        return tool in {"authorize", "relieve_crew"} and _is_freight(static, args)

    def drop_half_freight(static, tool, args):
        if not (tool in {"authorize", "relieve_crew"} and _is_freight(static, args)):
            return False
        freight = sorted(tid for tid, kind in _types(static).items() if kind == "freight")
        return args.get("train") in set(freight[::2])

    def to_main(static, tool, args):
        if tool != "authorize":
            return tool, args
        stations = {s["id"] for s in static.network["stations"]}
        if args.get("station") in stations:
            args = {**args, "track": "main"}
        return tool, args

    return {
        "PlanNoRelief": PlanPolicy(plan, drop=drop_relief),
        "PlanDropFreight": PlanPolicy(plan, drop=drop_freight),
        "PlanHalfFreight": PlanPolicy(plan, drop=drop_half_freight),
        "PlanMainTracks": PlanPolicy(plan, rewrite=to_main),
    }


def plan_neutral(plan: list[dict]) -> dict[str, PlanPolicy]:
    """Mutations that must change nothing at all: cancelling and re-granting the plan's own authorities."""
    return {"PlanCancelChurn": PlanPolicy(plan, churn=True)}


GREEDY_ATTACKERS = {
    cls.__name__: cls
    for cls in (
        HoldAll,
        PassengersOnly,
        PartialFreightHold,
        MisrouteAll,
        BypassSkip,
        LongAuthHold,
        BalloonVia,
        TerminalJam,
        CrewTimeoutFarm,
    )
}
SLOW_ATTACKERS = {cls.__name__: cls for cls in (NoRelief,)}  # reference-based: minutes per T2 episode
