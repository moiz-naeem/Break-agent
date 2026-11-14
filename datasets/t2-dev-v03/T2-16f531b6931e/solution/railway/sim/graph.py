"""The network as a graph: stations and terminals are nodes, sections are edges.

A train never runs back over the section it has just used, so every search here is over
(node, section used to arrive) states. Everything iterates in sorted order: results never depend on
dict or set iteration order.
"""

from __future__ import annotations

from collections import deque

AMBIGUOUS = "ambiguous"


class Graph:
    def __init__(self, network):
        self.chain = network.yards is not None
        self.stations = {s.id: s for s in network.stations}
        # A legacy yard (chain scenarios) has unlimited capacity and no tracks: value None.
        self.terminals = {y: None for y in (network.yards or {}).values()} | {t.id: t for t in network.terminals}
        self.sections = {s.id: s for s in network.sections}
        self.nodes = [*self.terminals, *self.stations]
        self.adj: dict[str, list[tuple[str, str]]] = {n: [] for n in self.nodes}
        for s in network.sections:
            self.adj[s.west].append((s.id, s.east))
            self.adj[s.east].append((s.id, s.west))
        for hops in self.adj.values():
            hops.sort()
        self.tracks: dict[str, dict] = {s.id: {loop.id: loop for loop in s.loops} for s in network.stations}
        for tid, terminal in self.terminals.items():
            if terminal is not None:
                self.tracks[tid] = {track.id: track for track in terminal.tracks}

    def unlimited(self, node: str) -> bool:
        return node in self.terminals and self.terminals[node] is None

    def other_end(self, section_id: str, node: str) -> str:
        s = self.sections[section_id]
        return s.east if node == s.west else s.west

    def _leg(self, start: str, dest: str, last_section: str | None, strict: bool = True):
        """Shortest walk start -> dest by number of sections, not starting with `last_section` and not through
        a terminal. Returns [(section, node), ...], None (unreachable) or AMBIGUOUS (two shortest walks)."""
        if start == dest:
            return None
        dist = {(start, last_section): 0}
        ways = {(start, last_section): 1}
        prev: dict = {}
        queue = deque([(start, last_section)])
        best = None
        while queue:
            node, came = queue.popleft()
            d = dist[(node, came)]
            if best is not None and d >= best:
                continue
            if node in self.terminals and node != start:
                continue
            for section, nxt in self.adj[node]:
                if section == came:
                    continue
                state = (nxt, section)
                if state not in dist:
                    dist[state], ways[state], prev[state] = d + 1, ways[(node, came)], (node, came)
                    queue.append(state)
                    if nxt == dest and best is None:
                        best = d + 1
                elif dist[state] == d + 1:
                    ways[state] += ways[(node, came)]
        if best is None:
            return None
        ends = sorted(s for s in dist if s[0] == dest and dist[s] == best)
        if strict and (len(ends) > 1 or ways[ends[0]] > 1):
            return AMBIGUOUS
        hops, state = [], ends[0]
        while state != (start, last_section):
            hops.append((state[1], state[0]))
            state = prev[state]
        return hops[::-1]

    def walk(self, start: str, dest: str, last_section: str | None = None, via=(), strict: bool = True):
        """The walk start -> via... -> dest as [(section, node), ...], None or AMBIGUOUS."""
        hops, here, came = [], start, last_section
        for target in [*via, dest]:
            leg = self._leg(here, target, came, strict)
            if leg is None or leg == AMBIGUOUS:
                return leg
            hops.extend(leg)
            here, came = target, leg[-1][0]
        return hops

    def reach(self, states: set[tuple[str, str | None]], target: str) -> set[tuple[str, str | None]]:
        """Every (target, arriving section) state reachable from `states` (a state at `target` counts)."""
        seen, queue = set(states), deque(sorted(states, key=lambda s: (s[0], s[1] or "")))
        found = {s for s in states if s[0] == target}
        start_nodes = {s[0] for s in states}
        while queue:
            node, came = queue.popleft()
            if node in self.terminals and node not in start_nodes:
                continue
            for section, nxt in self.adj[node]:
                state = (nxt, section)
                if section == came or state in seen:
                    continue
                seen.add(state)
                if nxt == target:
                    found.add(state)
                queue.append(state)
        return found

    def path_nodes(self, start: str, path: list[str]) -> list[str]:
        """The nodes a resource path visits after `start`, one per section in it."""
        nodes, here = [], start
        for resource in path:
            if resource in self.sections:
                here = self.other_end(resource, here)
                nodes.append(here)
        return nodes
