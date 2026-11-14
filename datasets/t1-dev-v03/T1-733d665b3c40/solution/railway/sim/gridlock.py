from __future__ import annotations


def find_gridlock(state_like) -> list[list[str]]:
    """Return cyclic SCCs from a deterministic wait-for adjacency mapping."""
    graph = state_like if isinstance(state_like, dict) else getattr(state_like, "wait_for", {})
    nodes = sorted(graph)
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    low = {}
    result = []

    def visit(v):
        nonlocal index
        indices[v] = low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, [])):
            if w not in indices:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            if len(component) >= 2:
                result.append(sorted(component))

    for node in nodes:
        if node not in indices:
            visit(node)
    return sorted(result)
