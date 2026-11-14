"""OpenAI function specs built from the Session tool registry (T08 §2.2).

The registry belongs to T05 (`railway.tools.session.REGISTRY`: name -> (pydantic args model, handler)).
Names and descriptions come from it, so they match the MCP server's.
"""

from __future__ import annotations


def _registry():
    from railway.tools import session

    return session.REGISTRY


def _description(name: str, model, handler) -> str:
    for source in (handler, model):
        doc = (getattr(source, "__doc__", None) or "").strip()
        if doc:
            return doc.splitlines()[0].strip()
    return f"{name}. See manual()."


def _parameters(model) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema.setdefault("properties", {})
    schema["additionalProperties"] = False
    return schema


def tool_schemas(disabled=(), registry=None) -> list[dict]:
    """One `{"type": "function", ...}` spec per registry tool, in registry order, minus `disabled`."""
    registry = _registry() if registry is None else registry
    skip = set(disabled)
    specs = []
    for name, entry in registry.items():
        if name in skip:
            continue
        model, handler = entry[0], entry[1]
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _description(name, model, handler),
                    "parameters": _parameters(model),
                },
            }
        )
    return specs
