"""Tool registry for ThreatLens AI investigation agent."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    category: str
    func: Callable
    requires_auth: bool = False


# Global registry
_tools: list[ToolDefinition] = []


def register_tool(tool_def: ToolDefinition):
    """Register a tool in the global registry."""
    _tools.append(tool_def)


def get_all_tools() -> list[ToolDefinition]:
    return list(_tools)


def get_tool_by_name(name: str) -> ToolDefinition | None:
    for t in _tools:
        if t.name == name:
            return t
    return None


def get_tools_for_category(category: str) -> list[ToolDefinition]:
    return [t for t in _tools if t.category == category]


def serialize_tools() -> list[dict]:
    """Return tool definitions as dicts for LLM tool-calling."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in _tools
    ]


def execute_tool(tool_name: str, arguments: dict) -> dict[str, Any]:
    """Execute a registered tool by name."""
    tool = get_tool_by_name(tool_name)
    if not tool:
        return {"success": False, "error": f"Tool '{tool_name}' not found", "data": {}}
    try:
        result = tool.func(**arguments)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e), "data": {}}
