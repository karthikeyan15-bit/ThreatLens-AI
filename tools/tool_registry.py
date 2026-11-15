"""Tool registry for ThreatLens AI investigation agent.

Every registered tool must expose a `run(args: dict) -> dict` entry point.
The registry validates arguments against a JSON Schema before execution,
logs every call, and wraps execution in safe error handling.

Model-generated arguments can never cause arbitrary code execution:
only registered tool functions are callable, and all arguments are
validated against a strict JSON Schema.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from config.settings import TOOL_TIMEOUT_SECONDS

log = logging.getLogger("threatlens.tools")


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


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> tuple[bool, str]:
    """Validate arguments against a JSON Schema.

    Checks:
        - Required fields present
        - No unknown fields (strict mode)
        - Type checking for known fields
        - Enum constraints

    Returns (valid, error_message).
    """
    if not isinstance(arguments, dict):
        return False, "Arguments must be a JSON object"

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    allowed_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # Required fields
    for req in required:
        if req not in arguments:
            return False, f"Missing required argument: {req}"

    # Unknown fields (strict mode — reject extras)
    for key in arguments:
        if key not in properties:
            return False, f"Unknown argument: '{key}' is not a valid parameter"

    # Type checking
    for key, value in arguments.items():
        prop = properties.get(key, {})
        expected_type = prop.get("type")
        if expected_type and expected_type in allowed_types:
            allowed = allowed_types[expected_type]
            if not isinstance(value, allowed):
                return False, (
                    f"Argument '{key}' must be {expected_type}, "
                    f"got {type(value).__name__}"
                )

        # Enum constraints
        if "enum" in prop and value not in prop["enum"]:
            return False, (
                f"Argument '{key}' must be one of {prop['enum']}, "
                f"got {value}"
            )

        # Minimum length for strings
        if isinstance(value, str) and "minLength" in prop:
            if len(value) < prop["minLength"]:
                return False, (
                    f"Argument '{key}' must be at least "
                    f"{prop['minLength']} characters"
                )

    return True, ""


def execute_tool(tool_name: str, arguments: dict) -> dict[str, Any]:
    """Execute a registered tool by name with full validation and logging.

    Returns a dict with:
        success: bool
        data: tool result (or empty dict on failure)
        error: error message (or None)
        duration_ms: execution time
        source: inference source label (if applicable)
    """
    tool = get_tool_by_name(tool_name)
    if not tool:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found",
            "data": {},
            "duration_ms": 0,
        }

    # Validate arguments against JSON Schema
    valid, error = validate_arguments(tool.parameters, arguments)
    if not valid:
        log.warning("Tool argument validation failed: %s = %s → %s", tool_name, arguments, error)
        return {
            "success": False,
            "error": error,
            "data": {},
            "duration_ms": 0,
        }

    start = time.time()
    try:
        result = tool.func(arguments)
        duration_ms = int((time.time() - start) * 1000)
        log.info("Tool %s executed in %dms", tool_name, duration_ms)
        return {
            "success": True,
            "data": result,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.time() - start) * 1000)
        log.error("Tool %s failed after %dms: %s", tool_name, duration_ms, exc)
        return {
            "success": False,
            "error": str(exc),
            "data": {},
            "duration_ms": duration_ms,
        }