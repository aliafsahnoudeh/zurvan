"""Tests for zurvan.register_tool decorator + the global tool registries."""

import pytest

from zurvan import register_tool
from zurvan.decorators import tools, tools_by_tag


@pytest.fixture(autouse=True)
def isolate_tool_registry():
    """Snapshot and restore the global tool dicts so tests don't bleed."""
    saved_tools = dict(tools)
    saved_by_tag = {k: list(v) for k, v in tools_by_tag.items()}
    yield
    tools.clear()
    tools.update(saved_tools)
    tools_by_tag.clear()
    tools_by_tag.update(saved_by_tag)


def test_register_tool_uses_function_name_by_default():
    @register_tool()
    def my_func(x: int, y: str = "default"):
        """Does something useful."""
        return x

    assert "my_func" in tools
    entry = tools["my_func"]
    assert entry["description"] == "Does something useful."
    assert entry["function"] is my_func
    assert entry["terminal"] is False


def test_register_tool_infers_parameter_schema_from_signature():
    @register_tool()
    def with_types(name: str, age: int, active: bool = True):
        """Type-annotated function."""
        return name

    schema = tools["with_types"]["parameters"]
    assert schema["type"] == "object"
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["active"]["type"] == "boolean"
    # Required = parameters without defaults
    assert "name" in schema["required"]
    assert "age" in schema["required"]
    assert "active" not in schema["required"]


def test_register_tool_skips_action_context_and_action_agent_params():
    @register_tool()
    def with_context(action_context, action_agent, real_arg: str):
        """Has internal-only args."""
        return real_arg

    properties = tools["with_context"]["parameters"]["properties"]
    assert "real_arg" in properties
    assert "action_context" not in properties
    assert "action_agent" not in properties


def test_register_tool_supports_tags():
    @register_tool(tags=["math", "geometry"])
    def area(radius: float):
        """Compute area."""
        return radius

    assert "area" in tools_by_tag["math"]
    assert "area" in tools_by_tag["geometry"]
    assert tools["area"]["tags"] == ["math", "geometry"]


def test_register_tool_overrides_metadata():
    @register_tool(
        tool_name="custom_name",
        description="custom desc",
        terminal=True,
        parameters_override={"type": "object", "properties": {}, "required": []},
    )
    def whatever(x: str):
        """Original docstring."""
        return x

    assert "custom_name" in tools
    entry = tools["custom_name"]
    assert entry["description"] == "custom desc"
    assert entry["terminal"] is True
    assert entry["parameters"] == {"type": "object", "properties": {}, "required": []}
