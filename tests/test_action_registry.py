"""Tests for zurvan.ActionRegistry."""

import pytest

from zurvan import Action, ActionRegistry, register_tool
from zurvan.decorators import tools, tools_by_tag


def make_action(name="t", terminal=False):
    return Action(
        name=name,
        function=lambda: name,
        description=f"action {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        terminal=terminal,
    )


@pytest.fixture
def isolated_tool_registry():
    """Snapshot/restore the global decorator registries so registry tests
    that exercise tag-loading don't bleed into one another."""
    saved_tools = dict(tools)
    saved_by_tag = {k: list(v) for k, v in tools_by_tag.items()}
    yield
    tools.clear()
    tools.update(saved_tools)
    tools_by_tag.clear()
    tools_by_tag.update(saved_by_tag)


def test_register_and_get_action():
    reg = ActionRegistry()
    a = make_action("alpha")
    reg.register(a)
    assert reg.get_action("alpha") is a


def test_get_unknown_action_returns_none():
    reg = ActionRegistry()
    assert reg.get_action("does_not_exist") is None


def test_get_actions_returns_all_registered():
    reg = ActionRegistry()
    a, b, c = make_action("a"), make_action("b"), make_action("c", terminal=True)
    reg.register(a)
    reg.register(b)
    reg.register(c)
    actions = reg.get_actions()
    assert {a.name for a in actions} == {"a", "b", "c"}


def test_register_same_name_overwrites():
    reg = ActionRegistry()
    first = make_action("dup")
    second = make_action("dup")
    reg.register(first)
    reg.register(second)
    assert reg.get_action("dup") is second
    assert len(reg.get_actions()) == 1


# ── Tag-based loading via the decorator registry ─────────────────────────


def test_constructor_loads_tools_matching_tags(isolated_tool_registry):
    """``ActionRegistry(tags=[...])`` pulls every decorated tool whose tags
    intersect with the requested tags into the registry."""

    @register_tool(tags=["math"])
    def adder(x: int, y: int):
        """Adds two integers."""
        return x + y

    @register_tool(tags=["string"])
    def shouter(s: str):
        """Uppercase a string."""
        return s.upper()

    reg = ActionRegistry(tags=["math"])

    assert reg.get_action("adder") is not None
    assert reg.get_action("adder").execute(x=1, y=2) == 3
    # Tools without the requested tag are NOT loaded
    assert reg.get_action("shouter") is None


def test_constructor_with_empty_tags_loads_nothing(isolated_tool_registry):
    @register_tool(tags=["math"])
    def adder(x: int, y: int):
        """Adds."""
        return x + y

    reg = ActionRegistry()  # no tags
    assert reg.get_action("adder") is None


def test_register_from_tools_loads_all_when_no_tags(isolated_tool_registry):
    @register_tool(tags=["math"])
    def adder(x: int, y: int):
        """Adds."""
        return x + y

    @register_tool(tags=["string"])
    def shouter(s: str):
        """Uppercase."""
        return s.upper()

    reg = ActionRegistry()
    reg.register_from_tools()  # no tag filter

    assert reg.get_action("adder") is not None
    assert reg.get_action("shouter") is not None


def test_register_from_tools_filters_by_tags(isolated_tool_registry):
    @register_tool(tags=["math"])
    def adder(x: int, y: int):
        """Adds."""
        return x + y

    @register_tool(tags=["string"])
    def shouter(s: str):
        """Uppercase."""
        return s.upper()

    reg = ActionRegistry()
    reg.register_from_tools(tags=["string"])

    assert reg.get_action("adder") is None
    assert reg.get_action("shouter") is not None


def test_constructor_does_not_double_register_when_tag_overlaps(isolated_tool_registry):
    """A tool tagged with both 'a' and 'b' is registered once even when
    both tags are requested."""

    @register_tool(tags=["a", "b"])
    def both(s: str):
        """Has multiple tags."""
        return s

    reg = ActionRegistry(tags=["a", "b"])
    assert len(reg.get_actions()) == 1
