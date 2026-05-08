"""Tests for zurvan.agent_registry.AgentRegistry (submodule API)."""

from zurvan.agent_registry import AgentRegistry


def test_register_and_get_agent():
    registry = AgentRegistry()

    def run(task: str) -> str:
        return f"ran: {task}"

    registry.register_agent("worker", run)
    assert registry.get_agent("worker") is run
    assert registry.get_agent("worker")("hi") == "ran: hi"


def test_get_unknown_agent_returns_none():
    assert AgentRegistry().get_agent("nope") is None


def test_register_overwrites_same_name():
    registry = AgentRegistry()
    registry.register_agent("a", lambda: 1)
    registry.register_agent("a", lambda: 2)
    assert registry.get_agent("a")() == 2


def test_multiple_agents_isolated():
    registry = AgentRegistry()
    registry.register_agent("a", lambda: "A")
    registry.register_agent("b", lambda: "B")
    assert registry.get_agent("a")() == "A"
    assert registry.get_agent("b")() == "B"
