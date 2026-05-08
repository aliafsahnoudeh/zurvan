"""Lock down the public API surface.

Every name in ``zurvan.__all__`` must resolve and be of the right kind.
This guards against accidental removals or import-breakage in
``__init__.py`` — if a symbol vanishes, this test screams.

Also includes basic smoke tests for primitives that don't have their
own dedicated test file (Goal, Prompt, Environment, LogLevel, Logger).
"""

import re
from enum import Enum

import pytest

import zurvan


EXPECTED_PUBLIC_API = {
    "__version__",
    # Core GAME primitives
    "Action",
    "ActionContext",
    "ActionRegistry",
    "Agent",
    "Capability",
    "Environment",
    "Goal",
    "Memory",
    "Prompt",
    # Logging
    "LogLevel",
    "Logger",
    # Tool decorator
    "register_tool",
    # Token tracking
    "TokenTracker",
    # Agent languages
    "AgentLanguage",
    "AgentFunctionCallingActionLanguage",
    "AgentFunctionCallingActionLanguageGemini",
    "AgentFunctionCallingActionLanguageGroq",
    "AgentFunctionCallingActionLanguageOpenAI",
    # Capabilities
    "CanaryCapability",
    "PromptInjectionDetected",
    "TimeAwareCapability",
}


def test_all_matches_expected_set():
    """If you intentionally add or remove a public symbol, update this test
    too — that's the trigger for a deliberate API change."""
    assert set(zurvan.__all__) == EXPECTED_PUBLIC_API


@pytest.mark.parametrize("name", sorted(EXPECTED_PUBLIC_API))
def test_each_public_name_is_resolvable(name):
    assert hasattr(zurvan, name), f"zurvan.{name} is missing"
    assert getattr(zurvan, name) is not None


def test_version_is_semver_like():
    assert re.match(r"^\d+\.\d+\.\d+", zurvan.__version__)


# ── Smoke tests for primitives without dedicated test files ──────────────


def test_goal_is_frozen_dataclass():
    g = zurvan.Goal(priority=1, name="g", description="d")
    with pytest.raises(Exception):  # FrozenInstanceError or TypeError
        g.priority = 99


def test_prompt_constructs_with_defaults():
    p = zurvan.Prompt()
    assert p.messages == []
    assert p.tools == []
    assert p.metadata == {}


def test_environment_executes_action_and_formats_result():
    env = zurvan.Environment()
    action = zurvan.Action(
        name="add", function=lambda a, b: a + b, description="add",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    result = env.execute_action(action, {"a": 2, "b": 3})
    assert result["tool_executed"] is True
    assert result["result"] == 5
    assert "timestamp" in result


def test_environment_traps_action_exceptions():
    env = zurvan.Environment()
    action = zurvan.Action(
        name="boom", function=lambda: 1 / 0, description="boom",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    result = env.execute_action(action, {})
    assert result["tool_executed"] is False
    assert "error" in result
    assert "traceback" in result


def test_loglevel_is_strenum():
    assert isinstance(zurvan.LogLevel.INFO, str)
    assert issubclass(zurvan.LogLevel, Enum)


def test_default_logger_is_no_op():
    logger = zurvan.Logger()
    # Must not raise on any call
    logger.log("any", level=zurvan.LogLevel.DEBUG)
    logger.log("any", level=zurvan.LogLevel.ERROR, env=object())


def test_capability_is_concrete_base_with_default_hooks():
    """Capability is a concrete base class — every hook has a no-op or
    pass-through default, so subclasses override only what they need."""

    class Minimal(zurvan.Capability):
        def __init__(self):
            super().__init__(name="m", description="m")

    cap = Minimal()
    assert cap.start_agent_loop(None, None) is True
    assert cap.process_prompt(None, None, "p") == "p"
    assert cap.process_response(None, None, "r") == "r"
    assert cap.process_action(None, None, {"x": 1}) == {"x": 1}
    assert cap.should_terminate(None, None, "r") is False


def test_capability_can_be_instantiated_directly():
    """Trivial pass-through capability — useful for tests, mocks, etc."""
    cap = zurvan.Capability(name="noop", description="does nothing")
    assert cap.name == "noop"
    # Default hooks all work without raising
    assert cap.process_prompt(None, None, "p") == "p"
