"""Tests for zurvan.agent_json_action_language (submodule API).

This is the markdown-block alternative to the function-calling language —
the LLM is instructed to wrap its tool call in ```action ... ``` blocks.
"""

import pytest

from zurvan import Action, Environment, Goal, Memory
from zurvan.agent_json_action_language import AgentJsonActionLanguage


def test_construct_prompt_contains_goals_memory_and_tools():
    lang = AgentJsonActionLanguage()
    memory = Memory()
    memory.add_memory({"type": "user", "content": "hi"})
    action = Action(
        name="echo",
        function=lambda text: text,
        description="Echo the input",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    prompt = lang.construct_prompt(
        actions=[action], environment=Environment(),
        goals=[Goal(priority=1, name="g", description="be helpful")],
        memory=memory,
    )

    roles = [m["role"] for m in prompt.messages]
    assert "system" in roles  # goal block
    assert "user" in roles    # memory item
    # Tools section is also a system message describing available tools
    tools_systems = [m for m in prompt.tools if m["role"] == "system"]
    assert any("echo" in m["content"] for m in tools_systems)
    assert any("```action" in m["content"] for m in tools_systems)


def test_parse_response_extracts_action_block():
    lang = AgentJsonActionLanguage()
    response = """Let me think about this...

I should call the echo tool.

```action
{"tool": "echo", "args": {"text": "hi"}}
```
"""
    result = lang.parse_response(response)
    assert result == {"tool": "echo", "args": {"text": "hi"}}


def test_parse_response_raises_on_unparseable_block():
    lang = AgentJsonActionLanguage()
    with pytest.raises(Exception):
        lang.parse_response("no markdown action block at all")


def test_format_memory_maps_environment_role_to_assistant():
    """The environment-result memory items show up as assistant messages
    so the LLM treats them as prior model output, not as new input."""
    lang = AgentJsonActionLanguage()
    memory = Memory()
    memory.add_memory({"type": "environment", "content": "tool result"})
    memory.add_memory({"type": "user", "content": "u"})
    memory.add_memory({"type": "assistant", "content": "a"})

    formatted = lang.format_memory(memory)
    roles = [m["role"] for m in formatted]
    assert roles == ["assistant", "user", "assistant"]
