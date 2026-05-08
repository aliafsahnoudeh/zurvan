"""Shared pytest fixtures and a scripted-LLM helper for fast unit tests."""

import json
from typing import List, Union

import pytest

from zurvan import (
    Action,
    ActionRegistry,
    AgentLanguage,
    Environment,
    Goal,
    Prompt,
)


class _ScriptedLanguage(AgentLanguage):
    """Fake :class:`AgentLanguage` that returns canned responses.

    Pass a list of items where each item is either:

    - a ``str`` (returned as-is by ``generate_response`` and parsed as
      JSON by ``parse_response``)
    - a ``dict`` (auto-serialised — convenience for tool-call payloads)
    - an ``Exception`` (raised on that call — simulates LLM failure)

    The fake records ``self.calls`` so tests can assert how many LLM
    round-trips an agent did.
    """

    def __init__(self, responses: List[Union[str, dict, Exception]]):
        super().__init__()
        self._scripted = list(responses)
        self.calls = 0
        self.prompts_seen: List[Prompt] = []

    def generate_response(self, prompt: Prompt) -> str:
        self.prompts_seen.append(prompt)
        if self.calls >= len(self._scripted):
            raise RuntimeError(
                f"_ScriptedLanguage exhausted after {self.calls} calls; "
                f"test gave {len(self._scripted)} responses."
            )
        item = self._scripted[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return json.dumps(item)
        return item

    def construct_prompt(self, actions, environment, goals, memory):
        return Prompt(messages=[], tools=[])

    def parse_response(self, response: str) -> dict:
        return json.loads(response)


@pytest.fixture
def scripted_language():
    """Factory fixture: ``scripted_language([...])`` builds a _ScriptedLanguage."""
    return _ScriptedLanguage


@pytest.fixture
def terminate_action():
    """A canonical terminal action used by most agent loop tests."""
    return Action(
        name="terminate",
        function=lambda message: message,
        description="End the conversation with a final message.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        terminal=True,
    )


@pytest.fixture
def echo_action():
    """A non-terminal action that returns its input — keeps the loop alive."""
    return Action(
        name="echo",
        function=lambda text: f"echoed: {text}",
        description="Echo back the input.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        terminal=False,
    )


@pytest.fixture
def basic_registry(terminate_action, echo_action):
    """An ActionRegistry preloaded with terminate + echo."""
    registry = ActionRegistry()
    registry.register(terminate_action)
    registry.register(echo_action)
    return registry


@pytest.fixture
def basic_goals():
    return [Goal(priority=1, name="test", description="A test goal.")]


@pytest.fixture
def environment():
    return Environment()
