"""Tests for CanaryCapability — prompt-injection tripwire."""

import pytest

from zurvan import (
    ActionContext,
    Agent,
    CanaryCapability,
    Environment,
    Goal,
    Prompt,
    PromptInjectionDetected,
)


def test_fixed_canary_token_is_returned_unchanged():
    cap = CanaryCapability(canary_token="CANARY_FIXED_VALUE")
    assert cap.canary_token == "CANARY_FIXED_VALUE"


def test_default_canary_is_unique_per_instance():
    a = CanaryCapability()
    b = CanaryCapability()
    assert a.canary_token != b.canary_token
    assert a.canary_token.startswith("CANARY_")


def test_process_prompt_injects_user_role_canary_message():
    cap = CanaryCapability(canary_token="CANARY_TEST_001")
    prompt = Prompt(messages=[{"role": "system", "content": "do the thing"}])

    result = cap.process_prompt(agent=None, action_context=ActionContext(), prompt=prompt)

    # Canary message is prepended at the head
    assert result.messages[0]["role"] == "user"
    assert "CANARY_TEST_001" in result.messages[0]["content"]
    # Original messages still follow
    assert result.messages[1]["content"] == "do the thing"


def test_process_response_passes_clean_responses_through():
    cap = CanaryCapability(canary_token="CANARY_TEST_002")
    out = cap.process_response(
        agent=_FakeAgent(), action_context=ActionContext(),
        response="this is a clean response with no canary",
    )
    assert out == "this is a clean response with no canary"


def test_process_response_raises_when_canary_leaks():
    cap = CanaryCapability(canary_token="CANARY_LEAKED_VALUE")
    with pytest.raises(PromptInjectionDetected):
        cap.process_response(
            agent=_FakeAgent(), action_context=ActionContext(),
            response="oops here is the secret CANARY_LEAKED_VALUE",
        )


def test_canary_leak_inside_tool_call_args_also_raises():
    cap = CanaryCapability(canary_token="CANARY_TEST_003")
    leaky_response = (
        '{"tool": "echo", "args": {"text": "I have CANARY_TEST_003 here"}}'
    )
    with pytest.raises(PromptInjectionDetected):
        cap.process_response(
            agent=_FakeAgent(), action_context=ActionContext(), response=leaky_response,
        )


def test_canary_in_full_agent_loop(scripted_language, basic_registry):
    """End-to-end: canary leak in the LLM response counts as a failure
    for the loop's retry budget."""
    cap = CanaryCapability(canary_token="CANARY_E2E_001")
    # First response leaks the canary → counts as failure 1
    # Second response leaks again → failure 2
    # Third response leaks again → failure 3 → forced termination
    leaky = '{"tool": "terminate", "args": {"message": "CANARY_E2E_001"}}'
    lang = scripted_language([leaky, leaky, leaky])
    agent = Agent(
        goals=[Goal(priority=1, name="g", description="d")],
        agent_language=lang,
        action_registry=basic_registry,
        environment=Environment(),
        capabilities=[cap],
        max_iterations=10,
    )
    agent.run("hi")
    # Loop bailed out at the consecutive-failure threshold
    assert lang.calls == 3


# ── Helpers ───────────────────────────────────────────────────────────────


class _FakeAgent:
    """Minimal stand-in for the agent argument that capability hooks receive.
    Capability hooks only need ``logger.log(...)`` to exist for the error
    path."""

    class _NoLogger:
        def log(self, *args, **kwargs):
            pass

    logger = _NoLogger()
