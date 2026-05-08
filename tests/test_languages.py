"""Tests for the function-calling AgentLanguage subclasses.

Real ``litellm.completion`` is patched out — these tests verify our
adapter logic, not the providers.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from zurvan import (
    Action,
    AgentFunctionCallingActionLanguage,
    AgentFunctionCallingActionLanguageGemini,
    AgentFunctionCallingActionLanguageGroq,
    AgentFunctionCallingActionLanguageOpenAI,
    Environment,
    Goal,
    Memory,
    Prompt,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _fake_tool_response(tool_name="terminate", args=None):
    """Stand-in for a litellm response carrying a tool call."""
    args = args or {"message": "done"}
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name=tool_name, arguments=json.dumps(args))
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _fake_text_response(text="hello"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _fake_empty_choices():
    return SimpleNamespace(choices=[])


def _fake_empty_content():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
    )


# ── parse_response (the JSON parser) ─────────────────────────────────────


class TestParseResponse:
    def setup_method(self):
        self.lang = AgentFunctionCallingActionLanguage(model="x")

    def test_parses_clean_single_json(self):
        result = self.lang.parse_response('{"tool": "x", "args": {"a": 1}}')
        assert result == {"tool": "x", "args": {"a": 1}}

    def test_parses_first_of_concatenated_json(self):
        # Two JSON objects glued together — should pick the first
        payload = '{"tool": "first", "args": {}}{"tool": "second", "args": {}}'
        assert self.lang.parse_response(payload)["tool"] == "first"

    def test_falls_back_to_terminate_for_unparseable_text(self):
        result = self.lang.parse_response("not JSON at all")
        assert result == {"tool": "terminate", "args": {"message": "not JSON at all"}}

    def test_raises_on_empty_response(self):
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            self.lang.parse_response("")

    def test_raises_on_whitespace_only_response(self):
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            self.lang.parse_response("   \n\t  ")

    def test_raises_on_none_response(self):
        with pytest.raises(ValueError):
            self.lang.parse_response(None)


# ── Base class hook contract ─────────────────────────────────────────────


class TestBaseHookDefaults:
    def test_default_api_key_env_var_is_none(self):
        lang = AgentFunctionCallingActionLanguage(model="x")
        assert lang._api_key_env_var() is None

    def test_default_extra_completion_kwargs_is_empty(self):
        lang = AgentFunctionCallingActionLanguage(model="x")
        assert lang._extra_completion_kwargs(Prompt()) == {}

    def test_response_observers_can_be_added_and_notified(self):
        lang = AgentFunctionCallingActionLanguage(model="x")
        captured = []

        def observer(model, response):
            captured.append((model, response))

        lang.add_response_observer(observer)
        lang._notify_response("test-model", "fake-response")
        assert captured == [("test-model", "fake-response")]

    def test_failing_observer_is_swallowed_not_propagated(self):
        lang = AgentFunctionCallingActionLanguage(model="x")
        lang.add_response_observer(lambda m, r: 1 / 0)
        # Should not raise — broken observers must not break the loop
        lang._notify_response("m", "r")


# ── _parse_completion ────────────────────────────────────────────────────


class TestParseCompletion:
    def setup_method(self):
        self.lang = AgentFunctionCallingActionLanguage(model="x")

    def test_extracts_tool_call_as_json(self):
        out = self.lang._parse_completion(
            _fake_tool_response("terminate", {"message": "done"}),
            tools=[{"type": "function", "function": {"name": "terminate"}}],
        )
        assert json.loads(out) == {"tool": "terminate", "args": {"message": "done"}}

    def test_returns_text_content_when_no_tool_call(self):
        out = self.lang._parse_completion(_fake_text_response("hello world"), tools=None)
        assert out == "hello world"

    def test_raises_on_empty_choices(self):
        with pytest.raises(RuntimeError, match="empty choices"):
            self.lang._parse_completion(_fake_empty_choices(), tools=None)

    def test_raises_on_empty_content(self):
        with pytest.raises(RuntimeError, match="empty response"):
            self.lang._parse_completion(_fake_empty_content(), tools=None)


# ── End-to-end generate_response via patched litellm.completion ──────────


class TestGenerateResponseEndToEnd:
    """Verifies the kwargs construction + dispatch path for each provider."""

    def test_openai_uses_no_explicit_api_key(self):
        lang = AgentFunctionCallingActionLanguageOpenAI(model="openai/gpt-4o-mini")
        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            return_value=_fake_text_response("ok"),
        ) as mock_completion:
            lang.generate_response(Prompt(messages=[{"role": "user", "content": "hi"}]))

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o-mini"
        assert "api_key" not in kwargs  # OpenAI lets LiteLLM auto-resolve

    def test_gemini_forwards_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-gemini-key")
        lang = AgentFunctionCallingActionLanguageGemini()
        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            return_value=_fake_text_response("ok"),
        ) as mock_completion:
            lang.generate_response(Prompt(messages=[{"role": "user", "content": "hi"}]))

        assert mock_completion.call_args.kwargs["api_key"] == "fake-gemini-key"

    def test_groq_forwards_groq_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
        lang = AgentFunctionCallingActionLanguageGroq()
        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            return_value=_fake_text_response("ok"),
        ) as mock_completion:
            lang.generate_response(Prompt(messages=[{"role": "user", "content": "hi"}]))

        assert mock_completion.call_args.kwargs["api_key"] == "fake-groq-key"

    def test_tools_are_only_forwarded_when_present(self):
        lang = AgentFunctionCallingActionLanguageOpenAI()

        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            return_value=_fake_text_response("ok"),
        ) as mock_completion:
            # No tools in prompt
            lang.generate_response(Prompt(messages=[{"role": "user", "content": "hi"}]))
            assert "tools" not in mock_completion.call_args.kwargs

            # With tools
            lang.generate_response(
                Prompt(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "x"}}],
                )
            )
            assert "tools" in mock_completion.call_args.kwargs

    def test_response_observer_invoked_after_completion(self):
        lang = AgentFunctionCallingActionLanguageOpenAI()
        captured = []
        lang.add_response_observer(lambda model, resp: captured.append(model))
        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            return_value=_fake_text_response("ok"),
        ):
            lang.generate_response(Prompt(messages=[{"role": "user", "content": "hi"}]))
        assert captured == [lang.model]


# ── Gemini-specific shape: roles + thinking_budget + merging ─────────────


class TestGemini:
    def test_format_goals_translates_system_to_user(self):
        lang = AgentFunctionCallingActionLanguageGemini()
        goals = [Goal(priority=1, name="g", description="d")]
        formatted = lang.format_goals(goals)
        assert all(m["role"] == "user" for m in formatted)

    def test_format_memory_translates_system_to_user(self):
        lang = AgentFunctionCallingActionLanguageGemini()
        memory = Memory()
        memory.add_memory({"type": "system", "content": "system msg"})
        memory.add_memory({"type": "user", "content": "user msg"})
        memory.add_memory({"type": "assistant", "content": "assistant msg"})
        formatted = lang.format_memory(memory)
        roles = [m["role"] for m in formatted]
        # system → user; assistant stays assistant; user stays user
        assert roles == ["user", "user", "assistant"]

    def test_merge_consecutive_collapses_same_role_runs(self):
        lang = AgentFunctionCallingActionLanguageGemini()
        merged = lang._merge_consecutive(
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
                {"role": "user", "content": "d"},
                {"role": "user", "content": "e"},
            ]
        )
        roles = [m["role"] for m in merged]
        assert roles == ["user", "assistant", "user"]
        assert merged[0]["content"] == "a\n\nb"
        assert merged[2]["content"] == "d\n\ne"

    def test_construct_prompt_runs_merge_consecutive(self):
        """Goals as system + memory items can produce same-role runs that
        Gemini rejects; construct_prompt should normalise them."""
        lang = AgentFunctionCallingActionLanguageGemini()
        memory = Memory()
        memory.add_memory({"type": "user", "content": "u1"})
        memory.add_memory({"type": "user", "content": "u2"})
        prompt = lang.construct_prompt(
            actions=[], environment=Environment(),
            goals=[Goal(priority=1, name="g", description="d")], memory=memory,
        )
        # No two adjacent messages share a role
        for a, b in zip(prompt.messages, prompt.messages[1:]):
            assert a["role"] != b["role"]

    def test_thinking_budget_added_to_extra_body(self):
        lang = AgentFunctionCallingActionLanguageGemini(thinking_budget=0)
        extra = lang._extra_completion_kwargs(Prompt())
        assert extra["extra_body"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0

    def test_no_thinking_budget_means_no_extra_body(self):
        lang = AgentFunctionCallingActionLanguageGemini(thinking_budget=None)
        assert lang._extra_completion_kwargs(Prompt()) == {}

    def test_empty_choices_message_mentions_thinking_tokens(self):
        lang = AgentFunctionCallingActionLanguageGemini()
        assert "thinking-token" in lang._empty_choices_message()


# ── Groq retry behaviour ─────────────────────────────────────────────────


class TestGroqRetry:
    def _make_rate_limit_error(self, msg="Rate limit hit. Please try again in 1.5s."):
        from litellm.exceptions import RateLimitError

        # RateLimitError signature varies between litellm versions.
        # We only care about str(exc) in our retry parser; subclass to keep
        # the test independent of litellm's constructor changes.
        return type("FakeRL", (RateLimitError,), {"__init__": lambda self: Exception.__init__(self, msg), "__str__": lambda self: msg})()

    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        # Patch sleep so the test is fast
        monkeypatch.setattr(
            "zurvan.agent_function_calling_action_language_groq.time.sleep",
            lambda s: None,
        )
        lang = AgentFunctionCallingActionLanguageGroq()
        rl = self._make_rate_limit_error()

        call_count = {"n": 0}

        def fake_completion(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise rl
            return _fake_text_response("finally")

        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            side_effect=fake_completion,
        ):
            out = lang.generate_response(
                Prompt(messages=[{"role": "user", "content": "hi"}])
            )

        assert out == "finally"
        assert call_count["n"] == 3

    def test_gives_up_after_max_retries(self, monkeypatch):
        from litellm.exceptions import RateLimitError

        monkeypatch.setenv("GROQ_API_KEY", "x")
        monkeypatch.setattr(
            "zurvan.agent_function_calling_action_language_groq.time.sleep",
            lambda s: None,
        )
        lang = AgentFunctionCallingActionLanguageGroq()
        rl = self._make_rate_limit_error()

        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            side_effect=rl,
        ):
            with pytest.raises(RateLimitError):
                lang.generate_response(
                    Prompt(messages=[{"role": "user", "content": "hi"}])
                )

    def test_unrecoverable_tpm_fails_fast_without_retrying(self, monkeypatch):
        from litellm.exceptions import RateLimitError

        monkeypatch.setenv("GROQ_API_KEY", "x")
        sleep_calls = []
        monkeypatch.setattr(
            "zurvan.agent_function_calling_action_language_groq.time.sleep",
            lambda s: sleep_calls.append(s),
        )
        lang = AgentFunctionCallingActionLanguageGroq()
        # Request bigger than per-minute quota → unrecoverable
        msg = "Rate limit hit on tokens per minute. Limit 1000, Requested 5000"
        unrecoverable = type(
            "RL",
            (RateLimitError,),
            {
                "__init__": lambda self: Exception.__init__(self, msg),
                "__str__": lambda self: msg,
            },
        )()

        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            side_effect=unrecoverable,
        ):
            with pytest.raises(RateLimitError):
                lang.generate_response(
                    Prompt(messages=[{"role": "user", "content": "hi"}])
                )

        # Did not sleep — failed fast on first attempt
        assert sleep_calls == []


# ── OpenAI defaults ──────────────────────────────────────────────────────


class TestOpenAIDefaults:
    def test_construction_with_defaults(self):
        lang = AgentFunctionCallingActionLanguageOpenAI()
        assert lang.model == "openai/gpt-4o"
        assert lang.max_tokens == 1024
        assert lang._api_key_env_var() is None

    def test_response_observers_supported(self):
        lang = AgentFunctionCallingActionLanguageOpenAI()
        captured = []
        lang.add_response_observer(lambda m, r: captured.append(m))
        with patch(
            "zurvan.agent_function_calling_action_language.completion",
            return_value=_fake_text_response("ok"),
        ):
            lang.generate_response(Prompt(messages=[{"role": "user", "content": "hi"}]))
        assert captured == ["openai/gpt-4o"]


# ── format_actions / construct_prompt across base ────────────────────────


def test_format_actions_truncates_long_descriptions():
    lang = AgentFunctionCallingActionLanguage(model="x")
    long_desc = "x" * 2000
    action = Action(
        name="a", function=lambda: None, description=long_desc,
        parameters={"type": "object", "properties": {}, "required": []},
    )
    formatted = lang.format_actions([action])
    assert len(formatted[0]["function"]["description"]) == 1024


def test_construct_prompt_includes_goals_memory_and_tools():
    lang = AgentFunctionCallingActionLanguage(model="x")
    memory = Memory()
    memory.add_memory({"type": "user", "content": "u1"})
    action = Action(
        name="t", function=lambda: None, description="t",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    prompt = lang.construct_prompt(
        actions=[action], environment=Environment(),
        goals=[Goal(priority=1, name="g", description="d")], memory=memory,
    )
    roles = [m["role"] for m in prompt.messages]
    assert "system" in roles  # goal
    assert "user" in roles    # memory item
    assert prompt.tools[0]["function"]["name"] == "t"
