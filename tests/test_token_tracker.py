"""Tests for zurvan.TokenTracker."""

from types import SimpleNamespace

from zurvan import TokenTracker


def fake_response(prompt=10, completion=5, total=15):
    """Build a minimal stand-in for a litellm/OpenAI response object."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )
    )


def test_record_accumulates_tokens_per_model():
    t = TokenTracker()
    t.record("openai/gpt-4o", fake_response(10, 5, 15))
    t.record("openai/gpt-4o", fake_response(20, 10, 30))
    assert t.total_tokens == 45
    assert t.total_calls == 2


def test_record_separates_models():
    t = TokenTracker()
    t.record("openai/gpt-4o", fake_response(10, 5, 15))
    t.record("groq/llama", fake_response(100, 50, 150))
    assert t.total_tokens == 165
    assert t.total_calls == 2


def test_record_response_without_usage_is_safe():
    t = TokenTracker()
    t.record("openai/gpt-4o", SimpleNamespace())  # no .usage attribute
    assert t.total_tokens == 0
    assert t.total_calls == 0


def test_reset_clears_state():
    t = TokenTracker()
    t.record("openai/gpt-4o", fake_response())
    assert t.total_tokens > 0
    t.reset()
    assert t.total_tokens == 0
    assert t.total_calls == 0
    assert t.report() == "No LLM calls were recorded."


def test_report_empty_tracker():
    assert TokenTracker().report() == "No LLM calls were recorded."


def test_report_includes_model_breakdown():
    t = TokenTracker()
    t.record("openai/gpt-4o", fake_response(10, 5, 15))
    t.record("groq/llama-3.3-70b-versatile", fake_response(100, 50, 150))
    report = t.report()
    assert "openai/gpt-4o" in report
    assert "groq/llama-3.3-70b-versatile" in report
    assert "GRAND TOTAL" in report
    assert "165" in report  # grand-total token count


def test_record_handles_partial_usage_fields():
    """Some providers omit individual fields; tracker should default to 0."""
    t = TokenTracker()
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=None, completion_tokens=None, total_tokens=None
        )
    )
    t.record("openai/gpt-4o", response)
    assert t.total_tokens == 0
    assert t.total_calls == 1
