"""Tests for TimeAwareCapability."""

from zurvan import ActionContext, Memory, Prompt, TimeAwareCapability


def test_init_adds_system_memory_with_time_info():
    cap = TimeAwareCapability()
    memory = Memory()
    ctx = ActionContext({"memory": memory, "time_zone": "UTC"})

    cap.init(agent=None, action_context=ctx)

    items = memory.get_memories()
    assert len(items) == 1
    entry = items[0]
    assert entry["type"] == "system"
    assert "UTC" in entry["content"]


def test_init_uses_default_timezone_when_unset():
    cap = TimeAwareCapability()
    memory = Memory()
    ctx = ActionContext({"memory": memory})

    cap.init(agent=None, action_context=ctx)

    items = memory.get_memories()
    assert "America/Chicago" in items[0]["content"]


def test_process_prompt_prepends_time_to_existing_system_message():
    cap = TimeAwareCapability()
    ctx = ActionContext({"time_zone": "UTC"})
    prompt = Prompt(messages=[{"role": "system", "content": "be helpful"}])

    out = cap.process_prompt(agent=None, action_context=ctx, prompt=prompt)

    assert "Current time" in out.messages[0]["content"]
    assert "be helpful" in out.messages[0]["content"]


def test_process_prompt_creates_system_message_when_absent():
    cap = TimeAwareCapability()
    ctx = ActionContext({"time_zone": "UTC"})
    prompt = Prompt(messages=[{"role": "user", "content": "hi"}])

    out = cap.process_prompt(agent=None, action_context=ctx, prompt=prompt)

    assert out.messages[0]["role"] == "system"
    assert "Current time" in out.messages[0]["content"]
    assert out.messages[1]["content"] == "hi"
