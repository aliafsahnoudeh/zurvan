"""Tests for zurvan.Memory."""

import json

from zurvan import Memory


def test_add_and_get_memories():
    m = Memory()
    assert m.get_memories() == []

    m.add_memory({"type": "user", "content": "hi"})
    m.add_memory({"type": "assistant", "content": "hello"})
    assert len(m.get_memories()) == 2
    assert m.get_memories()[0]["content"] == "hi"


def test_get_memories_with_limit():
    m = Memory()
    for i in range(5):
        m.add_memory({"type": "user", "content": f"msg{i}"})
    assert len(m.get_memories(limit=3)) == 3
    assert m.get_memories(limit=3)[-1]["content"] == "msg2"


def test_copy_without_system_memories_excludes_system_only():
    m = Memory()
    m.add_memory({"type": "user", "content": "u1"})
    m.add_memory({"type": "system", "content": "s1"})
    m.add_memory({"type": "assistant", "content": "a1"})
    m.add_memory({"type": "system", "content": "s2"})

    copy = m.copy_without_system_memories()
    types = [item["type"] for item in copy.get_memories()]
    assert types == ["user", "assistant"]
    # Original is unchanged
    assert len(m.get_memories()) == 4


def test_copy_without_system_returns_independent_memory():
    original = Memory()
    original.add_memory({"type": "user", "content": "u1"})
    copy = original.copy_without_system_memories()
    copy.add_memory({"type": "user", "content": "u2"})
    assert len(original.get_memories()) == 1
    assert len(copy.get_memories()) == 2


def test_compact_short_entries_unchanged():
    m = Memory()
    for i in range(10):
        m.add_memory({"type": "user", "content": f"short{i}"})
    compacted = m.compact()
    assert compacted == 0
    assert all("compacted" not in item["content"] for item in m.get_memories())


def test_compact_truncates_middle_long_entries_only():
    m = Memory()
    long_text = "x" * 1000
    m.add_memory({"type": "user", "content": "first"})  # preserve_first=1
    m.add_memory({"type": "assistant", "content": long_text})  # middle (truncatable)
    m.add_memory({"type": "user", "content": long_text})  # middle
    # keep_recent=4 so the next four are kept intact
    for _ in range(4):
        m.add_memory({"type": "assistant", "content": long_text})

    compacted = m.compact(preserve_first=1, keep_recent=4, max_entry_length=50)
    assert compacted == 2
    items = m.get_memories()
    assert items[0]["content"] == "first"
    assert "compacted" in items[1]["content"]
    assert "compacted" in items[2]["content"]
    # Recent four are untouched
    for item in items[3:]:
        assert item["content"] == long_text


def test_compact_handles_json_result_field():
    m = Memory()
    big_result = "y" * 500
    m.add_memory({"type": "user", "content": "first"})
    payload = json.dumps({"tool_executed": True, "result": big_result})
    m.add_memory({"type": "user", "content": payload})
    for _ in range(4):
        m.add_memory({"type": "assistant", "content": "tail"})

    compacted = m.compact(preserve_first=1, keep_recent=4, max_entry_length=50)
    assert compacted == 1
    middle_payload = json.loads(m.get_memories()[1]["content"])
    assert middle_payload["tool_executed"] is True
    assert "compacted" in middle_payload["result"]
