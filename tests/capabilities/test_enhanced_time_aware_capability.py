"""Tests for EnhancedTimeAwareCapability (submodule API).

Adds per-action ``execution_time`` and per-result ``action_duration`` on
top of the base ``TimeAwareCapability``.
"""

import time

from zurvan import ActionContext, Memory
from zurvan.capabilities.enhanced_time_aware_capability import (
    EnhancedTimeAwareCapability,
)


def test_init_inherited_from_base_capability_adds_system_memory():
    """The enhanced subclass keeps the base class' init behaviour."""
    cap = EnhancedTimeAwareCapability()
    memory = Memory()
    ctx = ActionContext({"memory": memory, "time_zone": "UTC"})
    cap.init(agent=None, action_context=ctx)
    assert memory.get_memories()[0]["type"] == "system"


def test_process_action_stamps_execution_time():
    cap = EnhancedTimeAwareCapability()
    ctx = ActionContext({"time_zone": "UTC"})
    action = {"action_def": object(), "invocation": {"args": {}}}

    out = cap.process_action(agent=None, action_context=ctx, action=action)

    assert "execution_time" in out
    # ISO-8601 with timezone — produced by datetime.isoformat()
    assert "T" in out["execution_time"]


def test_process_result_records_action_duration_for_dict_results():
    cap = EnhancedTimeAwareCapability()
    ctx = ActionContext({"time_zone": "UTC"})
    action = {}
    cap.process_action(agent=None, action_context=ctx, action=action)
    # Sleep so the duration is non-zero
    time.sleep(0.02)

    result = {"tool_executed": True, "result": "ok"}
    out = cap.process_result(
        agent=None, action_context=ctx, response="r",
        action_def=None, action=action, result=result,
    )

    assert "action_duration" in out
    assert out["action_duration"] > 0


def test_process_result_passes_through_non_dict_results_unchanged():
    cap = EnhancedTimeAwareCapability()
    ctx = ActionContext({"time_zone": "UTC"})
    action = {}
    cap.process_action(agent=None, action_context=ctx, action=action)

    out = cap.process_result(
        agent=None, action_context=ctx, response="r",
        action_def=None, action=action, result="not a dict",
    )
    assert out == "not a dict"
