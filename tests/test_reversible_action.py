"""Tests for zurvan.reversible_action.ReversibleAction (submodule API)."""

import pytest

from zurvan.reversible_action import ReversibleAction


def test_run_executes_and_records_invocation():
    captured = {}

    def execute(**kwargs):
        captured.update(kwargs)
        return "executed"

    def reverse(**_):
        pass

    action = ReversibleAction(execute, reverse)
    result = action.run(file="x.txt", op="create")

    assert result == "executed"
    assert captured == {"file": "x.txt", "op": "create"}
    assert action.execution_record is not None
    assert action.execution_record["args"] == {"file": "x.txt", "op": "create"}
    assert action.execution_record["result"] == "executed"
    assert "timestamp" in action.execution_record


def test_undo_calls_reverse_with_execution_record_kwargs():
    """The reverse function receives args/result/timestamp as kwargs from
    the execution record — so it can undo using whatever info was captured
    during ``run()``."""
    received = {}

    def execute(**kwargs):
        return "done"

    def reverse(args, result, timestamp):
        received["args"] = args
        received["result"] = result
        received["timestamp"] = timestamp
        return "reversed"

    action = ReversibleAction(execute, reverse)
    action.run(target="x")
    out = action.undo()

    assert out == "reversed"
    assert received["args"] == {"target": "x"}
    assert received["result"] == "done"
    assert received["timestamp"] is not None


def test_undo_without_run_raises():
    action = ReversibleAction(lambda **k: None, lambda **k: None)
    with pytest.raises(ValueError, match="No action to reverse"):
        action.undo()
