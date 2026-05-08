"""Tests for zurvan.action_transaction.ActionTransaction (submodule API).

Note: ``ActionTransaction.execute()`` and ``rollback()`` are async, but
``ReversibleAction.undo()`` returns whatever ``reverse(...)`` returns —
so tests pair the transaction's ``await ...undo()`` with async reverse
functions. We use ``asyncio.run`` to drive these without pulling in
pytest-asyncio.
"""

import asyncio

import pytest

from zurvan.action_transaction import ActionTransaction
from zurvan.reversible_action import ReversibleAction


def make_action(execute_calls, reverse_calls, name):
    def execute(**kwargs):
        execute_calls.append((name, kwargs))
        return f"{name}:{kwargs}"

    async def reverse(**kwargs):
        reverse_calls.append((name, kwargs))
        return f"reversed:{name}"

    return ReversibleAction(execute, reverse)


def test_add_queues_actions():
    tx = ActionTransaction()
    a = make_action([], [], "a")
    b = make_action([], [], "b")
    tx.add(a, x=1)
    tx.add(b, y=2)
    assert len(tx.actions) == 2


def test_add_after_commit_raises():
    tx = ActionTransaction()
    tx.commit()
    with pytest.raises(ValueError, match="already committed"):
        tx.add(make_action([], [], "x"))


def test_execute_runs_actions_in_order():
    execute_calls: list = []
    reverse_calls: list = []
    tx = ActionTransaction()
    tx.add(make_action(execute_calls, reverse_calls, "a"), v=1)
    tx.add(make_action(execute_calls, reverse_calls, "b"), v=2)
    tx.add(make_action(execute_calls, reverse_calls, "c"), v=3)

    asyncio.run(tx.execute())

    assert [name for name, _ in execute_calls] == ["a", "b", "c"]
    # No rollback fired
    assert reverse_calls == []
    assert len(tx.executed) == 3


def test_failure_mid_execute_triggers_rollback_in_reverse_order():
    execute_calls: list = []
    reverse_calls: list = []
    tx = ActionTransaction()

    def boom_execute(**kwargs):
        raise RuntimeError("nope")

    async def noop_reverse(**kwargs):
        return None

    boom_action = ReversibleAction(boom_execute, noop_reverse)

    tx.add(make_action(execute_calls, reverse_calls, "a"))
    tx.add(make_action(execute_calls, reverse_calls, "b"))
    tx.add(boom_action)
    tx.add(make_action(execute_calls, reverse_calls, "never"))

    with pytest.raises(RuntimeError, match="nope"):
        asyncio.run(tx.execute())

    # First two actions executed, then 'boom' raised, then rollback
    assert [name for name, _ in execute_calls] == ["a", "b"]
    # Rollback fired in reverse order: b then a
    assert [name for name, _ in reverse_calls] == ["b", "a"]
    # 'never' should not have been touched
    assert "never" not in [name for name, _ in execute_calls]


def test_commit_marks_transaction_as_finalised():
    tx = ActionTransaction()
    assert tx.committed is False
    tx.commit()
    assert tx.committed is True


def test_transaction_id_is_unique_per_instance():
    a = ActionTransaction()
    b = ActionTransaction()
    assert a.transaction_id != b.transaction_id
