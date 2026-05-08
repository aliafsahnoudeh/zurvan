"""Tests for zurvan.progress_reporter (submodule API)."""

from zurvan.progress_reporter import ProgressEvent, ProgressReporter


def test_emit_dispatches_to_subscribers():
    reporter = ProgressReporter()
    received = []
    reporter.subscribe(lambda e: received.append(e))

    reporter.emit("step_started", "Working on it", n=1)

    assert len(received) == 1
    event = received[0]
    assert event.kind == "step_started"
    assert event.message == "Working on it"
    assert event.data == {"n": 1}


def test_emit_with_no_subscribers_is_silent():
    """Agents must be able to emit unconditionally without checking
    whether anyone is listening."""
    reporter = ProgressReporter()
    # Should not raise
    reporter.emit("noop", "no one listening")


def test_unsubscribe_removes_callback():
    reporter = ProgressReporter()
    received = []
    callback = received.append
    reporter.subscribe(callback)
    reporter.emit("a", "first")
    reporter.unsubscribe(callback)
    reporter.emit("b", "second")
    assert len(received) == 1
    assert received[0].kind == "a"


def test_unsubscribe_unknown_callback_is_noop():
    reporter = ProgressReporter()
    # Removing something never subscribed should not raise
    reporter.unsubscribe(lambda e: None)


def test_subscribe_returns_callback_for_chaining():
    reporter = ProgressReporter()

    def cb(e):
        pass

    assert reporter.subscribe(cb) is cb


def test_faulty_subscriber_is_isolated():
    """A raising subscriber must not break the dispatch chain — agents
    keep running even if a UI/observer is broken."""
    reporter = ProgressReporter()
    received = []
    reporter.subscribe(lambda e: 1 / 0)  # broken
    reporter.subscribe(lambda e: received.append(e))  # good

    reporter.emit("k", "m")  # should not raise

    assert len(received) == 1


def test_emit_event_dispatches_prebuilt_event():
    reporter = ProgressReporter()
    received = []
    reporter.subscribe(received.append)
    event = ProgressEvent(kind="custom", message="hi", data={"a": 1})
    reporter.emit_event(event)
    assert received == [event]


def test_progress_event_is_frozen_dataclass():
    event = ProgressEvent(kind="x", message="y")
    try:
        event.kind = "z"
    except (AttributeError, Exception):
        pass
    else:
        raise AssertionError("ProgressEvent should be frozen")


def test_progress_event_data_defaults_to_empty_dict():
    event = ProgressEvent(kind="x", message="y")
    assert event.data == {}
