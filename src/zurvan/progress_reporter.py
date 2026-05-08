"""
Lightweight progress-event pub/sub for agents.

Agents emit structured ``ProgressEvent`` objects at meaningful points in
their workflow (starting a search, expanding a query, refining the final
answer, …). Callers who care about surfacing those steps — a CLI that
prints them, a web UI that streams them to a WebSocket — subscribe a
callback via :meth:`ProgressReporter.subscribe`. Callers who don't care
pass nothing: emission against an empty reporter is a no-op, so agents
can always call ``reporter.emit(...)`` without guarding for ``None``.

This is intentionally separate from logging: logs are diagnostic and
persistent; progress events are user-facing signals about *what the
agent is doing right now*.

Usage::

    from zurvan.progress_reporter import ProgressReporter

    reporter = ProgressReporter()
    reporter.subscribe(lambda e: print(e.message))

    reporter.emit("search_started", "Searching 'atomic_habits'...",
                  book_id="atomic_habits")

Concrete observers (terminal printers, WebSocket bridges, …) belong in
the application layer, not here — the framework defines the contract,
callers plug in the transport.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ProgressEvent:
    """A single progress signal emitted by an agent.

    Attributes
    ----------
    kind
        Short machine-readable event type (e.g. ``"search_started"``,
        ``"query_expanded"``, ``"refining_answer"``). Consumers can
        dispatch on this.
    message
        Human-readable description suitable for direct display.
    data
        Optional structured payload (book id, query string, counts, …)
        for consumers that want to render richer UI than ``message``.
    """

    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


class ProgressReporter:
    """Fan-out hub for :class:`ProgressEvent` notifications.

    Subscribers are plain callables taking a single ``ProgressEvent``.
    A reporter with zero subscribers silently drops events, so agents
    can emit unconditionally without a null-check.

    Subscriber callbacks run synchronously on the emitting thread. They
    should be fast and non-throwing; a raising callback is caught and
    suppressed so it cannot break the agent loop. Callbacks that need
    to do real work (network I/O, WebSocket pushes) should hand off to
    their own executor or event loop.
    """

    def __init__(self) -> None:
        self._subscribers: list[ProgressCallback] = []

    def subscribe(self, callback: ProgressCallback) -> ProgressCallback:
        """Register *callback* to receive every future event.

        Returns the callback unchanged so this can be used as a
        decorator or chained.
        """
        self._subscribers.append(callback)
        return callback

    def unsubscribe(self, callback: ProgressCallback) -> None:
        """Remove a previously-subscribed callback. No-op if absent."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def emit(self, kind: str, message: str, **data: Any) -> None:
        """Build a :class:`ProgressEvent` and dispatch it to subscribers."""
        event = ProgressEvent(kind=kind, message=message, data=dict(data))
        self.emit_event(event)

    def emit_event(self, event: ProgressEvent) -> None:
        """Dispatch a pre-built :class:`ProgressEvent` to subscribers."""
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:
                # A broken observer must never break the agent.
                pass
