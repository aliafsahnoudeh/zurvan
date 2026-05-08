"""
Reusable token-usage and cost tracker for LLM calls.

Works with any LLM client that returns response objects with a ``usage``
attribute (litellm, OpenAI SDK, etc.).  Accumulates per-model statistics
and produces a human-readable report.

Usage::

    from zurvan.token_tracker import TokenTracker

    tracker = TokenTracker()

    # After each LLM call:
    response = completion(model=model, messages=messages)
    tracker.record(model, response)

    # When done:
    print(tracker.report())
"""

from collections import defaultdict
from typing import Any


class TokenTracker:
    """Accumulates token usage and cost per LLM model."""

    def __init__(self):
        self._usage: dict[str, dict] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "calls": 0,
            }
        )

    def record(self, model: str, response: Any) -> None:
        """Extract usage from an LLM response and accumulate it.

        Expects *response* to carry a ``usage`` attribute with
        ``prompt_tokens``, ``completion_tokens``, and ``total_tokens``
        fields (the convention used by litellm and the OpenAI SDK).

        Cost is estimated via ``litellm.completion_cost`` when available;
        if litellm is not installed or cost data is missing the call
        count and token totals are still recorded.
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return

        entry = self._usage[model]
        entry["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        entry["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        entry["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        entry["calls"] += 1

        try:
            from litellm import completion_cost

            cost = completion_cost(completion_response=response)
            entry["cost"] += cost
        except Exception:
            pass

    @property
    def total_tokens(self) -> int:
        """Return the grand total of tokens across all models."""
        return sum(u["total_tokens"] for u in self._usage.values())

    @property
    def total_cost(self) -> float:
        """Return the grand total estimated cost across all models."""
        return sum(u["cost"] for u in self._usage.values())

    @property
    def total_calls(self) -> int:
        """Return the grand total of LLM calls across all models."""
        return sum(u["calls"] for u in self._usage.values())

    def reset(self) -> None:
        """Clear all accumulated usage data."""
        self._usage.clear()

    def report(self) -> str:
        """Return a human-readable token usage & cost report."""
        if not self._usage:
            return "No LLM calls were recorded."

        lines = [
            "",
            "=" * 70,
            "TOKEN USAGE & COST REPORT",
            "=" * 70,
        ]

        grand_prompt = 0
        grand_completion = 0
        grand_total = 0
        grand_cost = 0.0
        grand_calls = 0

        for model, u in sorted(self._usage.items()):
            grand_prompt += u["prompt_tokens"]
            grand_completion += u["completion_tokens"]
            grand_total += u["total_tokens"]
            grand_cost += u["cost"]
            grand_calls += u["calls"]

            lines.append(f"\nModel: {model}")
            lines.append(f"  Calls:             {u['calls']}")
            lines.append(f"  Prompt tokens:     {u['prompt_tokens']:,}")
            lines.append(f"  Completion tokens: {u['completion_tokens']:,}")
            lines.append(f"  Total tokens:      {u['total_tokens']:,}")
            if u["cost"] > 0:
                lines.append(f"  Estimated cost:    ${u['cost']:.6f}")
            else:
                lines.append("  Estimated cost:    (unavailable)")

        lines.append("\n" + "-" * 70)
        lines.append("GRAND TOTAL (all models)")
        lines.append(f"  Calls:             {grand_calls}")
        lines.append(f"  Prompt tokens:     {grand_prompt:,}")
        lines.append(f"  Completion tokens: {grand_completion:,}")
        lines.append(f"  Total tokens:      {grand_total:,}")
        if grand_cost > 0:
            lines.append(f"  Estimated cost:    ${grand_cost:.6f}")
        else:
            lines.append("  Estimated cost:    (unavailable)")
        lines.append("=" * 70)

        return "\n".join(lines)
