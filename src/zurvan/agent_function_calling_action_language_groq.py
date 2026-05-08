"""Groq-adapted function-calling action language.

Groq serves Llama / Mixtral / Qwen models behind an OpenAI-compatible
API. Roles and tool-calling schema match OpenAI, so the only Groq-
specific bits are:

- the ``groq/<model>`` prefix expected by LiteLLM
- the ``GROQ_API_KEY`` env var (read and forwarded explicitly for
  symmetry with the Gemini language, which does the same for
  ``GOOGLE_API_KEY``)
- a 429-retry loop tuned to Groq's tight TPM quota

Pick up a free API key at https://console.groq.com. The free tier is
generous enough to run small agents without hitting 429s:
llama-3.3-70b-versatile gives ~30 RPM and 14.4k requests/day.
"""

import re
import time

from litellm.exceptions import RateLimitError

from zurvan.agent_function_calling_action_language import (
    AgentFunctionCallingActionLanguage,
)
from zurvan.logger.logger import LogLevel
from zurvan.prompt import Prompt


_RETRY_AFTER_RE = re.compile(r"try again in (\d+(?:\.\d+)?)s", re.IGNORECASE)
_TPM_LIMIT_RE = re.compile(r"Limit\s+(\d+),\s*Requested\s+(\d+)", re.IGNORECASE)

# Multiple attempts let an agent push through transient TPM pressure
# (waiting out the 60-second window) without the framework's outer
# ``consecutive_llm_failures`` counter tripping.
_MAX_RETRIES = 3


def _retry_after_seconds(err_msg: str, default: float = 45.0, cap: float = 60.0) -> float:
    """Parse Groq's "Please try again in Xs" hint (plus a small safety margin)."""
    match = _RETRY_AFTER_RE.search(err_msg)
    if match:
        return min(float(match.group(1)) + 1.0, cap)
    return default


def _is_unrecoverable_tpm(err_msg: str) -> bool:
    """``True`` when the request itself is bigger than the per-minute quota.

    Groq's TPM error message is ``Limit X, Requested Y``. When ``Y > X``
    no amount of waiting will help — the next minute's quota still
    can't fit the request — so we should stop retrying and fail fast
    with an actionable error.
    """
    if "tokens per minute" not in err_msg.lower():
        return False
    m = _TPM_LIMIT_RE.search(err_msg)
    if not m:
        return False
    try:
        return int(m.group(2)) > int(m.group(1))
    except ValueError:
        return False


class AgentFunctionCallingActionLanguageGroq(AgentFunctionCallingActionLanguage):
    """Groq function-calling language with built-in 429 retry.

    Groq's TPM quota is tight enough that transient 429s are part of
    normal operation. On ``RateLimitError`` we sleep for the duration
    Groq suggests (``Please try again in 41.65s``) and retry up to
    ``_MAX_RETRIES`` times. The framework's outer loop treats a
    successful retry as a regular success, so an agent under transient
    TPM pressure can push through without tripping
    ``Max consecutive LLM failures``.

    Unrecoverable cases (request size already exceeds the per-minute
    quota) fail fast with an explicit log so the caller knows to
    switch backends or shrink the request.
    """

    def __init__(
        self,
        model: str = "groq/llama-3.3-70b-versatile",
        max_tokens: int = 4096,
        logger=None,
        response_observers=None,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            logger=logger,
            response_observers=response_observers,
        )

    # ── Provider extension hooks ──────────────────────────────────────────

    def _api_key_env_var(self) -> str:
        return "GROQ_API_KEY"

    def _empty_content_message(self) -> str:
        return (
            f"{self.model} returned empty response "
            "(no tool_calls and no text content) — reasoning/thinking "
            "models may have emitted only reasoning_content, or hit "
            "max_tokens before completing the final answer"
        )

    # ── 429-retry wrapper around the base call ────────────────────────────

    def generate_response(self, prompt: Prompt) -> str:
        """Call the Groq LLM, retrying on 429 with backoff."""
        last_exc: RateLimitError | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return super().generate_response(prompt)
            except RateLimitError as exc:
                last_exc = exc
                err_msg = str(exc)
                if _is_unrecoverable_tpm(err_msg):
                    self.logger.log(
                        f"Groq request size exceeds the per-minute quota "
                        f"on {self.model}. Waiting won't help — switch to "
                        f"a different backend or reduce request size. "
                        f"Error: {err_msg[:240]}",
                        level=LogLevel.ERROR,
                        env=self,
                    )
                    raise
                if attempt >= _MAX_RETRIES:
                    self.logger.log(
                        f"Groq 429 on {self.model} — exhausted "
                        f"{_MAX_RETRIES} retries; giving up.",
                        level=LogLevel.ERROR,
                        env=self,
                    )
                    break
                wait_s = _retry_after_seconds(err_msg)
                self.logger.log(
                    f"Groq 429 on {self.model} (attempt "
                    f"{attempt}/{_MAX_RETRIES}) — sleeping {wait_s:.1f}s",
                    level=LogLevel.WARNING,
                    env=self,
                )
                time.sleep(wait_s)
        assert last_exc is not None
        raise last_exc
