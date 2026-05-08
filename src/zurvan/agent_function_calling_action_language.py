"""Function-calling action language.

Uses an LLM's native function-calling support (via LiteLLM) to specify
actions as JSON tool calls rather than free-form text. Provider
subclasses (OpenAI, Gemini, Groq) override a small set of extension
hooks; the LLM call itself lives in :meth:`generate_response` here.
"""

import json
import os
from typing import Any, Callable, List, Optional

from litellm import completion

from zurvan.action import Action
from zurvan.agent_language import AgentLanguage
from zurvan.environment import Environment
from zurvan.goal import Goal
from zurvan.logger.logger import LogLevel, Logger
from zurvan.memory import Memory
from zurvan.prompt import Prompt


ResponseObserver = Callable[[str, Any], None]


class AgentFunctionCallingActionLanguage(AgentLanguage):
    """Generic LiteLLM-backed function-calling language.

    Subclasses customise behaviour by overriding the hooks below rather
    than reimplementing :meth:`generate_response`:

    - :meth:`_api_key_env_var` — env var to forward as ``api_key``
    - :meth:`_extra_completion_kwargs` — provider-specific kwargs
    - :meth:`_empty_choices_message` / :meth:`_empty_content_message`
      — error messages for the two empty-response failure modes

    Override :meth:`generate_response` directly only when the call
    shape genuinely differs (see ``AgentFunctionCallingActionLanguageGroq``,
    which wraps the base call in a 429-retry loop).
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        logger=None,
        response_observers: Optional[List[ResponseObserver]] = None,
    ):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logger or Logger()
        self._response_observers: List[ResponseObserver] = list(
            response_observers or []
        )

    def add_response_observer(self, observer: ResponseObserver) -> None:
        """Register a callback invoked with ``(model, raw_response)`` after
        every LLM call. Use for cross-cutting concerns (token tracking,
        tracing, audit logging) without subclassing the language."""
        self._response_observers.append(observer)

    def _notify_response(self, model: str, response: Any) -> None:
        """Invoke all registered observers. A faulty observer is logged and
        skipped so it can't break the agent loop."""
        for obs in self._response_observers:
            try:
                obs(model, response)
            except Exception as e:
                self.logger.log(
                    f"Response observer {obs!r} raised: {e}",
                    level=LogLevel.WARNING,
                    env=self,
                )

    # ── Provider extension hooks ──────────────────────────────────────────

    def _api_key_env_var(self) -> Optional[str]:
        """Env var to read and forward as ``api_key`` to ``litellm.completion``.

        LiteLLM auto-resolves most provider keys from the environment, so
        the default returns ``None`` and lets LiteLLM handle it. Override
        only when you need explicit forwarding (Gemini / Groq do this).
        """
        return None

    def _extra_completion_kwargs(self, prompt: Prompt) -> dict:
        """Provider-specific extras for ``litellm.completion``.

        Gemini uses this to pass ``extra_body`` for thinking-budget
        config; most providers don't need it.
        """
        return {}

    def _empty_choices_message(self) -> str:
        return (
            f"{self.model} returned a response with empty choices "
            "(no completion produced)."
        )

    def _empty_content_message(self) -> str:
        return (
            f"{self.model} returned empty response "
            "(no tool_calls and no text content)."
        )

    # ── Prompt formatting ─────────────────────────────────────────────────

    def format_goals(self, goals: List[Goal]) -> List:
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join(
            [f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals]
        )
        return [{"role": "system", "content": goal_instructions}]

    def format_memory(self, memory: Memory) -> List:
        items = memory.get_memories()
        mapped_items = []
        for item in items:
            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)

            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item["type"] == "environment":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items

    def format_actions(self, actions: List[Action]) -> List:
        return [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            }
            for action in actions
        ]

    def construct_prompt(
        self,
        actions: List[Action],
        environment: Environment,
        goals: List[Goal],
        memory: Memory,
    ) -> Prompt:
        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)

        tools = self.format_actions(actions)

        self.logger.log(
            f"Constructed prompt with {prompt} messages and {tools} tools",
            level=LogLevel.DEBUG,
            env=self,
        )

        return Prompt(messages=prompt, tools=tools)

    def adapt_prompt_after_parsing_error(
        self,
        prompt: Prompt,
        response: str,
        traceback: str,
        error: Any,
        retries_left: int,
    ) -> Prompt:
        return prompt

    def parse_response(self, response: str) -> dict:
        """Parse LLM response into structured format.

        Handles three cases:
        1. A single valid JSON object  → use it directly.
        2. Multiple concatenated JSON objects → extract and use the first one.
        3. Anything else → fall back to terminate with the raw text.

        Raises ``ValueError`` when the response is empty or whitespace-only,
        because silently terminating with an empty message turns a transient
        LLM failure into a confident blank answer downstream.
        """
        if response is None or not response.strip():
            raise ValueError(
                "LLM returned empty or whitespace-only response; "
                "cannot parse as an action."
            )

        try:
            self.logger.log(
                f"Attempting to parse response as JSON - Case 1: {response}",
                level=LogLevel.DEBUG,
                env=self,
            )
            return json.loads(response)
        except Exception as e:
            self.logger.log(
                f"Failed to parse response as JSON: {e}",
                level=LogLevel.ERROR,
                env=self,
            )

        self.logger.log(
            f"Attempting to parse response as JSON - Case 2: {response}",
            level=LogLevel.DEBUG,
            env=self,
        )
        stripped = response.strip()
        if stripped.startswith("{"):
            depth = 0
            for i, ch in enumerate(stripped):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(stripped[: i + 1])
                        except Exception as e:
                            self.logger.log(
                                f"Failed to parse response as JSON: {e}",
                                level=LogLevel.ERROR,
                                env=self,
                            )
                            break

        return {"tool": "terminate", "args": {"message": response}}

    # ── LLM call ──────────────────────────────────────────────────────────

    def generate_response(self, prompt: Prompt) -> str:
        """Call the LLM via LiteLLM and return a JSON tool-call string or
        free-form text content."""
        kwargs = {
            "model": self.model,
            "messages": prompt.messages,
            "max_tokens": self.max_tokens,
        }
        env_var = self._api_key_env_var()
        if env_var:
            kwargs["api_key"] = os.getenv(env_var)
        if prompt.tools:
            kwargs["tools"] = prompt.tools
        kwargs.update(self._extra_completion_kwargs(prompt))

        response = completion(**kwargs)
        self._notify_response(self.model, response)
        return self._parse_completion(response, prompt.tools)

    def _parse_completion(self, response: Any, tools: Optional[List[dict]]) -> str:
        """Extract the model's tool call (if any) or text content from a
        LiteLLM response. Raises with provider-tunable messages on the two
        empty-response failure modes."""
        if not response.choices:
            raise RuntimeError(self._empty_choices_message())

        message = response.choices[0].message
        if tools and message.tool_calls:
            tool = message.tool_calls[0]
            return json.dumps(
                {
                    "tool": tool.function.name,
                    "args": json.loads(tool.function.arguments),
                }
            )

        if not message.content or not message.content.strip():
            raise RuntimeError(self._empty_content_message())
        return message.content
