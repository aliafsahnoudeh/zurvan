"""Gemini-adapted function-calling action language.

Gemini's native API uses ``model`` instead of ``assistant`` and has no
``system`` role, but we talk to it through LiteLLM's OpenAI-compatible
interface which expects standard OpenAI roles (``user``/``assistant``/
``system``) and performs the Gemini-specific translation internally. So
we only need to handle the Gemini-specific constraints LiteLLM does NOT
cover:

1. Merge ``system`` into ``user`` at the head — LiteLLM handles this for
   a single leading system message, but our prompts may interleave goals
   as ``system`` alongside user content, so we normalize here.
2. Merge consecutive messages with the same role — Gemini rejects
   adjacent same-role messages and LiteLLM does not deduplicate.

NOTE: Do NOT map ``assistant → model`` here. Newer LiteLLM versions
reject ``role: 'model'`` in the input payload; they expect OpenAI-style
``assistant`` and convert to Gemini's ``model`` internally.
"""

from typing import List, Optional

from zurvan.action import Action
from zurvan.agent_function_calling_action_language import (
    AgentFunctionCallingActionLanguage,
)
from zurvan.environment import Environment
from zurvan.goal import Goal
from zurvan.memory import Memory
from zurvan.prompt import Prompt

# Only ``system`` needs translation — LiteLLM handles assistant→model.
GEMINI_ROLE_MAP = {
    "system": "user",
}


class AgentFunctionCallingActionLanguageGemini(AgentFunctionCallingActionLanguage):
    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        max_tokens: int = 4096,
        logger=None,
        response_observers=None,
        thinking_budget: Optional[int] = None,
    ):
        """
        Args:
            thinking_budget: Optional override for Gemini 2.5+ ``thinkingBudget``
                (number of internal reasoning tokens the model may spend
                before producing visible output). ``None`` (default) leaves
                the provider's default in place. ``0`` disables thinking
                entirely — useful for structured-extraction tasks where
                reasoning adds no value but creates a thinking-token-
                exhaustion failure mode (the model burns the whole
                ``max_tokens`` budget on internal reasoning and returns
                an empty completion).
        """
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            logger=logger,
            response_observers=response_observers,
        )
        self.thinking_budget = thinking_budget

    # ── Provider extension hooks ──────────────────────────────────────────

    def _api_key_env_var(self) -> str:
        return "GOOGLE_API_KEY"

    def _extra_completion_kwargs(self, prompt: Prompt) -> dict:
        if self.thinking_budget is None:
            return {}
        # extra_body is LiteLLM's escape hatch for provider-specific
        # parameters not modelled at the OpenAI-compat layer.
        return {
            "extra_body": {
                "generationConfig": {
                    "thinkingConfig": {"thinkingBudget": self.thinking_budget},
                }
            }
        }

    def _empty_choices_message(self) -> str:
        # Most common cause on Gemini 2.5 Flash is thinking-token exhaustion
        # — the model spends its entire max_tokens budget on internal
        # reasoning and returns no completion. Surface that hint.
        return (
            f"{self.model} returned a response with empty choices "
            "(no completion produced). Most common cause on Gemini "
            "2.5 Flash is thinking-token exhaustion — the model spent "
            f"its entire max_tokens budget ({self.max_tokens}) on "
            "internal reasoning. Retrying or bumping max_tokens "
            "usually fixes this."
        )

    def _empty_content_message(self) -> str:
        return (
            f"{self.model} returned empty response "
            "(no tool_calls and no text content) — likely a safety filter, "
            "content-moderation stop, or thinking-token budget exhaustion"
        )

    # ── Prompt-shape overrides (Gemini-specific) ──────────────────────────

    def _translate_role(self, role: str) -> str:
        return GEMINI_ROLE_MAP.get(role, role)

    def format_goals(self, goals: List[Goal]) -> List:
        messages = super().format_goals(goals)
        for msg in messages:
            msg["role"] = self._translate_role(msg["role"])
        return messages

    def format_memory(self, memory: Memory) -> List:
        messages = super().format_memory(memory)
        for msg in messages:
            msg["role"] = self._translate_role(msg["role"])
        return messages

    def _merge_consecutive(self, messages: List[dict]) -> List[dict]:
        """Merge consecutive messages with the same role.

        Gemini rejects payloads where two adjacent messages share a role.
        """
        if not messages:
            return messages
        merged = [messages[0].copy()]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg.copy())
        return merged

    def construct_prompt(
        self,
        actions: List[Action],
        environment: Environment,
        goals: List[Goal],
        memory: Memory,
    ) -> Prompt:
        prompt = super().construct_prompt(actions, environment, goals, memory)
        prompt.messages = self._merge_consecutive(prompt.messages)
        return prompt
