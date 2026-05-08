"""OpenAI-adapted function-calling action language.

OpenAI uses the standard roles the base class already produces
(``system`` / ``assistant`` / ``user``) and LiteLLM auto-resolves
``OPENAI_API_KEY`` from the environment, so this subclass only sets
provider-appropriate defaults — the LLM call lives in the base class.
"""

from zurvan.agent_function_calling_action_language import (
    AgentFunctionCallingActionLanguage,
)


class AgentFunctionCallingActionLanguageOpenAI(AgentFunctionCallingActionLanguage):
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 1024,
        logger=None,
        response_observers=None,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            logger=logger,
            response_observers=response_observers,
        )
