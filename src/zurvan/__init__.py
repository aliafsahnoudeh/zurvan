"""zurvan — composable Python framework for goal-directed LLM agents.

The package implements the **GAME loop** (Goals, Actions, Memory,
Environment) and a :class:`Capability` hook system. Behaviour is
composed by passing a list of capabilities to the :class:`Agent`,
not by subclassing.

LLM providers are abstracted behind :class:`AgentLanguage`. The
function-calling subclasses (Gemini, Groq, OpenAI) talk to providers
via LiteLLM.
"""

from zurvan.action import Action
from zurvan.action_context import ActionContext
from zurvan.action_registry import ActionRegistry
from zurvan.agent import Agent
from zurvan.agent_function_calling_action_language import (
    AgentFunctionCallingActionLanguage,
)
from zurvan.agent_function_calling_action_language_gemini import (
    AgentFunctionCallingActionLanguageGemini,
)
from zurvan.agent_function_calling_action_language_groq import (
    AgentFunctionCallingActionLanguageGroq,
)
from zurvan.agent_function_calling_action_language_openai import (
    AgentFunctionCallingActionLanguageOpenAI,
)
from zurvan.agent_language import AgentLanguage
from zurvan.capabilities.canary_capability import (
    CanaryCapability,
    PromptInjectionDetected,
)
from zurvan.capabilities.time_aware_capability import TimeAwareCapability
from zurvan.capability import Capability
from zurvan.decorators import register_tool
from zurvan.environment import Environment
from zurvan.goal import Goal
from zurvan.logger.logger import LogLevel, Logger
from zurvan.memory import Memory
from zurvan.prompt import Prompt
from zurvan.token_tracker import TokenTracker

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core GAME primitives
    "Action",
    "ActionContext",
    "ActionRegistry",
    "Agent",
    "Capability",
    "Environment",
    "Goal",
    "Memory",
    "Prompt",
    # Logging
    "LogLevel",
    "Logger",
    # Tool decorator
    "register_tool",
    # Token tracking
    "TokenTracker",
    # Agent languages
    "AgentLanguage",
    "AgentFunctionCallingActionLanguage",
    "AgentFunctionCallingActionLanguageGemini",
    "AgentFunctionCallingActionLanguageGroq",
    "AgentFunctionCallingActionLanguageOpenAI",
    # Capabilities
    "CanaryCapability",
    "PromptInjectionDetected",
    "TimeAwareCapability",
]
