"""It is responsible for formatting the prompt that is sent to the LLM
and parsing the response from the LLM.
So it's a translator between our structured agent components and
the language model's input/output format.

The AgentLanguage component has two primary responsibilities:
1. Prompt Construction: Transforming our GAME components into a format the LLM can understand
2. Response Parsing: Interpreting the LLM's response to determine what action the agent should take


The same agent can behave differently just by changing its language implementation.
This separation of concerns means we can:
1. Experiment with different prompt formats without changing the agent's logic
2. Support different LLM providers with their own communication styles, allowing us to adjust prompting style to match the LLM's strengths
3. Add new response formats without modifying existing code
4. Handle errors and retry logic at the language level

"""

from typing import List

from zurvan.action import Action
from zurvan.environment import Environment
from zurvan.goal import Goal
from zurvan.memory import Memory
from zurvan.prompt import Prompt


class AgentLanguage:
    def __init__(self):
        pass

    def generate_response(self, prompt: Prompt) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def construct_prompt(
        self,
        actions: List[Action],
        environment: Environment,
        goals: List[Goal],
        memory: Memory,
    ) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")

    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")
