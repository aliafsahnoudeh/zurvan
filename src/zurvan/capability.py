from typing import List

from zurvan.action import Action
from zurvan.action_context import ActionContext
from zurvan.memory import Memory
from zurvan.prompt import Prompt


class Capability:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def init(self, agent, action_context: ActionContext) -> dict:
        """Called once when the agent starts running.
        runs once when the agent starts.
        This is where you set up any initial state or add starting information to the agent’s memory.
        """
        pass

    def start_agent_loop(self, agent, action_context: ActionContext) -> bool:
        """Called at the start of each iteration through the agent loop.
        You can use this to check conditions or prepare for the next iteration.
        """
        return True

    def process_prompt(
        self, agent, action_context: ActionContext, prompt: Prompt
    ) -> Prompt:
        """Called right before the prompt is sent to the LLM.
        Lets you modify the prompt.
        """
        return prompt

    def process_response(
        self, agent, action_context: ActionContext, response: str
    ) -> str:
        """Called after getting a response from the LLM.
        Lets you modify or validate the raw response text.
        """
        return response

    def process_action(
        self, agent, action_context: ActionContext, action: dict
    ) -> dict:
        """Called after parsing the response into an action.
        Lets you modify the action before it’s executed. You might add metadata or validate the action
        """
        return action

    def process_result(
        self,
        agent,
        action_context: ActionContext,
        response: str,
        action_def: Action,
        action: dict,
        result: any,
    ) -> any:
        """Called after executing the action.
        Lets you modify the result. This is useful for adding additional context or transforming the result format
        """
        return result

    def process_new_memories(
        self,
        agent,
        action_context: ActionContext,
        memory: Memory,
        response,
        result,
        memories: List[dict],
    ) -> List[dict]:
        """Called when new memories are being added.
        Lets you modify the action before it’s executed. You might add metadata or validate the action
        """
        return memories

    def end_agent_loop(self, agent, action_context: ActionContext):
        """Called at the end of each iteration through the agent loop.
        This is useful for cleanup or logging what happened during the iteration.
        """
        pass

    def should_terminate(
        self, agent, action_context: ActionContext, response: str
    ) -> bool:
        """Called to check if the agent should stop running."""
        return False

    def terminate(self, agent, action_context: ActionContext) -> dict:
        """Called when the agent is shutting down.
        Handles any final cleanup when the agent stops
        """
        pass
