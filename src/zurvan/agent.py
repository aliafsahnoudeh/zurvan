import json
import time
from functools import reduce
from typing import Callable, List, Optional

from zurvan.action_context import ActionContext
from zurvan.action_registry import ActionRegistry
from zurvan.agent_language import AgentLanguage
from zurvan.capability import Capability
from zurvan.environment import Environment
from zurvan.goal import Goal
from zurvan.logger.logger import LogLevel, Logger
from zurvan.memory import Memory
from zurvan.prompt import Prompt


class Agent:
    """
    Goal-based agent running the GAME loop (Goals, Actions, Memory, Environment).

    "Mindsets" — plan-first, reflective, budget-conscious, injection-resistant,
    etc. — are composed by passing a list of ``Capability`` instances rather
    than by subclassing. A ``Capability`` can hook into every loop phase
    (prompt, response, action, result, memory, terminate) without modifying
    the agent itself. With an empty capability list the loop reduces to a
    plain Goals → Actions → Memory cycle.
    """

    def __init__(
        self,
        goals: List[Goal],
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        environment: Environment,
        generate_response: Optional[Callable[[Prompt], str]] = None,
        logger=None,
        capabilities: Optional[List[Capability]] = None,
        max_iterations: int = 50,
        max_duration_seconds: Optional[int] = None,
    ):
        """
        Args:
            goals: What the agent is trying to accomplish.
            agent_language: How the agent formats prompts and parses responses.
            action_registry: The actions (tools) the agent can invoke.
            environment: Executes actions and returns results.
            generate_response: Optional explicit LLM callable; defaults to
                ``agent_language.generate_response``.
            logger: Optional logger instance.
            capabilities: Composable hooks into every loop phase. Empty list
                (the default) gives a plain GAME loop with no extras.
            max_iterations: Hard cap on loop iterations.
            max_duration_seconds: Hard wall-clock cap, or ``None`` (the
                default) for no timeout.
        """
        self.goals = goals
        self.logger = logger or Logger()
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment
        self._generate_response = generate_response
        self.capabilities: List[Capability] = list(capabilities or [])
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds

    def construct_prompt(
        self, goals: List[Goal], memory: Memory, actions: ActionRegistry
    ) -> Prompt:
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory,
        )

    def get_action(self, response):
        self.logger.log(
            f"Parsing agent response to determine action: {response}",
            level=LogLevel.DEBUG,
            env=self,
        )
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        self.logger.log(
            f"Parsed action: {action.name} with args {invocation['args']}",
            level=LogLevel.DEBUG,
            env=self,
        )
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        action_def, _ = self.get_action(response)
        return action_def.terminal

    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        new_memories = [
            {"type": "assistant", "content": response},
            {"type": "user", "content": json.dumps(result)},
        ]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
        if self._generate_response is not None:
            return self._generate_response(full_prompt)
        return self.agent_language.generate_response(full_prompt)

    def run(
        self,
        user_input: str,
        memory: Optional[Memory] = None,
        action_context_props: Optional[dict] = None,
    ) -> Memory:
        """Execute the GAME loop.

        Iteration cap: ``self.max_iterations``. Wall-clock cap:
        ``self.max_duration_seconds`` (or unbounded if ``None``). Capability
        hooks fire at every phase; with no capabilities the loop reduces to
        a plain Goals → Actions → Memory cycle.
        """
        memory = memory or Memory()
        self.set_current_task(memory, user_input)
        action_context = ActionContext(
            {**(action_context_props or {}), "memory": memory}
        )

        for capability in self.capabilities:
            capability.init(self, action_context)

        self.logger.log(
            f"Agent starting with goals: {[g.description for g in self.goals]}",
            level=LogLevel.INFO,
            env=self,
        )

        start_time = time.time()
        consecutive_llm_failures = 0
        max_consecutive_llm_failures = 3
        terminated = False

        for i in range(self.max_iterations):
            if (
                self.max_duration_seconds is not None
                and time.time() - start_time >= self.max_duration_seconds
            ):
                self.logger.log(
                    f"Max duration of {self.max_duration_seconds}s exceeded. "
                    "Forcing termination.",
                    level=LogLevel.WARNING,
                    env=self,
                )
                break

            self.logger.log(
                f" Iteration {i + 1}/{self.max_iterations} - "
                f"Current memory: {memory.get_memories()}",
                level=LogLevel.DEBUG,
                env=self,
            )

            if not all(
                c.start_agent_loop(self, action_context) for c in self.capabilities
            ):
                continue

            base_prompt = self.construct_prompt(self.goals, memory, self.actions)
            prompt = reduce(
                lambda p, c: c.process_prompt(self, action_context, p),
                self.capabilities,
                base_prompt,
            )

            self.logger.log(
                f"Agent thinking with prompt: {prompt}",
                level=LogLevel.DEBUG,
                env=self,
            )

            try:
                response = self.prompt_llm_for_action(prompt)
                response = reduce(
                    lambda r, c: c.process_response(self, action_context, r),
                    self.capabilities,
                    response,
                )
                self.logger.log(
                    f"Agent Decision: {response}",
                    level=LogLevel.DEBUG,
                    env=self,
                )
                action_def, invocation = self.get_action(response)
            except Exception as e:
                consecutive_llm_failures += 1
                self.logger.log(
                    f"LLM call failed on iteration {i + 1}/{self.max_iterations} "
                    f"(consecutive failures: {consecutive_llm_failures}/"
                    f"{max_consecutive_llm_failures}): {e}",
                    level=LogLevel.WARNING,
                    env=self,
                )
                if consecutive_llm_failures >= max_consecutive_llm_failures:
                    self.logger.log(
                        f"Max consecutive LLM failures ({max_consecutive_llm_failures}) "
                        "reached. Forcing termination.",
                        level=LogLevel.ERROR,
                        env=self,
                    )
                    break
                continue

            consecutive_llm_failures = 0

            action = {"action_def": action_def, "invocation": invocation}
            action = reduce(
                lambda a, c: c.process_action(self, action_context, a),
                self.capabilities,
                action,
            )

            result = self.environment.execute_action(
                action["action_def"], action["invocation"]["args"]
            )
            self.logger.log(
                f"Action Result: {result}",
                level=LogLevel.DEBUG,
                env=self,
            )

            result = reduce(
                lambda r, c: c.process_result(
                    self, action_context, response, action["action_def"], action, r
                ),
                self.capabilities,
                result,
            )

            new_memories = [
                {"type": "assistant", "content": response},
                {"type": "user", "content": json.dumps(result)},
            ]
            new_memories = reduce(
                lambda m, c: c.process_new_memories(
                    self, action_context, memory, response, result, m
                ),
                self.capabilities,
                new_memories,
            )
            for m in new_memories:
                memory.add_memory(m)

            should_stop = action["action_def"].terminal or any(
                c.should_terminate(self, action_context, response)
                for c in self.capabilities
            )
            if should_stop:
                self.logger.log(
                    "Agent has decided to terminate.",
                    level=LogLevel.INFO,
                    env=self,
                )
                action_context.properties["terminal_response"] = response
                action_context.properties["terminal_action"] = action
                action_context.properties["terminal_result"] = result
                for capability in self.capabilities:
                    capability.terminate(self, action_context)
                terminated = True
                break

            for capability in self.capabilities:
                capability.end_agent_loop(self, action_context)

        if not terminated:
            self.logger.log(
                "Agent loop ended without normal termination. "
                "Forcing termination.",
                level=LogLevel.WARNING,
                env=self,
            )
            for capability in self.capabilities:
                capability.terminate(self, action_context)

            terminal_actions = [a for a in self.actions.get_actions() if a.terminal]
            if terminal_actions:
                self.logger.log(
                    f"Available terminal actions: "
                    f"{[a.name for a in terminal_actions]}",
                    level=LogLevel.DEBUG,
                    env=self,
                )
                terminal_action = terminal_actions[0]
                fallback_message = "Sorry I couldn't figure this out!"
                result = self.environment.execute_action(
                    terminal_action, {"message": fallback_message}
                )
                fallback_response = json.dumps(
                    {
                        "tool": terminal_action.name,
                        "args": {"message": fallback_message},
                    }
                )
                self.update_memory(memory, fallback_response, result)

        return memory
