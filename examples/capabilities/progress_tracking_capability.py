from zurvan.action_context import ActionContext
from zurvan.action_registry import ActionRegistry
from zurvan.capability import Capability
from zurvan.memory import Memory


@register_tool(tags=["prompts"])
def track_progress(
    action_context: ActionContext, _memory: Memory, action_registry: ActionRegistry
) -> str:
    """Generate a progress report based on the current task, available tools, and memory context."""

    # Get tool descriptions for the prompt
    tool_descriptions = "\n".join(
        f"- {action.name}: {action.description}"
        for action in action_registry.get_actions()
    )

    # Get relevant memory content
    memory_content = "\n".join(
        f"{m['type']}: {m['content']}"
        for m in _memory.items
        if m["type"] in ["user", "system"]
    )

    # Construct the prompt as a string
    prompt = f"""Given the current task and available tools, generate a progress report.
Think through this step by step:

1. Identify the key components of the task and the intended outcome.
2. Assess the progress made so far based on available information.
3. Identify any blockers or issues preventing completion.
4. Suggest the next steps to move forward efficiently.
5. Recommend any tool usage that might help complete the task.

Write your progress report in clear, structured points.

Available tools:
{tool_descriptions}

Task context from memory:
{memory_content}

Provide a well-organized report on the current progress and next steps."""

    return prompt_llm(action_context=action_context, prompt=prompt)


class ProgressTrackingCapability(Capability):
    def __init__(self, memory_type="system", track_frequency=1):
        super().__init__(
            name="Progress Tracking",
            description="Tracks progress and enables reflection after actions",
        )
        self.memory_type = memory_type
        self.track_frequency = track_frequency
        self.iteration_count = 0

    def end_agent_loop(self, agent, action_context: ActionContext):
        """Generate and store progress report at the end of each iteration."""
        self.iteration_count += 1

        # Only track progress on specified iterations
        if self.iteration_count % self.track_frequency != 0:
            return

        # Get the memory and action registry from context
        memory = action_context.get_memory()
        action_registry = action_context.get_action_registry()

        # Generate progress report
        progress_report = track_progress(
            action_context=action_context,
            _memory=memory,
            action_registry=action_registry,
        )

        # Add the progress report to memory
        memory.add_memory(
            {
                "type": self.memory_type,
                "content": f"Progress Report (Iteration {self.iteration_count}):\n{progress_report}",
            }
        )
