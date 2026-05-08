from zurvan.action_context import ActionContext
from zurvan.action_registry import ActionRegistry
from zurvan.capability import Capability
from zurvan.decorators import register_tool
from zurvan.memory import Memory


@register_tool(tags=["planning"])
def create_plan(
    action_context: ActionContext, memory: Memory, action_registry: ActionRegistry
) -> str:
    """Create a detailed execution plan based on the task and available tools."""

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
    prompt = f"""Given the task in memory and the available tools, create a detailed plan.
Think through this step by step:

1. First, identify the key components of the task
2. Consider what tools you have available
3. Break down the task into logical steps
4. For each step, specify:
   - What needs to be done
   - What tool(s) will be used
   - What information is needed
   - What the expected outcome is

Write your plan in clear, numbered steps. Each step should be specific and actionable.

Available tools:
{tool_descriptions}

Task context from memory:
{memory_content}

Create a plan that accomplishes this task effectively."""

    return prompt_llm(action_context=action_context, prompt=prompt)


class PlanFirstCapability(Capability):
    def __init__(self, plan_memory_type="system", track_progress=False):
        super().__init__(
            name="Plan First Capability",
            description="The Agent will always create a plan and add it to memory",
        )
        self.plan_memory_type = plan_memory_type
        self.first_call = True
        self.track_progress = track_progress

    def init(self, agent, action_context):
        if self.first_call:
            self.first_call = False
            plan = create_plan(
                action_context=action_context,
                memory=action_context.get_memory(),
                action_registry=action_context.get_action_registry(),
            )

            action_context.get_memory().add_memory(
                {
                    "type": self.plan_memory_type,
                    "content": "You must follow these instructions carefully to complete the task:\n"
                    + plan,
                }
            )
