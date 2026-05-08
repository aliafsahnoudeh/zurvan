# GAME Framework

The GAME framework provides a structured way to design AI agents, ensuring modularity and adaptability. It breaks agent design into four essential components:

**G - Goals / Instructions**: What the agent is trying to accomplish and its instructions on how to try to achieve its goals.
**A - Actions**: The tools the agent can use to achieve its goals.
**M - Memory**: How the agent retains information across interactions, which determines what information it will have available in each iteration of the agent loop.
**E - Environment**: The agent’s interface to the external world where it executes actions and gets feedback on the results of those actions.

Goals and instructions are grouped together under “G” because they work in tandem to shape the agent’s behavior. Goals specify what the agent is trying to achieve, serving as the high-level objectives that define the desired outcomes of the agent’s operation. Instructions, on the other hand, provide the how, detailing the specific steps, strategies, and constraints that guide the agent toward fulfilling its goals effectively. Together, they form the foundation that ensures the agent not only understands its purpose but also follows a structured approach to accomplishing its tasks.
One important discussion is the relationship between Actions and the Environment. Actions define what the agent can do—they are abstract descriptions of potential choices available to the agent. The Environment, on the other hand, determines how those actions are carried out, providing concrete implementations that execute within the real-world context of the agent. This distinction allows us to separate high-level decision-making from the execution details, making the agent more modular and adaptable.
You can think of Actions as an “interface” specifying the available capabilities, while the Environment acts as the “implementation” that brings those capabilities to life.
