# zurvan

A small, composable Python framework for building goal-directed LLM agents.

![Zurvan Illustration](assets/zurvan.png)

## What it is

`zurvan` runs a **GAME loop** — Goals, Actions, Memory, Environment — over
an LLM. Behaviour like plan-first, time-aware, prompt-injection-resistant,
or progress-tracking is composed by passing a list of `Capability`
instances rather than by subclassing `Agent`. With no capabilities it
reduces to a plain Goals → Actions → Memory cycle.

LLM providers are abstracted behind `AgentLanguage`. Built-in subclasses
(via [LiteLLM](https://github.com/BerriAI/litellm)) cover OpenAI, Gemini,
and Groq.

## Install

```bash
pip install zurvan
```

Requires Python 3.11+.

## Quick start

```python
from zurvan import (
    Action,
    ActionRegistry,
    Agent,
    AgentFunctionCallingActionLanguageOpenAI,
    Environment,
    Goal,
)

def terminate(message: str) -> str:
    return message

actions = ActionRegistry()
actions.register(Action(
    name="terminate",
    function=terminate,
    description="End the conversation with a final message.",
    parameters={
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
    terminal=True,
))

agent = Agent(
    goals=[Goal(priority=1, name="Greet", description="Greet the user warmly.")],
    agent_language=AgentFunctionCallingActionLanguageOpenAI(model="openai/gpt-4o-mini"),
    action_registry=actions,
    environment=Environment(),
)

memory = agent.run("Say hi.")
```

## Pydantic-validated tool inputs

`Action` accepts either a JSON-Schema dict (`parameters=`) or a Pydantic
model (`input_model=`). With a model, the schema sent to the LLM is
derived from the model and the model's args are **validated before the
function runs** — a `ValidationError` is caught by `Environment` and
fed back to the LLM as a failed-tool result, so the model gets a chance
to retry with corrected args.

```python
from typing import Literal
from pydantic import BaseModel
from zurvan import Action

class GetWeatherArgs(BaseModel):
    city: str
    unit: Literal["c", "f"] = "c"

def get_weather(city: str, unit: str = "c") -> str:
    return f"It's nice in {city} ({unit}°)"

action = Action(
    name="get_weather",
    function=get_weather,
    description="Look up the weather for a city.",
    input_model=GetWeatherArgs,
)
```

`parameters=` and `input_model=` are mutually exclusive — pass exactly one.

## Capabilities

Capabilities hook into every loop phase (`init`, `start_agent_loop`,
`process_prompt`, `process_response`, `process_action`, `process_result`,
`process_new_memories`, `end_agent_loop`, `should_terminate`,
`terminate`). Compose them — don't subclass `Agent`.

```python
from zurvan import Agent, CanaryCapability, TimeAwareCapability

agent = Agent(
    ...,
    capabilities=[CanaryCapability(), TimeAwareCapability()],
)
```

## License

MIT.
