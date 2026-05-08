"""Action — the interface definition of what an agent can do.

Two equivalent shapes are supported:

1. **Dict / JSON Schema** (the original): pass a JSON-Schema dict as
   ``parameters=``. Nothing validates the model's args before
   ``function`` runs — it's on the LLM to send the right shape.
2. **Pydantic model**: pass a ``BaseModel`` subclass as
   ``input_model=``. The schema sent to the LLM is derived from the
   model and the model's args are **validated before** ``function``
   runs. ``pydantic.ValidationError`` is allowed to propagate and is
   trapped by ``Environment.execute_action`` like any action exception
   — the failure goes back to the LLM as a tool result, giving the
   model a chance to retry with corrected args.

The two paths are mutually exclusive — pass exactly one.
"""

from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel


class Action:
    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Optional[Dict] = None,
        input_model: Optional[Type[BaseModel]] = None,
        terminal: bool = False,
    ):
        if parameters is None and input_model is None:
            raise ValueError(
                "Action requires either `parameters` (a JSON-Schema dict) "
                "or `input_model` (a Pydantic BaseModel subclass)."
            )
        if parameters is not None and input_model is not None:
            raise ValueError(
                "Action accepts either `parameters` or `input_model`, not both."
            )

        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.input_model = input_model
        self.parameters = (
            input_model.model_json_schema() if input_model is not None else parameters
        )

    def execute(self, **args) -> Any:
        """Execute the action's function.

        With ``input_model`` set, args are first validated through the
        model. ``pydantic.ValidationError`` propagates on failure —
        ``Environment.execute_action`` catches it and surfaces the
        failure to the LLM so the model can retry with corrected args.
        """
        if self.input_model is not None:
            validated = self.input_model(**args)
            return self.function(**validated.model_dump())
        return self.function(**args)
