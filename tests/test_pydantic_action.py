"""Tests for the Pydantic-input shape of zurvan.Action."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from zurvan import Action, Environment


# ── Construction-time mutual exclusion ───────────────────────────────────


def test_neither_parameters_nor_input_model_raises():
    with pytest.raises(ValueError, match="either `parameters`"):
        Action(name="x", function=lambda: None, description="x")


def test_both_parameters_and_input_model_raises():
    class M(BaseModel):
        x: int

    with pytest.raises(ValueError, match="not both"):
        Action(
            name="x",
            function=lambda x: x,
            description="x",
            parameters={"type": "object"},
            input_model=M,
        )


# ── Schema derived from Pydantic model ───────────────────────────────────


def test_input_model_derives_json_schema_into_parameters():
    class GetWeatherArgs(BaseModel):
        city: str
        unit: Literal["c", "f"] = "c"

    action = Action(
        name="get_weather",
        function=lambda city, unit="c": f"{city}/{unit}",
        description="Look up weather.",
        input_model=GetWeatherArgs,
    )
    schema = action.parameters
    assert schema["type"] == "object"
    assert "city" in schema["properties"]
    assert "unit" in schema["properties"]
    # Required fields = those without defaults
    assert "city" in schema["required"]
    assert "unit" not in schema["required"]
    # Literal types render as enum
    assert schema["properties"]["unit"].get("enum") == ["c", "f"]


def test_input_model_attribute_is_preserved():
    class M(BaseModel):
        x: int

    action = Action(
        name="x", function=lambda x: x, description="x", input_model=M,
    )
    assert action.input_model is M


# ── Execute-time validation ──────────────────────────────────────────────


def test_execute_validates_args_through_model():
    class Args(BaseModel):
        x: int
        y: int

    captured = {}

    def add(x, y):
        captured["x"] = x
        captured["y"] = y
        return x + y

    action = Action(name="add", function=add, description="add", input_model=Args)
    result = action.execute(x=2, y=3)
    assert result == 5
    assert captured == {"x": 2, "y": 3}


def test_execute_coerces_string_ints_per_pydantic_default():
    """Pydantic v2 coerces string-int by default — the function sees a real int."""

    class Args(BaseModel):
        x: int

    seen = {}

    def f(x):
        seen["type"] = type(x).__name__
        seen["value"] = x
        return x

    action = Action(name="x", function=f, description="x", input_model=Args)
    action.execute(x="42")
    assert seen == {"type": "int", "value": 42}


def test_execute_fills_in_defaults_when_arg_omitted():
    class Args(BaseModel):
        city: str
        unit: Literal["c", "f"] = "c"

    captured = {}

    def f(city, unit):
        captured["unit"] = unit
        return f"{city}/{unit}"

    action = Action(name="x", function=f, description="x", input_model=Args)
    action.execute(city="paris")  # unit omitted
    assert captured["unit"] == "c"


def test_execute_raises_validation_error_on_bad_args():
    class Args(BaseModel):
        x: int = Field(gt=0)

    action = Action(
        name="x", function=lambda x: x, description="x", input_model=Args,
    )
    with pytest.raises(ValidationError):
        action.execute(x=-1)


def test_execute_raises_validation_error_on_missing_required_field():
    class Args(BaseModel):
        a: str
        b: str

    action = Action(
        name="x", function=lambda a, b: a + b, description="x", input_model=Args,
    )
    with pytest.raises(ValidationError):
        action.execute(a="only one")


# ── Environment trapping (failure surfaces back to the LLM) ──────────────


def test_environment_traps_validation_error_from_pydantic_action():
    """Verify the existing Environment.execute_action exception trap catches
    ValidationError and surfaces it as a normal failed-tool result, so the
    LLM can see the error and retry with corrected args."""

    class Args(BaseModel):
        x: int = Field(gt=0)

    env = Environment()
    action = Action(
        name="positive",
        function=lambda x: x * 2,
        description="x",
        input_model=Args,
    )
    result = env.execute_action(action, {"x": -5})

    assert result["tool_executed"] is False
    assert "error" in result
    # ValidationError text mentions the violated rule somewhere
    assert "x" in result["error"]


def test_environment_returns_success_for_valid_pydantic_args():
    class Args(BaseModel):
        x: int

    env = Environment()
    action = Action(
        name="double", function=lambda x: x * 2, description="x", input_model=Args,
    )
    result = env.execute_action(action, {"x": 21})
    assert result["tool_executed"] is True
    assert result["result"] == 42


# ── Backwards compat: dict path still works ──────────────────────────────


def test_dict_parameters_path_unchanged():
    """Sanity: existing call sites with parameters=... behave exactly as before.
    No validation, no Pydantic anywhere."""
    action = Action(
        name="legacy",
        function=lambda **kw: kw,
        description="legacy",
        parameters={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    assert action.input_model is None
    assert action.parameters == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    # No validation: any dict-shaped input goes straight to the function
    assert action.execute(a="hi", extra="not-in-schema") == {
        "a": "hi",
        "extra": "not-in-schema",
    }
