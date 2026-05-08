"""Tests for the Agent GAME loop.

These exercise:
- terminal-action termination
- ``max_iterations`` forced-termination fallback
- capability hook ordering across the loop phases
- a capability short-circuiting to terminate via ``should_terminate``
- consecutive LLM failures forcing termination
- ``max_duration_seconds`` wall-clock cap
- empty capability list reduces to a plain GAME loop

LLM calls are stubbed out via the ``_ScriptedLanguage`` helper in
``conftest.py`` so the tests are fast and deterministic.
"""

import time

from zurvan import (
    Action,
    ActionContext,
    ActionRegistry,
    Agent,
    Capability,
    Environment,
    Goal,
    Memory,
)


def make_agent(language, registry, **kwargs):
    return Agent(
        goals=[Goal(priority=1, name="g", description="d")],
        agent_language=language,
        action_registry=registry,
        environment=Environment(),
        **kwargs,
    )


# ── Terminal action terminates ───────────────────────────────────────────


def test_terminal_action_terminates_loop(scripted_language, basic_registry):
    """One terminal tool-call → loop exits cleanly."""
    lang = scripted_language(
        [{"tool": "terminate", "args": {"message": "done"}}]
    )
    agent = make_agent(lang, basic_registry, max_iterations=5)
    memory = agent.run("hi")
    assert lang.calls == 1  # only one LLM call needed
    # Memory contains the user task, the assistant response, and the
    # tool result — three entries minimum.
    assert len(memory.get_memories()) >= 3


def test_loop_runs_multiple_iterations_until_terminal(scripted_language, basic_registry):
    lang = scripted_language(
        [
            {"tool": "echo", "args": {"text": "hi"}},
            {"tool": "echo", "args": {"text": "bye"}},
            {"tool": "terminate", "args": {"message": "done"}},
        ]
    )
    agent = make_agent(lang, basic_registry, max_iterations=10)
    agent.run("start")
    assert lang.calls == 3


# ── max_iterations forces fallback ────────────────────────────────────────


def test_max_iterations_forces_termination_fallback(scripted_language, basic_registry):
    """Loop hits max_iterations without a terminal action → fallback fires."""
    # Always return non-terminal echo
    lang = scripted_language(
        [{"tool": "echo", "args": {"text": "x"}} for _ in range(5)]
    )
    agent = make_agent(lang, basic_registry, max_iterations=3)
    memory = agent.run("hi")

    # Loop did exactly max_iterations LLM calls
    assert lang.calls == 3

    # Forced fallback inserts a synthesized terminate response into memory
    last_assistant = [m for m in memory.get_memories() if m["type"] == "assistant"][-1]
    assert "terminate" in last_assistant["content"]
    assert "Sorry I couldn't figure this out" in last_assistant["content"]


# ── Capability hook ordering ─────────────────────────────────────────────


class _RecordingCapability(Capability):
    """A capability that appends every hook call to a shared log."""

    def __init__(self, name, log):
        super().__init__(name=name, description=f"recorder {name}")
        self._log = log

    def _record(self, phase):
        self._log.append((self.name, phase))

    def init(self, agent, action_context):
        self._record("init")

    def start_agent_loop(self, agent, action_context):
        self._record("start_agent_loop")
        return True

    def process_prompt(self, agent, action_context, prompt):
        self._record("process_prompt")
        return prompt

    def process_response(self, agent, action_context, response):
        self._record("process_response")
        return response

    def process_action(self, agent, action_context, action):
        self._record("process_action")
        return action

    def process_result(self, agent, action_context, response, action_def, action, result):
        self._record("process_result")
        return result

    def process_new_memories(self, agent, action_context, memory, response, result, memories):
        self._record("process_new_memories")
        return memories

    def end_agent_loop(self, agent, action_context):
        self._record("end_agent_loop")

    def should_terminate(self, agent, action_context, response):
        self._record("should_terminate")
        return False

    def terminate(self, agent, action_context):
        self._record("terminate")


def test_capability_hooks_fire_in_expected_order_for_terminal_action(
    scripted_language, basic_registry
):
    """Terminal action short-circuits ``should_terminate`` (Python's ``or``
    is lazy) — capabilities don't get a vote when the action itself
    declares terminality."""
    log: list[tuple[str, str]] = []
    cap = _RecordingCapability("cap1", log)
    lang = scripted_language(
        [{"tool": "terminate", "args": {"message": "done"}}]
    )
    agent = make_agent(lang, basic_registry, capabilities=[cap])
    agent.run("hi")

    phases = [phase for (_, phase) in log]
    expected = [
        "init",
        "start_agent_loop",
        "process_prompt",
        "process_response",
        "process_action",
        "process_result",
        "process_new_memories",
        "terminate",
    ]
    assert phases == expected


def test_should_terminate_hook_runs_when_action_is_non_terminal(
    scripted_language, basic_registry
):
    """When the action isn't terminal, the loop polls every capability's
    ``should_terminate`` to decide whether to stop."""
    log: list[tuple[str, str]] = []
    cap = _RecordingCapability("cap1", log)
    lang = scripted_language(
        [
            {"tool": "echo", "args": {"text": "hi"}},  # non-terminal
            {"tool": "terminate", "args": {"message": "done"}},
        ]
    )
    agent = make_agent(lang, basic_registry, capabilities=[cap])
    agent.run("hi")

    # First (non-terminal) iteration goes through should_terminate then
    # end_agent_loop. The second (terminal) one short-circuits both.
    phases = [phase for (_, phase) in log]
    assert "should_terminate" in phases
    assert "end_agent_loop" in phases


def test_multiple_capabilities_run_in_order(scripted_language, basic_registry):
    log: list[tuple[str, str]] = []
    a = _RecordingCapability("A", log)
    b = _RecordingCapability("B", log)
    lang = scripted_language(
        [{"tool": "terminate", "args": {"message": "done"}}]
    )
    agent = make_agent(lang, basic_registry, capabilities=[a, b])
    agent.run("hi")

    # For each phase, A's hook always runs before B's
    init_order = [name for (name, phase) in log if phase == "init"]
    assert init_order == ["A", "B"]
    pp_order = [name for (name, phase) in log if phase == "process_prompt"]
    assert pp_order == ["A", "B"]
    term_order = [name for (name, phase) in log if phase == "terminate"]
    assert term_order == ["A", "B"]


# ── Capability short-circuits termination ────────────────────────────────


class _ForceTerminateCapability(Capability):
    def __init__(self):
        super().__init__(name="force-terminate", description="force termination")

    def should_terminate(self, agent, action_context, response):
        return True


def test_capability_can_short_circuit_termination(scripted_language, basic_registry):
    """Even a non-terminal action terminates if a capability says so."""
    lang = scripted_language(
        [{"tool": "echo", "args": {"text": "hi"}}]  # echo is non-terminal
    )
    agent = make_agent(
        lang, basic_registry, capabilities=[_ForceTerminateCapability()],
        max_iterations=10,
    )
    agent.run("hi")
    # Only one iteration ran because should_terminate returned True
    assert lang.calls == 1


class _SkipFirstIterationCapability(Capability):
    """Returns False from start_agent_loop on the first iteration."""

    def __init__(self):
        super().__init__(name="skip-first", description="skip first iteration")
        self.calls = 0

    def start_agent_loop(self, agent, action_context):
        self.calls += 1
        return self.calls > 1  # First call → False, skip; later → True


def test_capability_can_skip_iteration_via_start_agent_loop(scripted_language, basic_registry):
    lang = scripted_language(
        [{"tool": "terminate", "args": {"message": "done"}}]
    )
    agent = make_agent(
        lang, basic_registry, capabilities=[_SkipFirstIterationCapability()],
        max_iterations=10,
    )
    agent.run("hi")
    # First iteration was skipped (no LLM call), second went through
    assert lang.calls == 1


# ── Consecutive LLM failures force termination ───────────────────────────


def test_three_consecutive_llm_failures_force_termination(
    scripted_language, basic_registry
):
    """Three consecutive raises → loop bails out via the failure path."""
    lang = scripted_language(
        [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            RuntimeError("fail 3"),
            # Even though more responses follow, the loop should give up first
            {"tool": "terminate", "args": {"message": "should not reach"}},
        ]
    )
    agent = make_agent(lang, basic_registry, max_iterations=10)
    memory = agent.run("hi")
    assert lang.calls == 3  # didn't reach the fourth
    # Forced-termination fallback should have written its message
    last_assistant = [m for m in memory.get_memories() if m["type"] == "assistant"][-1]
    assert "terminate" in last_assistant["content"]


def test_failure_counter_resets_on_success(scripted_language, basic_registry):
    """Two failures, then a success, then one more failure should NOT terminate."""
    lang = scripted_language(
        [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            {"tool": "echo", "args": {"text": "ok"}},
            RuntimeError("fail 3"),  # counter reset to 1, not 3
            {"tool": "terminate", "args": {"message": "done"}},
        ]
    )
    agent = make_agent(lang, basic_registry, max_iterations=20)
    agent.run("hi")
    assert lang.calls == 5


# ── max_duration_seconds wall-clock cap ──────────────────────────────────


class _SlowAction(Action):
    def __init__(self):
        super().__init__(
            name="slow",
            function=self._sleep,
            description="A slow action.",
            parameters={"type": "object", "properties": {}, "required": []},
            terminal=False,
        )

    @staticmethod
    def _sleep():
        time.sleep(0.05)
        return "ok"


def test_max_duration_seconds_caps_wall_clock(scripted_language, terminate_action):
    """The loop bails out once wall-clock exceeds max_duration_seconds."""
    registry = ActionRegistry()
    registry.register(terminate_action)
    registry.register(_SlowAction())

    # Always return non-terminal slow action — duration accumulates
    lang = scripted_language(
        [{"tool": "slow", "args": {}} for _ in range(50)]
    )
    agent = make_agent(lang, registry, max_iterations=50, max_duration_seconds=0.1)
    start = time.time()
    agent.run("hi")
    elapsed = time.time() - start

    # Should have aborted well before 50 iterations × 0.05s = 2.5s
    assert elapsed < 1.0
    # And done at least one but fewer than 50 iterations
    assert 1 <= lang.calls < 50


# ── Empty capabilities = plain GAME loop ─────────────────────────────────


def test_empty_capabilities_behaves_like_plain_loop(scripted_language, basic_registry):
    """No capabilities → still terminates cleanly on a terminal action."""
    lang = scripted_language(
        [
            {"tool": "echo", "args": {"text": "a"}},
            {"tool": "terminate", "args": {"message": "done"}},
        ]
    )
    agent = make_agent(lang, basic_registry, capabilities=[])  # explicit empty
    memory = agent.run("hi")
    assert lang.calls == 2
    assert isinstance(memory, Memory)


def test_default_capabilities_is_none_treated_as_empty(scripted_language, basic_registry):
    lang = scripted_language(
        [{"tool": "terminate", "args": {"message": "done"}}]
    )
    # Note: capabilities omitted entirely
    agent = Agent(
        goals=[Goal(priority=1, name="g", description="d")],
        agent_language=lang,
        action_registry=basic_registry,
        environment=Environment(),
    )
    agent.run("hi")
    assert agent.capabilities == []


# ── ActionContext is propagated to capabilities ──────────────────────────


def test_action_context_carries_caller_props(scripted_language, basic_registry):
    captured: dict = {}

    class _ContextSnoop(Capability):
        def __init__(self):
            super().__init__(name="snoop", description="record context")

        def init(self, agent, action_context):
            captured["context"] = action_context

    lang = scripted_language(
        [{"tool": "terminate", "args": {"message": "done"}}]
    )
    agent = make_agent(lang, basic_registry, capabilities=[_ContextSnoop()])
    agent.run("hi", action_context_props={"user_id": "alice"})

    ctx: ActionContext = captured["context"]
    assert ctx.get("user_id") == "alice"
    assert ctx.get_memory() is not None
