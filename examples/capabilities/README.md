# Example capabilities (illustrative — not tested, not shipped)

The capabilities in this directory are **sketches**, not part of the
installed package. They reference helpers (`prompt_llm`,
`action_context.get_action_registry()`) that don't exist in zurvan and
will not run as-is.

They live here as design references for users writing their own
capabilities — patterns for "plan first" and "track progress after
each iteration" — not as drop-in code.

If you want a working version, you'll need to:

1. Replace `prompt_llm(...)` calls with `agent.agent_language.generate_response(Prompt(messages=[...]))`.
2. Get the `ActionRegistry` from the agent (`agent.actions`) rather
   than via `action_context.get_action_registry()` (no such method).
3. Decide where the plan/progress prompt should run — `init` for
   plan-first, `end_agent_loop` for progress-tracking — and wire it
   end-to-end.

Tested, working capabilities live in [src/zurvan/capabilities/](../../src/zurvan/capabilities/).
