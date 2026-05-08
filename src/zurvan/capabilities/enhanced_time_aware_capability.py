from datetime import datetime
from zoneinfo import ZoneInfo

from zurvan.action import Action
from zurvan.action_context import ActionContext
from zurvan.capabilities.time_aware_capability import TimeAwareCapability


class EnhancedTimeAwareCapability(TimeAwareCapability):
    def process_action(
        self, agent, action_context: ActionContext, action: dict
    ) -> dict:
        """Add timing information to action results."""
        # Add execution time to action metadata
        action["execution_time"] = datetime.now(
            ZoneInfo(action_context.get("time_zone", "America/Chicago"))
        ).isoformat()
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
        """Add duration information to results."""
        if isinstance(result, dict):
            result["action_duration"] = (
                datetime.now(ZoneInfo(action_context.get("time_zone")))
                - datetime.fromisoformat(action["execution_time"])
            ).total_seconds()
        return result
