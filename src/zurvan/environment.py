import time
import traceback
from typing import Any

from zurvan.action import Action
from zurvan.logger.logger import LogLevel, Logger


class Environment:
    logger: Logger

    def __init__(self, logger: Logger | None = None):
        self.logger = logger or Logger()

    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            self.logger.log(
                f"Executing action with args: {args}", level=LogLevel.DEBUG, env=self
            )
            result = action.execute(**args)
            self.logger.log(
                f"Action executed. The result: {result}", level=LogLevel.DEBUG, env=self
            )
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        self.logger.log("Formatting result", level=LogLevel.DEBUG, env=self)
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
