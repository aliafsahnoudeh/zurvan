from datetime import datetime


class ReversibleAction:
    def __init__(self, execute_func, reverse_func):
        self.execute = execute_func
        self.reverse = reverse_func
        self.execution_record = None

    def run(self, **args):
        """Execute action and record how to reverse it."""
        result = self.execute(**args)
        self.execution_record = {
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        return result

    def undo(self):
        """Reverse the action using recorded information."""
        if not self.execution_record:
            raise ValueError("No action to reverse")
        return self.reverse(**self.execution_record)
