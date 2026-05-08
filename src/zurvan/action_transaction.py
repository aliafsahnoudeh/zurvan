import uuid

from zurvan.reversible_action import ReversibleAction


class ActionTransaction:
    def __init__(self):
        self.actions = []
        self.executed = []
        self.committed = False
        self.transaction_id = str(uuid.uuid4())

    def add(self, action: ReversibleAction, **args):
        """Queue an action for execution."""
        if self.committed:
            raise ValueError("Transaction already committed")
        self.actions.append((action, args))

    async def execute(self):
        """Execute all actions in the transaction."""
        try:
            for action, args in self.actions:
                result = action.run(**args)
                self.executed.append(action)
        except Exception as e:
            # If any action fails, reverse everything done so far
            await self.rollback()
            raise e

    async def rollback(self):
        """Reverse all executed actions in reverse order."""
        for action in reversed(self.executed):
            await action.undo()
        self.executed = []

    def commit(self):
        """Mark transaction as committed."""
        self.committed = True
