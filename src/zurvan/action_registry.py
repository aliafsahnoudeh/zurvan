from typing import List

from zurvan.action import Action
from zurvan.decorators import tools, tools_by_tag  # the global dicts


class ActionRegistry:
    def __init__(self, tags: list[str] | None = None):
        self.actions = {}
        if tags:
            self._load_from_tags(tags)

    def _load_from_tags(self, tags: list[str]):
        """Load tools matching any of the given tags via tools_by_tag."""
        seen = set()
        for tag in tags:
            for tool_name in tools_by_tag.get(tag, []):
                if tool_name not in seen:
                    seen.add(tool_name)
                    desc = tools[tool_name]
                    self.register(
                        Action(
                            name=tool_name,
                            function=desc["function"],
                            description=desc["description"],
                            parameters=desc.get("parameters", {}),
                            terminal=desc.get("terminal", False),
                        )
                    )

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> Action | None:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())

    def register_from_tools(self, tags: list[str] | None = None):
        """Load decorated tools into this registry, optionally filtered by tags."""
        for name, desc in tools.items():
            if tags and not set(tags) & set(desc.get("tags", [])):
                continue
            self.register(
                Action(
                    name=name,
                    function=desc["function"],
                    description=desc["description"],
                    parameters=desc.get("parameters", {}),
                    terminal=desc.get("terminal", False),
                )
            )
