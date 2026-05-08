import json
from typing import Dict, List


class Memory:
    def __init__(self):
        self.items = []  # Basic conversation histor

    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]

    def copy_without_system_memories(self):
        """Return a copy of the memory without system memories"""
        filtered_items = [m for m in self.items if m["type"] != "system"]
        memory = Memory()
        memory.items = filtered_items
        return memory

    def compact(
        self,
        preserve_first: int = 1,
        keep_recent: int = 4,
        max_entry_length: int = 200,
    ) -> int:
        """Truncate middle memory entries to reduce context size.

        Preserves the first *preserve_first* entries intact (the initial
        task and any other instruction entries) and the most recent
        *keep_recent* entries intact.  Entries in between whose ``content``
        exceeds *max_entry_length* characters are truncated.  For
        JSON-encoded tool results the ``result`` field is shortened; plain
        text entries are cut directly.

        Returns the number of entries that were compacted.
        """
        compacted = 0
        start = max(0, preserve_first)
        cutoff = max(start, len(self.items) - keep_recent)
        for i in range(start, cutoff):
            entry = self.items[i]
            content = entry.get("content", "")
            if len(content) <= max_entry_length:
                continue
            try:
                data = json.loads(content)
                result = data.get("result")
                if result and len(str(result)) > max_entry_length:
                    data["result"] = str(result)[:max_entry_length] + "… (compacted)"
                    self.items[i] = {**entry, "content": json.dumps(data)}
                    compacted += 1
            except (json.JSONDecodeError, TypeError, AttributeError):
                self.items[i] = {
                    **entry,
                    "content": content[:max_entry_length] + "… (compacted)",
                }
                compacted += 1
        return compacted
