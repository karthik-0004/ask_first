"""Conversation memory management for Clary conversational AI."""

from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher

import config


class ConversationMemory:
    """Store and manage conversation state for a single user.

    The memory object is designed to be serializable so it can be persisted in
    Streamlit session state and reconstructed later.
    """

    def __init__(self, user_name: str, user_profile: dict = None):
        """Initialize a memory object for one user.

        Args:
            user_name: Display name of the user.
            user_profile: Optional profile metadata for the user.
        """
        self.user_name = user_name
        self.messages = []
        self.detected_patterns = []
        self.health_facts = []
        self.user_profile = user_profile or {}
        self.conversation_start = datetime.now().isoformat()

    def add_user_message(self, content: str) -> None:
        """Append a user message to memory.

        Args:
            content: Message text from the user.
        """
        message_index = len(self.messages) + 1
        self.messages.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "message_index": message_index,
            }
        )

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to memory.

        Args:
            content: Message text from Clary.
        """
        message_index = len(self.messages) + 1
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "message_index": message_index,
            }
        )

    def get_messages_for_api(self) -> list[dict]:
        """Return chat messages in OpenAI API format.

        Keeps the latest MAX_MEMORY_MESSAGES messages while always preserving
        the first two messages when present.

        Returns:
            List of role/content dictionaries.
        """
        if not self.messages:
            return []

        max_messages = int(getattr(config, "MAX_MEMORY_MESSAGES", 50))
        if len(self.messages) <= max_messages:
            selected = self.messages
        else:
            first_two = self.messages[:2]
            remaining_budget = max_messages - len(first_two)
            if remaining_budget > 0:
                selected = first_two + self.messages[-remaining_budget:]
            else:
                selected = first_two

        return [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            for msg in selected
        ]

    def add_detected_pattern(self, pattern: dict) -> None:
        """Store a new detected pattern when not already present.

        Duplicate checks are performed by title similarity.

        Args:
            pattern: Pattern dictionary to store.
        """
        new_title = str(pattern.get("title", "")).strip().lower()
        if not new_title:
            return

        for existing in self.detected_patterns:
            old_title = str(existing.get("title", "")).strip().lower()
            if not old_title:
                continue
            similarity = SequenceMatcher(None, old_title, new_title).ratio()
            if similarity >= 0.85 or new_title in old_title or old_title in new_title:
                return

        self.detected_patterns.append(pattern)

    def add_health_fact(self, fact: str, message_index: int) -> None:
        """Store an extracted health fact in memory.

        Args:
            fact: Fact text.
            message_index: Message index where the fact originated.
        """
        self.health_facts.append(
            {
                "fact": fact,
                "message_index": message_index,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_memory_summary(self) -> str:
        """Build a formatted text summary of conversation memory.

        Returns:
            Multi-line summary string of memory status.
        """
        facts_lines = []
        for fact in self.health_facts:
            facts_lines.append(
                f"- Message {fact.get('message_index', '?')}: {fact.get('fact', '')}"
            )
        facts_block = "\n".join(facts_lines) if facts_lines else "- None yet"

        pattern_lines = []
        for pattern in self.detected_patterns:
            pattern_lines.append(f"- {pattern.get('title', 'Untitled pattern')}")
        patterns_block = "\n".join(pattern_lines) if pattern_lines else "- None yet"

        return (
            "MEMORY SUMMARY:\n"
            f"- Conversation started: {self.conversation_start}\n"
            f"- Messages exchanged: {len(self.messages)}\n"
            "- Health facts noted:\n"
            f"{facts_block}\n"
            "- Patterns identified so far:\n"
            f"{patterns_block}"
        )

    def get_full_context_string(self) -> str:
        """Return full conversation history as readable text.

        Returns:
            Multi-line string with message index, role, timestamp, and content.
        """
        lines = []
        for msg in self.messages:
            role = msg.get("role", "user")
            role_label = "User" if role == "user" else "Clary"
            idx = msg.get("message_index", "?")
            ts = msg.get("timestamp", "")
            content = msg.get("content", "")
            lines.append(f"[Message {idx} - {role_label} - {ts}]: {content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize full memory state.

        Returns:
            Dictionary representation suitable for JSON/session persistence.
        """
        return {
            "user_name": self.user_name,
            "messages": self.messages,
            "detected_patterns": self.detected_patterns,
            "health_facts": self.health_facts,
            "user_profile": self.user_profile,
            "conversation_start": self.conversation_start,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        """Reconstruct memory object from serialized data.

        Args:
            data: Serialized memory dictionary.

        Returns:
            Reconstructed ConversationMemory object.
        """
        memory = cls(
            user_name=data.get("user_name", "User"),
            user_profile=data.get("user_profile", {}),
        )
        memory.messages = list(data.get("messages", []))
        memory.detected_patterns = list(data.get("detected_patterns", []))
        memory.health_facts = list(data.get("health_facts", []))
        memory.conversation_start = data.get("conversation_start", datetime.now().isoformat())
        return memory

    def reset(self) -> None:
        """Reset dynamic memory while preserving user identity/profile."""
        self.messages = []
        self.detected_patterns = []
        self.health_facts = []
