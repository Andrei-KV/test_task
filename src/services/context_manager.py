from .redis_service import RedisService
from typing import List

class ContextManagerService:
    """
    Manages the conversation context, including history and clarification questions.
    """
    def __init__(self, redis_service: RedisService):
        self._redis = redis_service

    async def get_full_history(self, session_id: str, limit: int = 50) -> List[dict]:
        """
        Retrieves the full conversation history for a session, up to a limit.
        """
        history = await self._redis.get_history(session_id)
        return history[-limit:]

    async def get_context_window(self, session_id: str, window_size: int = 5) -> List[dict]:
        """
        Retrieves the last few messages to be used as context for the RAG service.
        """
        history = await self._redis.get_history(session_id)
        return history[-window_size:]

    async def add_message(self, session_id: str, role: str, content: str):
        """
        Adds a new message to the conversation history.
        """
        message = {"role": role, "content": content}
        await self._redis.add_message_to_history(session_id, message)

    async def get_clarification_count(self, session_id: str) -> int:
        """
        Gets the current count of clarification questions for a session by looking for the last system message.
        """
        history = await self._redis.get_history(session_id)
        for message in reversed(history):
            if message.get("role") == "system" and "clarification_count" in message:
                return message["clarification_count"]
        return 0

    async def increment_clarification_count(self, session_id: str):
        """
        Increments the clarification question counter.
        """
        count = await self.get_clarification_count(session_id)
        message = {
            "role": "system",
            "content": "A clarifying question was asked.",
            "clarification_count": count + 1,
        }
        await self._redis.add_message_to_history(session_id, message)

    async def reset_context(self, session_id: str):
        """
        Resets the conversation context by clearing the history.
        """
        await self._redis.clear_history(session_id)
