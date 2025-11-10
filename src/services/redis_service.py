import aioredis
from typing import Optional, List
import json
from src.config import REDIS_HOST, REDIS_PORT

class RedisService:
    """
    An asynchronous Redis client for storing and retrieving conversation history.
    """
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT):
        self._redis = None
        self._host = host
        self._port = port

    async def connect(self):
        """
        Establishes the Redis connection pool.
        """
        if not self._redis:
            self._redis = await aioredis.from_url(f"redis://{self._host}:{self._port}")

    async def disconnect(self):
        """
        Closes the Redis connection pool.
        """
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def get_history(self, session_id: str) -> List[dict]:
        """
        Retrieves the conversation history for a given session ID.
        """
        history_json = await self._redis.get(session_id)
        if history_json:
            return json.loads(history_json)
        return []

    async def add_message_to_history(self, session_id: str, message: dict, ttl: int = 86400):
        """
        Adds a new message to the conversation history for a given session ID.
        """
        history = await self.get_history(session_id)
        history.append(message)
        await self._redis.set(session_id, json.dumps(history), ex=ttl)

    async def clear_history(self, session_id: str):
        """
        Clears the conversation history for a given session ID.
        """
        await self._redis.delete(session_id)
