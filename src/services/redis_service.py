from typing import Optional, List
import json
from redis.asyncio import Redis
from ..config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

class RedisService:
    """
    An asynchronous Redis client for storing and retrieving conversation history.
    """
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT):
        self._redis: Optional[Redis] = None
        self._host = host
        self._port = port

    async def connect(self):
        """
        Establishes the Redis connection.
        """
        if not self._redis:
            self._redis = Redis(
                host=self._host, 
                port=self._port, 
                password=REDIS_PASSWORD, 
                decode_responses=True
            )

    async def disconnect(self):
        """
        Closes the Redis connection.
        """
        if self._redis:
            await self._redis.close()
            await self._redis.connection_pool.disconnect()
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
        
        # Keep only last 50 messages
        if len(history) > 50:
            history = history[-50:]
            
        await self._redis.set(session_id, json.dumps(history), ex=ttl)

    async def clear_history(self, session_id: str):
        """
        Clears the conversation history for a given session ID.
        """
        await self._redis.delete(session_id)
