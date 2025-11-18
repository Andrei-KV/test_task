from ...services.redis_service import RedisService
from ...services.context_manager import ContextManagerService
from ...config import REDIS_HOST, REDIS_PORT

async def get_context_manager():
    """
    Dependency provider for the ContextManagerService.
    Initializes the RedisService and yields a ContextManagerService instance.
    """
    redis_service = RedisService(host=REDIS_HOST, port=REDIS_PORT)
    await redis_service.connect()
    context_manager = ContextManagerService(redis_service=redis_service)
    try:
        yield context_manager
    finally:
        await redis_service.disconnect()
