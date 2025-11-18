import asyncio
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine
from qdrant_client import AsyncQdrantClient, models
import logging

from src.config import DB_URI, QDRANT_HOST, COLLECTION_NAME

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Строки подключения
POSTGRES_DATABASE_URL = DB_URI
QDRANT_URL = QDRANT_HOST

async def clear_postgresql_tables():
    """
    Удаляет таблицы 'document_chunks' и 'documents' из базы данных PostgreSQL.
    """
    engine = create_async_engine(POSTGRES_DATABASE_URL, echo=False)
    
    async with engine.connect() as conn:
        logger.info("Начинаем очистку PostgreSQL...")
        try:
            # Используем `CASCADE` для удаления зависимых объектов
            await conn.execute(text('DROP TABLE IF EXISTS document_chunks CASCADE'))
            await conn.execute(text('DROP TABLE IF EXISTS documents CASCADE'))
            await conn.commit()
            logger.info("Таблицы 'document_chunks' и 'documents' успешно удалены.")
        except Exception as e:
            await conn.rollback()
            logger.error(f"Произошла ошибка при удалении таблиц в PostgreSQL: {e}")

async def clear_qdrant_collection():
    """
    Удаляет коллекцию векторов из Qdrant.
    """
    client = AsyncQdrantClient(url=QDRANT_URL)
    logger.info(f"Начинаем очистку Qdrant...")
    
    try:
        # Проверяем, существует ли коллекция
        collections_response = await client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        
        if COLLECTION_NAME in existing_collections:
            await client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Коллекция '{COLLECTION_NAME}' успешно удалена из Qdrant.")
        else:
            logger.info(f"Коллекция '{COLLECTION_NAME}' не найдена в Qdrant. Очистка не требуется.")
            
    except Exception as e:
        logger.error(f"Произошла ошибка при удалении коллекции в Qdrant: {e}")

async def main():
    logger.info("Запуск скрипта полной очистки баз данных.")
    
    # Запрос подтверждения от пользователя
    confirm = input("Вы уверены, что хотите полностью удалить все данные из PostgreSQL и Qdrant? (yes/no): ")
    if confirm.lower() != 'yes':
        logger.info("Операция отменена пользователем.")
        return
        
    await clear_postgresql_tables()
    await clear_qdrant_collection()
    logger.info("Очистка баз данных завершена.")

if __name__ == "__main__":
    asyncio.run(main())
