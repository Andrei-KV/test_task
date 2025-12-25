
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from opensearchpy import AsyncOpenSearch
import logging
import os

from src.config import (
    DB_URI, 
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    OPENSEARCH_INDEX,
    OPENSEARCH_USE_SSL,
    OPENSEARCH_VERIFY_CERTS,
    OPENSEARCH_PASSWORD
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Логика адаптации строк подключения для локального запуска ---

def get_postgres_url(original_url):
    """Заменяем хост контейнера на localhost и порт на внешний (7350)."""
    if os.getenv("RUNNING_IN_DOCKER") == "true":
        return original_url
        
    if "legal_rag_postgres" in original_url:
        return original_url.replace("legal_rag_postgres:5432", "localhost:7350")
    if "postgres" in original_url and "localhost" not in original_url:
         return original_url.replace("postgres:5432", "localhost:7350").replace("@postgres", "@localhost:7350")
    return original_url

def get_opensearch_host(original_host):
    """Заменяем хост контейнера на localhost."""
    if os.getenv("RUNNING_IN_DOCKER") == "true":
        return original_host
        
    if original_host == "opensearch":
        return "localhost"
    return original_host

# Применяем адаптацию
POSTGRES_DATABASE_URL = get_postgres_url(DB_URI)
LOCAL_OPENSEARCH_HOST = get_opensearch_host(OPENSEARCH_HOST)

# ------------------------------------------------------------------

async def clear_postgresql_tables():
    """
    Удаляет таблицы 'document_chunks' и 'documents' из базы данных PostgreSQL.
    """
    logger.info(f"Начинаем очистку PostgreSQL ({POSTGRES_DATABASE_URL.split('@')[-1]})...")
    engine = create_async_engine(POSTGRES_DATABASE_URL, echo=False)
    
    try:
        async with engine.begin() as conn:
            # Используем `CASCADE` для удаления зависимых объектов
            await conn.execute(text('DROP TABLE IF EXISTS document_chunks CASCADE'))
            await conn.execute(text('DROP TABLE IF EXISTS documents CASCADE'))
            logger.info("Таблицы 'document_chunks' и 'documents' успешно удалены.")
    except Exception as e:
        logger.error(f"Произошла ошибка при удалении таблиц в PostgreSQL: {e}")
    finally:
        await engine.dispose()

async def clear_opensearch_index():
    """
    Удаляет индекс из OpenSearch.
    """
    logger.info(f"Начинаем очистку OpenSearch ({LOCAL_OPENSEARCH_HOST}:{OPENSEARCH_PORT})...")
    
    http_auth = ("admin", OPENSEARCH_PASSWORD) if OPENSEARCH_PASSWORD else None
    
    client = AsyncOpenSearch(
        hosts=[{'host': LOCAL_OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=http_auth,
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False
    )
    
    try:
        exists = await client.indices.exists(index=OPENSEARCH_INDEX)
        if exists:
            await client.indices.delete(index=OPENSEARCH_INDEX)
            logger.info(f"Индекс '{OPENSEARCH_INDEX}' успешно удален из OpenSearch.")
        else:
            logger.info(f"Индекс '{OPENSEARCH_INDEX}' не найден в OpenSearch. Очистка не требуется.")
            
    except Exception as e:
        logger.error(f"Произошла ошибка при удалении индекса в OpenSearch: {e}")
    finally:
        await client.close()

async def main():
    logger.info("Запуск скрипта полной очистки баз данных.")
    
    # Запрос подтверждения от пользователя
    confirm = input("Вы уверены, что хотите полностью удалить все данные из PostgreSQL и OpenSearch? (yes/no): ")
    if confirm.lower() != 'yes':
        logger.info("Операция отменена пользователем.")
        return
        
    await clear_postgresql_tables()
    await clear_opensearch_index()
    logger.info("Очистка баз данных завершена.")

if __name__ == "__main__":
    asyncio.run(main())
