import asyncio
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from qdrant_client import AsyncQdrantClient, models
from opensearchpy import AsyncOpenSearch
import logging

from src.database.database import AsyncSessionLocal
from src.database.models import Document, DocumentChunk
from src.config import (
    QDRANT_HOST, 
    COLLECTION_NAME,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    OPENSEARCH_USE_SSL,
    OPENSEARCH_VERIFY_CERTS
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def delete_document(document_id: int):
    """
    Полностью удаляет документ, включая связанные с ним чанки из PostgreSQL,
    соответствующие векторы из Qdrant и документы из OpenSearch.
    """
    qdrant_client = AsyncQdrantClient(url=QDRANT_HOST)
    
    os_client = AsyncOpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=None,
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False
    )
    
    async with AsyncSessionLocal() as session:
        # 1. Находим документ и связанные с ним чанки
        stmt = (
            select(Document)
            .where(Document.document_id == document_id)
            .options(selectinload(Document.chunks))
        )
        result = await session.execute(stmt)
        document_to_delete = result.scalar_one_or_none()

        if not document_to_delete:
            logger.warning(f"Документ с ID {document_id} не найден.")
            await os_client.close()
            return

        logger.info(f"Найден документ: '{document_to_delete.title}' (ID: {document_id}).")

        # 2. Собираем Qdrant IDs
        qdrant_ids_to_delete = [chunk.qdrant_id for chunk in document_to_delete.chunks]

        if not qdrant_ids_to_delete:
            logger.info("Для этого документа нет связанных векторов в Qdrant.")
        else:
            # 3. Удаляем векторы из Qdrant
            logger.info(f"Удаление {len(qdrant_ids_to_delete)} векторов из коллекции '{COLLECTION_NAME}'...")
            try:
                await qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=qdrant_ids_to_delete),
                )
                logger.info("Векторы успешно удалены из Qdrant.")
            except Exception as e:
                logger.error(f"Произошла ошибка при удалении векторов из Qdrant: {e}")
                # Прерываем операцию, чтобы избежать рассинхронизации
                await os_client.close()
                return
            
            # 4. Удаляем документы из OpenSearch
            logger.info(f"Удаление документов из OpenSearch index '{OPENSEARCH_INDEX}'...")
            try:
                # Используем delete_by_query для удаления по document_id
                query = {
                    "query": {
                        "term": {
                            "document_id": document_id
                        }
                    }
                }
                response = await os_client.delete_by_query(index=OPENSEARCH_INDEX, body=query)
                logger.info(f"Удалено {response.get('deleted', 0)} документов из OpenSearch.")
            except Exception as e:
                logger.error(f"Произошла ошибка при удалении из OpenSearch: {e}")
                # Не прерываем, так как Qdrant уже очищен, продолжаем очистку Postgres

        # 5. Удаляем документ из PostgreSQL (каскадное удаление чанков сработает автоматически)
        logger.info(f"Удаление документа (ID: {document_id}) из PostgreSQL...")
        try:
            await session.delete(document_to_delete)
            await session.commit()
            logger.info("Документ и связанные с ним чанки успешно удалены из PostgreSQL.")
        except Exception as e:
            await session.rollback()
            logger.error(f"Произошла ошибка при удалении документа из PostgreSQL: {e}")
        finally:
            await os_client.close()

