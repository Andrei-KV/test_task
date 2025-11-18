import asyncio
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from qdrant_client import AsyncQdrantClient, models
import logging

from src.database.database import AsyncSessionLocal
from src.database.models import Document, DocumentChunk
from src.config import QDRANT_HOST, COLLECTION_NAME

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def delete_document(document_id: int):
    """
    Полностью удаляет документ, включая связанные с ним чанки из PostgreSQL
    и соответствующие векторы из Qdrant.
    """
    qdrant_client = AsyncQdrantClient(url=QDRANT_HOST)
    
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
                return

        # 4. Удаляем документ из PostgreSQL (каскадное удаление чанков сработает автоматически)
        logger.info(f"Удаление документа (ID: {document_id}) из PostgreSQL...")
        try:
            await session.delete(document_to_delete)
            await session.commit()
            logger.info("Документ и связанные с ним чанки успешно удалены из PostgreSQL.")
        except Exception as e:
            await session.rollback()
            logger.error(f"Произошла ошибка при удалении документа из PostgreSQL: {e}")

