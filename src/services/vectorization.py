from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from src.config import QDRANT_HOST, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from uuid import uuid4
from src.app.logging_config import get_logger

logger = get_logger(__name__)

# Variables check
if (QDRANT_HOST is None) or (COLLECTION_NAME is None) or (EMBEDDING_MODEL_NAME is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")

class EmbeddingService:
    """Инкапсулирует модель SentenceTransformer и логику векторизации."""
    
    def __init__(self, model_name: str):
        self.__model = SentenceTransformer(model_name)
        self.vector_dimension = self.__model.get_sentence_embedding_dimension()

    def vectorize_chunks(self, chunks: list[str]) -> list[list[float]]:
        """Векторизует список текстовых фрагментов, используя инкапсулированную модель."""
        logger.info("Vectorizing user file...")
        embeddings = self.__model.encode(
            chunks,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=False
        ).tolist()
        logger.info("Text vectorized successfully.")
        return embeddings

class QdrantClientWrapper:
    """Инкапсулирует клиент Qdrant и логику взаимодействия с коллекцией."""
    
    def __init__(self, host: str, collection_name: str):
        self.__client = QdrantClient(url=host)
        self.__collection_name = collection_name

    def ensure_collection_exists(self, vector_dimension: int):
        """Создает коллекцию, если она не существует."""
        try:
            if not self.__client.collection_exists(collection_name=self.__collection_name):
                self.__client.create_collection(
                    collection_name=self.__collection_name,
                    vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE), 
                )
                logger.info(f"✅ Collection '{self.__collection_name}' created.")
            else:
                logger.info(f"✅ Collection '{self.__collection_name}' already exists.")
        except Exception as e:
            logger.error(f"❌ Qdrant error: {e}. Make sure Qdrant is running.")

    def upsert_points(self, points: list):
        """Загружает (upserts) точки в коллекцию."""
        try:
            self.__client.upsert(
                collection_name=self.__collection_name,
                wait=True,
                points=points
            )
            logger.info(f"✅ {len(points)} vectors successfully uploaded to Qdrant (Collection: {self.__collection_name}).")
        except Exception as e:
            logger.error(f"❌ Error uploading to Qdrant: {e}")

class DataMapper:
    """Предоставляет статические методы для преобразования данных, не требующие состояния."""
    
    @staticmethod
    def to_qdrant_points(chunk_objects: list, embeddings: list[list[float]]) -> list:
        """Статический метод, преобразующий фрагменты и векторы в объекты PointStruct."""
        points = []
        for idx, vector in enumerate(embeddings):
            chunk_obj = chunk_objects[idx]
            qdrant_id_int = int(chunk_obj.qdrant_id.replace('-', '')[:15], 16)
            payload = {
                 "chunk_id": chunk_obj.chunk_id,
                 "document_id": chunk_obj.document_id,
                 "content_preview": chunk_obj.content[:100] + "..."
            }
            point = PointStruct(
                id=qdrant_id_int,
                vector=vector,
                payload=payload
            )
            points.append(point)
        return points

class IndexingPipeline:
    """Оркестратор для полного процесса индексации (векторизация + загрузка)."""

    def __init__(self, embedding_service: EmbeddingService, qdrant_wrapper: QdrantClientWrapper):
        self.__embedder = embedding_service
        self.__qdrant = qdrant_wrapper
        
    def run(self, chunk_objects: list, batch_size: int = 100):
        """
        Выполняет полный цикл индексации для списка фрагментов с использованием батчинга.
        """
        self.__qdrant.ensure_collection_exists(self.__embedder.vector_dimension)
        
        total_chunks = len(chunk_objects)
        logger.info(f"Starting indexing for {total_chunks} chunks...")

        for i in range(0, total_chunks, batch_size):
            batch_objects = chunk_objects[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size}...")
            
            chunks_text = [chunk.content for chunk in batch_objects]
            embeddings = self.__embedder.vectorize_chunks(chunks_text)
            points = DataMapper.to_qdrant_points(batch_objects, embeddings)
            
            self.__qdrant.upsert_points(points)
        
        logger.info("✅ Индексация завершена.")

embedding_service_instance = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)
qdrant_wrapper_instance = QdrantClientWrapper(
    host=QDRANT_HOST, 
    collection_name=COLLECTION_NAME
)
indexing_pipe_line = IndexingPipeline(
    embedding_service=embedding_service_instance,
    qdrant_wrapper=qdrant_wrapper_instance
)
