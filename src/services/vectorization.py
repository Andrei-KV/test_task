from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from config import QDRANT_HOST, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from uuid import uuid4
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Variables check
if (QDRANT_HOST is None) or (COLLECTION_NAME is None) or (EMBEDDING_MODEL_NAME is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")

#TEST Для исправления утечки RAM
# _embedding_model = None
# _qdrant_client = None


class EmbeddingService:
    """Инкапсулирует модель SentenceTransformer и логику векторизации."""
    
    def __init__(self, model_name: str):
        # 1. Загрузка тяжелого ресурса один раз при инициализации (инкапсуляция)
        self.__model = SentenceTransformer(model_name)
        # 2. Определение размерности векторов для Qdrant (зависит от модели)
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
        # Инкапсуляция клиента и конфигурации
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
            logger.info(f"❌ Qdrant error: {e}. Make sure Qdrant is running.")

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
            logger.info(f"❌ Error uploading to Qdrant: {e}")

class DataMapper:
    """Предоставляет статические методы для преобразования данных, не требующие состояния."""
    
    @staticmethod
    def to_qdrant_points(chunk_objects: list, embeddings: list[list[float]]) -> list:
        """Статический метод, преобразующий фрагменты и векторы в объекты PointStruct."""
        points = []
        for idx, vector in enumerate(embeddings):
            chunk_obj = chunk_objects[idx]
            #... вся логика преобразования ID и создания payload...
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
        # Принцип внедрения зависимостей (DI)
        self.__embedder = embedding_service
        self.__qdrant = qdrant_wrapper
        
    def run(self, chunk_objects: list):
        """Выполняет полный цикл индексации для списка фрагментов."""
        
        # 1. Обеспечение существования коллекции (используем инкапсулированные данные)
        self.__qdrant.ensure_collection_exists(self.__embedder.vector_dimension)
        
        # 2. Извлечение текста
        chunks_text = [chunk.content for chunk in chunk_objects]

        # 3. Трансформация (векторизация)
        embeddings = self.__embedder.vectorize_chunks(chunks_text)

        # 4. Преобразование в точки Qdrant (используем DataMapper)
        points = DataMapper.to_qdrant_points(chunk_objects, embeddings)

        # 5. Загрузка (Upsert)
        self.__qdrant.upsert_points(points)
        
        logger.info("✅ Индексация завершена.")

# --- Композиционный корень (Настройка и запуск) ---

# 1. Инициализация зависимости EmbeddingService
# Создаем экземпляр, передавая имя модели из конфигурации. 
# Тяжелая модель SentenceTransformer будет загружена здесь один раз.
embedding_service_instance = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)

# 2. Инициализация зависимости QdrantClientWrapper
# Создаем экземпляр, передавая хост и имя коллекции.
qdrant_wrapper_instance = QdrantClientWrapper(
    host=QDRANT_HOST, 
    collection_name=COLLECTION_NAME
)

# 3. Внедрение зависимостей (Dependency Injection)
# Создаем экземпляр оркестратора IndexingPipeline, передавая ему 
# готовые и настроенные сервисы.
indexing_pipe_line = IndexingPipeline(
    embedding_service=embedding_service_instance,
    qdrant_wrapper=qdrant_wrapper_instance
)

# TEST def get_embedding_model():
#     """
#     Returns a singleton instance of the embedding model.
#     Initializes the model on the first call.
#     """
#     global _embedding_model
#     if _embedding_model is None:
#         _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     return _embedding_model

# TEST def get_qdrant_client():
#     """
#     Returns a singleton instance of the Qdrant client.
#     Initializes the client on the first call.
#     """
#     global _qdrant_client
#     if _qdrant_client is None:
#         _qdrant_client = QdrantClient(url=QDRANT_HOST)
#     return _qdrant_client

# TEST def create_qdrant_collection(qdrant_client: QdrantClient, vector_dimension: int):
#     """Creates a new collection in Qdrant if it doesn't already exist."""
#     try:
#         if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
#             qdrant_client.create_collection(
#                 collection_name=COLLECTION_NAME,
#                 vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
#             )
#             print(f"✅ Collection '{COLLECTION_NAME}' created.")
#         else:
#             print(f"✅ Collection '{COLLECTION_NAME}' already exists.")
#     except Exception as e:
#         print(f"❌ Qdrant error: {e}. Make sure Qdrant is running.")

# def vectorize_chunks(embedding_model: SentenceTransformer, chunks: list[str]) -> list[list[float]]:
#     """Vectorizes a list of text chunks."""
#     embeddings = embedding_model.encode(
#         chunks,
#         show_progress_bar=True,
#         normalize_embeddings=True,
#         convert_to_tensor=False
#     ).tolist()
#     return embeddings

# def create_qdrant_points(chunk_objects: list[DocumentChunk], embeddings: list[list[float]]) -> list[PointStruct]:
#     """Creates a list of PointStruct objects for upserting into Qdrant."""
#     points = []
#     for idx, vector in enumerate(embeddings):
#         chunk_obj = chunk_objects[idx]
#         qdrant_id_int = int(chunk_obj.qdrant_id.replace('-', '')[:15], 16)
#         payload = {
#             "chunk_id": chunk_obj.chunk_id,
#             "document_id": chunk_obj.document_id,
#             "content_preview": chunk_obj.content[:100] + "..."
#         }
#         point = PointStruct(
#             id=qdrant_id_int,
#             vector=vector,
#             payload=payload
#         )
#         points.append(point)
#     return points

# def upsert_points_to_qdrant(qdrant_client: QdrantClient, points: list[PointStruct]):
#     """Upserts a list of points into a Qdrant collection."""
#     try:
#         qdrant_client.upsert(
#             collection_name=COLLECTION_NAME,
#             wait=True,
#             points=points
#         )
#         print(f"✅ {len(points)} vectors successfully uploaded to Qdrant (Collection: {COLLECTION_NAME}).")
#     except Exception as e:
#         print(f"❌ Error uploading to Qdrant: {e}")

