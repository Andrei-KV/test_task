from concurrent.futures import ProcessPoolExecutor
from opensearchpy import OpenSearch, helpers
from src.config import (
    OPENSEARCH_HOST, 
    OPENSEARCH_PORT, 
    OPENSEARCH_INDEX, 
    OPENSEARCH_USE_SSL, 
    OPENSEARCH_VERIFY_CERTS,
    EMBEDDING_MODEL_NAME
)
from uuid import uuid4
from src.app.logging_config import get_logger

logger = get_logger(__name__)

# Variables check
if (OPENSEARCH_HOST is None) or (OPENSEARCH_INDEX is None) or (EMBEDDING_MODEL_NAME is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")

from google import genai
from src.config import GEMINI_API_KEY, EMBEDDING_DIMENSION

import time
import random
from google.genai.errors import ClientError

class EmbeddingService:
    """Инкапсулирует клиент Gemini и логику векторизации."""
    
    def __init__(self, model_name: str):
        self.__client = genai.Client(api_key=GEMINI_API_KEY)
        self.__model_name = model_name
        self.vector_dimension = EMBEDDING_DIMENSION

    def vectorize_chunks(self, chunks: list[str]) -> list[list[float]]:
        """Векторизует список текстовых фрагментов, используя Gemini API с повторными попытками."""
        logger.info("Vectorizing user file with Gemini...")
        embeddings = []
        
        for chunk in chunks:
            max_retries = 5
            base_delay = 5
            
            for attempt in range(max_retries):
                try:
                    result = self.__client.models.embed_content(
                        model=self.__model_name,
                        contents=chunk,
                        config=genai.types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.vector_dimension
                        )
                    )
                    embeddings.append(result.embeddings[0].values)
                    break # Success, exit retry loop
                    
                except ClientError as e:
                    if e.code == 429:
                        if attempt < max_retries - 1:
                            delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                            logger.warning(f"⚠️ Quota exceeded (429). Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"❌ Max retries exceeded for 429 error: {e}")
                            raise e
                    else:
                        logger.error(f"❌ Gemini ClientError: {e}")
                        raise e
                except Exception as e:
                    # Check if it's a 429 wrapped in another exception
                    if "429" in str(e):
                         if attempt < max_retries - 1:
                            delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                            logger.warning(f"⚠️ Quota exceeded (429/Exception). Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                    
                    logger.error(f"❌ Error embedding chunk: {e}")
                    raise e
                
        logger.info("Text vectorized successfully.")
        return embeddings

class OpenSearchClientWrapper:
    """Инкапсулирует клиент OpenSearch и логику взаимодействия с индексом."""
    
    def __init__(self, host: str, port: int, index_name: str, use_ssl: bool, verify_certs: bool):
        self.__client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=None,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False,
            timeout=60  # Увеличен таймаут до 60 секунд для медленных операций
        )
        self.__index_name = index_name

    def ensure_index_exists(self, vector_dimension: int):
        """Создает индекс, если он не существует."""
        try:
            if not self.__client.indices.exists(index=self.__index_name):
                index_body = {
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 200
                        },
                        "analysis": {
                            "analyzer": {
                                "russian_analyzer": {
                                    "type": "standard",
                                    "stopwords": "_russian_"
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": vector_dimension,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {
                                        "ef_construction": 200,
                                        "m": 16
                                    }
                                }
                            },
                            "content": {
                                "type": "text",
                                "analyzer": "russian_analyzer"
                            },
                            "document_id": {"type": "integer"},
                            "chunk_id": {"type": "integer"},
                            "page_number": {"type": "integer"},
                            "type": {"type": "keyword"},
                            "sheet_name": {"type": "keyword"},
                            "qdrant_id": {"type": "keyword"}
                        }
                    }
                }
                self.__client.indices.create(index=self.__index_name, body=index_body)
                logger.info(f"✅ Index '{self.__index_name}' created with knn and text fields.")
            else:
                logger.info(f"✅ Index '{self.__index_name}' already exists.")
        except Exception as e:
            logger.error(f"❌ OpenSearch error: {e}. Make sure OpenSearch is running.")
            raise e # Fail fast if OpenSearch is down

    def bulk_index(self, actions: list):
        """Загружает (bulk index) документы в индекс."""
        try:
            success, failed = helpers.bulk(
                self.__client,
                actions,
                chunk_size=100,
                raise_on_error=False,
                refresh=True # Refresh to make documents immediately searchable
            )
            logger.info(f"✅ {success} documents successfully indexed to OpenSearch (Index: {self.__index_name}).")
            if failed:
                logger.warning(f"⚠️ Failed to index {len(failed)} documents.")
        except Exception as e:
            logger.error(f"❌ Error indexing to OpenSearch: {e}")

class DataMapper:
    """Предоставляет статические методы для преобразования данных, не требующие состояния."""
    
    @staticmethod
    def to_opensearch_actions(chunk_objects: list, embeddings: list[list[float]], index_name: str) -> list:
        """Статический метод, преобразующий фрагменты и векторы в действия для bulk API."""
        actions = []
        for idx, vector in enumerate(embeddings):
            chunk_obj = chunk_objects[idx]
            # Use qdrant_id as document _id for idempotency
            doc_id = chunk_obj.qdrant_id 
            
            action = {
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "embedding": vector,
                    "content": chunk_obj.content,
                    "document_id": chunk_obj.document_id,
                    "chunk_id": chunk_obj.chunk_id,
                    "page_number": chunk_obj.page_number,
                    "qdrant_id": chunk_obj.qdrant_id
                }
            }
            actions.append(action)
        return actions

class IndexingPipeline:
    """Оркестратор для полного процесса индексации (векторизация + загрузка)."""

    def __init__(self, embedding_service: EmbeddingService, os_wrapper: OpenSearchClientWrapper):
        self.__embedder = embedding_service
        self.__os_wrapper = os_wrapper
        
    def run(self, chunk_objects: list, batch_size: int = 100):
        """
        Выполняет полный цикл индексации для списка фрагментов с использованием батчинга.
        """
        self.__os_wrapper.ensure_index_exists(self.__embedder.vector_dimension)
        
        total_chunks = len(chunk_objects)
        logger.info(f"Starting indexing for {total_chunks} chunks...")

        for i in range(0, total_chunks, batch_size):
            batch_objects = chunk_objects[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size}...")
            
            chunks_text = [chunk.content for chunk in batch_objects]
            embeddings = self.__embedder.vectorize_chunks(chunks_text)
            actions = DataMapper.to_opensearch_actions(batch_objects, embeddings, OPENSEARCH_INDEX)
            
            self.__os_wrapper.bulk_index(actions)
        
        logger.info("✅ Индексация завершена.")

embedding_service_instance = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)
opensearch_wrapper_instance = OpenSearchClientWrapper(
    host=OPENSEARCH_HOST,
    port=OPENSEARCH_PORT,
    index_name=OPENSEARCH_INDEX,
    use_ssl=OPENSEARCH_USE_SSL,
    verify_certs=OPENSEARCH_VERIFY_CERTS
)
indexing_pipe_line = IndexingPipeline(
    embedding_service=embedding_service_instance,
    os_wrapper=opensearch_wrapper_instance
)
