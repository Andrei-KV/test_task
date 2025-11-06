from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from openai import OpenAI
from database.database import SessionLocal
from database.models import Document, DocumentChunk
from services.vectorization import EmbeddingService, QdrantClientWrapper
from config import COLLECTION_NAME, LLM_MODEL, DEEPSEEK_API_KEY, QDRANT_HOST, EMBEDDING_MODEL_NAME
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


# Variables check
if (LLM_MODEL is None) or (COLLECTION_NAME is None) or (DEEPSEEK_API_KEY is None) or (QDRANT_HOST is None) or (EMBEDDING_MODEL_NAME is None):
    raise ValueError("Переменные не найдены. Проверьте файл.env.")


# =====================================================================
# Сервис векторизации запроса
class QueryEmbeddingService:
    def __init__(self, model_name: str):
        # Загрузка тяжелого ресурса (SentenceTransformer) один раз
        self.__model = SentenceTransformer(model_name)

    def vectorize_query(self, query: str) -> list[float]:
        """Векторизует один текстовый запрос для поиска."""
        logger.info("Vectorizing user query...")
        query_embedding = self.__model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_tensor=False
        ).tolist()[0]
        logger.info("User query vectorized successfully.")
        return query_embedding
    

# Сервис семантического поиска
class QueryQdrantClient:
    def __init__(self, host: str, collection_name: str):
        # Инкапсуляция клиента и конфигурации
        self.__client = QdrantClient(url=host)
        self.__collection_name = collection_name

    def semantic_search(self, query_vector: list[float], limit_k: int = 10):
        """Выполняет семантический поиск в Qdrant."""
        logger.info("Performing semantic search in Qdrant...")
        search_result = self.__client.query_points(
            collection_name=self.__collection_name,
            query=query_vector,
            limit=limit_k,
            with_payload=True,
            search_params=SearchParams(
                exact=False,
                hnsw_ef=100
            )
        ).points
        logger.info("Semantic search completed.")
        return search_result

# Извлечение контекста из БД PostgreSQL
class ContextRetriever:
    """Инкапсулирует логику извлечения полного контекста из PostgreSQL."""

    def retrieve_full_context(self, qdrant_results, session: Session) -> tuple:
        """Извлекает полный текстовый контекст по результатам Qdrant."""
        logger.info("Retrieving full context from PostgreSQL...")
    
        try:
            top_document_id = qdrant_results[0].payload.get('document_id')
        except (IndexError, KeyError):
            logger.warning("No document ID found in Qdrant results.")
            return " ", None, None

        relevant_chunk_ids = [
            result.payload.get('chunk_id')
            for result in qdrant_results
            if result.payload.get('document_id') == top_document_id
        ]
        if not relevant_chunk_ids:
            logger.warning("No relevant chunk IDs found.")
            return " ", None

        from sqlalchemy import select
        stmt = (
            select(DocumentChunk.content, Document.web_link)
           .join(Document)
           .where(DocumentChunk.chunk_id.in_(relevant_chunk_ids))
           .order_by(DocumentChunk.chunk_id)
        )
        sql_results = session.execute(stmt).fetchall()

        if not sql_results:
            logger.warning("No results found in PostgreSQL for the given chunk IDs.")
            return " ", None

        full_context = [result.content for result in sql_results]
        web_link = sql_results[0].web_link
        context = "\n\n".join(full_context)
        logger.info("Full context retrieved successfully.")
        logger.debug(context[:500])
        return context, web_link, top_document_id


class LLMGenerator:
    """Инкапсулирует клиента LLM, системный промпт и логику генерации."""

    def __init__(self, api_key: str, model_name: str):
        # Клиент OpenAI инкапсулирован, API ключ не является глобальной переменной
        self.__client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.__model_name = model_name

    def generate_rag_response(self, context: str, user_query: str, system_instructions: str) -> str:
        """Генерирует ответ LLM с использованием контекста RAG."""
        logger.info("Generating RAG response...")
        try:
            response = self.__client.chat.completions.create(
                model=self.__model_name,
                messages=[
                {"role": "system", "content": f"{system_instructions}--- **КОНТЕКСТ** ---{context}"},
                {"role": "user", "content": user_query},
            ],
                stream=False,
                temperature=0.1,
                top_p=0.8,
                max_tokens=1000,
            )
            if not response:
                logger.error("Error generating response: No response object.")
                return 'Ошибка генерации ответа'
            logger.info(f"RAG response generated successfully: {response}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"An unexpected error occurred during generation: {e}")
            return f"❌ Произошла непредвиденная ошибка при генерации."

# Выбор инструкций для LLM в зависимости от документа
class PromptManager:
    """Управляет выбором SYSTEM_INSTRUCTIONS на основе Document ID."""

    # В реальной системе 'ID_...' — это ID документа из PostgreSQL.
    PROMPT_MAPPING = {
        "ID_DEFAULT": (
            "Ты — специализированный AI-ассистент, работающий исключительно на основе предоставленных "
            "официальных документов техническо направленности"
            "**ТВОИ ОСНОВНЫЕ ПРИНЦИПЫ:**"
            "1.**Точность и аутентичность:** Все твои ответы должны дословно или почти дословно основываться "
            "на предоставленном тексте. "
            "Нельзя привносить собственную интерпретацию, домыслы или знания извне."
            "2.**Ссылка на источник:** Всегда указывай точный источник информации. "
            "Формат: 'Согласно [название документа], пункт [номер пункта]...'."
            "3.**Структурированность:** Если вопрос сложный, структурируй ответ, используя списки или четкие абзацы."
            "4.**Недопустимость вымысла:** Если в предоставленном тебе фрагменте нет информации для ответа на вопрос,"
            " категорически запрещено придумывать ответ. "
            "Ты должен четко заявить попросить задать уточняющий вопрос. Если после двух уточняющих вопросов всё ещё "
            "не хватает информации для ответа на вопрос,"
            " то ты должен чётко сообщить об этом"
            " Сохраняйте профессиональный тон и отвечайте на русском языке."
        ),
        # Добавить другие ID документов и соответствующие инструкции
    }
    
    # Промпт для случая, когда контекст не найден
    NOT_FOUND_PROMPT = "Извините, в предоставленных документах точный ответ не найден. Уточните вопрос"


    def get_instructions_by_document_id(self, document_id: str) -> str:
        """Возвращает системные инструкции для заданного ID документа."""
        
        # Используем.get() для безопасного извлечения. Если ID не найден, 
        # возвращаем дефолтный промпт.
        return self.PROMPT_MAPPING.get(document_id, 'ID_DEFAULT')

    def get_not_found_message(self):
         return self.NOT_FOUND_PROMPT
    
# =====================================================================
# ОРКЕСТРАТОР
# =====================================================================

class RAGPipelineOrchestrator:
    """Центральный класс, управляющий последовательностью операций RAG."""

    def __init__(self, embedder: QueryEmbeddingService, searcher: QueryQdrantClient, 
                 retriever: ContextRetriever, generator: LLMGenerator, session_factory,
                 prompt_manager: PromptManager):
        # Внедрение зависимостей (Dependency Injection)
        self.__embedder = embedder
        self.__searcher = searcher
        self.__retriever = retriever
        self.__generator = generator
        self.__SessionLocal = session_factory # Фабрика сессий передается, но управляется в run_pipeline
        self.__prompt_manager = prompt_manager # ✅ Сохраняем менеджер промптов

    def run_pipeline(self, user_query: str) -> tuple[str, str | None]:
        """Основной метод, выполняющий полный цикл RAG."""
        logger.info("Starting RAG pipeline...")

        # 1. Векторизация запроса
        query_vector = self.__embedder.vectorize_query(user_query)
        
        # 2. Семантический поиск
        qdrant_results = self.__searcher.semantic_search(query_vector)

        # 3. Извлечение контекста (управление транзакцией БД)
        with self.__SessionLocal() as session:
            context, web_link, top_document_id = self.__retriever.retrieve_full_context(qdrant_results, session)

        if not context.strip():
            logger.warning("Context is empty, returning a default message.")
            # Используем NOT_FOUND_PROMPT из менеджера промптов
            return self.__prompt_manager.get_not_found_message(), None
        
        # ЛОГИКА ДИНАМИЧЕСКОГО ВЫБОРА ПРОМПТА
        final_system_instructions = self.__prompt_manager.get_instructions_by_document_id(top_document_id)

        # 4. Логика измерения контекста
        size_bytes = len(context.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"💾 Контекст из базы: {size_bytes} байт, что составляет {size_mb:.4f} МБ.")
        
        # 5. Генерация ответа
        final_answer = self.__generator.generate_rag_response(
            context=context, 
            user_query=user_query,
            system_instructions=final_system_instructions # Передаем выбранные инструкции
        )
        logger.info("RAG pipeline finished successfully.")
        return final_answer, web_link


# =====================================================================
# КОМПОЗИЦИОННЫЙ КОРЕНЬ (Точка входа)
# =====================================================================

# 1. Инициализация всех зависимостей, используя конфигурацию
QUERY_EMBEDDER = QueryEmbeddingService(model_name=EMBEDDING_MODEL_NAME)
QUERY_SEARCHER = QueryQdrantClient(host=QDRANT_HOST, collection_name=COLLECTION_NAME)
CONTEXT_RETRIEVER = ContextRetriever()
LLM_GENERATOR = LLMGenerator(api_key=DEEPSEEK_API_KEY, model_name=LLM_MODEL)
PROMPT_MANAGER = PromptManager()

# 2. Создание главного оркестратора (Внедрение зависимостей)
RAG_ORCHESTRATOR = RAGPipelineOrchestrator(
    embedder=QUERY_EMBEDDER,
    searcher=QUERY_SEARCHER,
    retriever=CONTEXT_RETRIEVER,
    generator=LLM_GENERATOR,
    session_factory=SessionLocal,
    prompt_manager=PROMPT_MANAGER
)

# 3. Публичная функция для использования в других модулях (например, в обработчике Telegram бота)
def run_rag_pipeline(user_query: str) -> tuple[str, str | None]:
    """Единственная публичная функция для запуска всего RAG-пайплайна."""
    return RAG_ORCHESTRATOR.run_pipeline(user_query)
