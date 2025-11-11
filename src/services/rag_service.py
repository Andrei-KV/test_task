import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import SearchParams
import tiktoken
from openai import AsyncOpenAI
from ..database.database import AsyncSessionLocal
from ..database.models import Document, DocumentChunk
from ..config import COLLECTION_NAME, LLM_MODEL, DEEPSEEK_API_KEY, QDRANT_HOST, EMBEDDING_MODEL_NAME
from src.app.logging_config import get_logger

logger = get_logger(__name__)


# Variables check
if (LLM_MODEL is None) or (COLLECTION_NAME is None) or (DEEPSEEK_API_KEY is None) or (QDRANT_HOST is None) or (EMBEDDING_MODEL_NAME is None):
    raise ValueError("Переменные не найдены. Проверьте файл.env.")


# =====================================================================
# Сервис векторизации запроса
class QueryEmbeddingService:
    def __init__(self, model_name: str):
        # Загрузка тяжелого ресурса (SentenceTransformer) один раз
        self.__model = SentenceTransformer(model_name)

    async def vectorize_query(self, query: str) -> list[float]:
        """Векторизует один текстовый запрос для поиска."""
        logger.info("Vectorizing user query...")
        query_embedding = await asyncio.to_thread(
            self.__model.encode,
            [query],
            normalize_embeddings=True,
            convert_to_tensor=False
        )
        logger.info("User query vectorized successfully.")
        return query_embedding.tolist()[0]
    

# Сервис семантического поиска
class QueryQdrantClient:
    def __init__(self, host: str, collection_name: str):
        # Инкапсуляция клиента и конфигурации
        self.__client = AsyncQdrantClient(url=host)
        self.__collection_name = collection_name

    async def semantic_search(self, query_vector: list[float], limit_k: int = 10):
        """Выполняет семантический поиск в Qdrant."""
        logger.info("Performing semantic search in Qdrant...")
        search_result = await self.__client.query_points(
            collection_name=self.__collection_name,
            query=query_vector,
            limit=limit_k,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(
                exact=False,
                hnsw_ef=100
            )
        )
        logger.info("Semantic search completed.")
        return search_result.points

# Извлечение контекста из БД PostgreSQL
class ContextRetriever:
    """Инкапсулирует логику извлечения полного контекста из PostgreSQL."""

    async def retrieve_full_context(self, qdrant_results, session: AsyncSession) -> tuple:
        """Извлекает полный текстовый контекст по результатам Qdrant."""
        logger.info("Retrieving full context from PostgreSQL...")
    
        try:
            top_document_id = qdrant_results[0].payload.get('document_id')
        except (IndexError, KeyError):
            logger.warning("No document ID found in Qdrant results.")
            return " ", None, None, None, 0.0

        relevant_chunk_ids = [
            result.payload.get('chunk_id')
            for result in qdrant_results
            if result.payload.get('document_id') == top_document_id
        ]
        if not relevant_chunk_ids:
            logger.warning("No relevant chunk IDs found.")
            return " ", None, None, None, 0.0

        from sqlalchemy import select
        stmt = (
            select(DocumentChunk.content, Document.web_link, Document.title)
           .join(Document)
           .where(DocumentChunk.chunk_id.in_(relevant_chunk_ids))
           .order_by(DocumentChunk.chunk_id)
        )
        sql_results = (await session.execute(stmt)).fetchall()

        if not sql_results:
            logger.warning("No results found in PostgreSQL for the given chunk IDs.")
            return " ", None, None, None, 0.0

        full_context = [result.content for result in sql_results]
        web_link = sql_results[0].web_link
        title = sql_results[0].title
        context = "\n\n".join(full_context)
        max_score = max([result.score for result in qdrant_results])
        logger.info("Full context retrieved successfully.")
        logger.debug(context[:500])
        return context, web_link, title, top_document_id, max_score


class LLMGenerator:
    """Инкапсулирует клиента LLM, системный промпт и логику генерации."""

    def __init__(self, api_key: str, model_name: str):
        self.__client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.__model_name = model_name

    async def generate_rag_response(self, context: str, user_query: str, system_instructions: str, title: str, web_link: str, low_precision: bool = False) -> str:
        """Генерирует ответ LLM с использованием контекста RAG."""
        logger.info("Generating RAG response...")
        # temperature = 0.6 if low_precision else 0.1
        temperature = 0

        # Формируем финальный системный промпт с информацией о документе
        final_system_prompt = (
            f"{system_instructions.format(web_link=web_link)}\n\n"
            f"**Название документа:** {title}\n"
            f"**Веб-ссылка на документ:** {web_link}\n\n"
            f"<КОНТЕКСТ>\n{context}\n</КОНТЕКСТ>"
        )

        try:
            response = await self.__client.chat.completions.create(
                model=self.__model_name,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": user_query},
                ],
                stream=False,
                temperature=temperature,
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
            "***\n"
            "РОЛЬ И ОСНОВНОЕ НАЗНАЧЕНИЕ:\n"
            "Ты — высококвалифицированный **Технический Аналитик и Эксперт по Нормативной Документации**.\n"
            "Твоя единственная задача — **анализировать предоставленный КОНТЕКСТ** и генерировать максимально точные, структурированные и легко читаемые ответы."
            "Ответ должен быть кратким. Главное -- это передать ссылку на документ и на пункты документа\n\n"
            "### ПРАВИЛА RAG (АНТИ-ГАЛЛЮЦИНАЦИИ):\n"
            "1.  **ИСКЛЮЧИТЕЛЬНО КОНТЕКСТ:** Твой ответ должен быть основан **ТОЛЬКО** на информации, содержащейся в блоке `<КОНТЕКСТ>...</КОНТЕКСТ>`.\n"
            "2.  **Запрет на Внешние Знания:** Тебе **строго запрещено** использовать любые знания, полученные в ходе обучения, или делать предположения.\n"
            "3.  **Обработка Недостатка Данных:** Если в `<КОНТЕКСТ>` отсутствует информация, необходимая для ответа, попроси уточнить вопрос. \n"
            "    Если после двух уточняющих вопросов всё ещё отсутствует информация, необходимая для ответа, то предоставь ответ пониженной точности, предупредив об этом, и попроси переформулировать вопрос.\n"
            "4.  **Синтез:** Если вопрос требует объединения фактов из разных частей `<КОНТЕКСТ>`, синтезируй их, сохраняя при этом точность формулировок.\n\n"
            "### ТРЕБОВАНИЯ К СТРУКТУРЕ И ФОРМАТИРОВАНИЮ (Markdown):\n"
            "1.  **Лаконичность и Полнота:** Ответ должен быть **сжатым, но полным**. Избегай излишней детализации, но убедись, что ответ на вопрос дан исчерпывающе и не обрывается на полуслове.\n"
            "2.  **Общая Структура:** Ответ должен быть разбит на логические секции с использованием заголовков Markdown.\n"
            "3.  **Начало Ответа:** Всегда начинай с **краткого итогового заключения (Summary)**, выделяя главную мысль.\n"
            "4.  **Детализация:**\n"
            "    * **Перечни и Условия:** Любые списки требований, шагов или перечней должны быть оформлены как **нумерованный список** (`1.`, `2.`, `3.`) или маркированный список (`*`).\n"
            "    * **Акценты:** **Ключевые термины, числовые значения, стандарты и важные условия** должны быть выделены **жирным шрифтом** (`**слово**`).\n"
            "5.  **Атрибуция (Источники):** В самом конце ответа, после горизонтальной черты (`---`), создай секцию под заголовком `### Использованные Источники`.\n"
            "    * Сначала перечисли все **части документа** (`Пункт 1`, `Глава 2` и т.д.), которые были использованы для формирования ответа.\n"
            "    * В конце этой секции добавь **веб-ссылку** на полный документ.\n\n"
            "6. Сокращение Вводных Фраз: Исключитm фразы типа 'Отличный вопрос. Это фундаментальное понятие...'. Система-промпт  'Эксперт', подразумевает, что модель сразу переходит к фактам."
            "### КОНТРОЛЬ ДЛИНЫ И ПОЛНОТЫ ОТВЕТА:"
                "1.  **Приоритет завершения:** Ответ должен быть **полностью завершен** и логически окончен. При приближении к лимиту токенов, модель должна **уплотнять информацию** и завершать текущую мысль."
                "2.  **Экономия токенов:** Избегай излишних вводных фраз, приветствий и развернутых предложений. Используй краткий, технический стиль, максимально используя **списки и выделение** для экономии токенов."
                "3.  **Максимальная длина:** Твой полный ответ (включая все заголовки и источники) **не должен превышать 800 токенов**, чтобы гарантировать завершение."
            "### ПРИМЕР ЖЕЛАЕМОГО ФОРМАТА:\n"
            "```markdown\n"
            "### Основные Требования к Экранированию Кабелей\n"
            "Изоляция кабеля должна выдерживать рабочее напряжение, и для кабелей выше **3 кВ** обязательно требуется заземленный экран.\n\n"
            "---\n\n"
            "### Детали Требований:\n"
            "1.  **Обязательность:** Экранирование обязательно для всех кабелей с номинальным напряжением **выше 3 кВ**.\n"
            "2.  **Материал:** Экран должен быть выполнен из **медного** или **алюминиевого** проводника.\n\n"
            "---\n\n"
            "### Использованные Источники:\n"
            "* **Части документа:** Секция 2.1.3, Параграф 5.2.14, Глава 7, стр. 15.\n"
            "* **Веб-ссылка:** [Название документа]({web_link})\n"
            "```"
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

class RAGService:
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

    async def aquery(self, user_query: str, low_precision: bool = False) -> tuple[str, str | None, float]:
        """Основной метод, выполняющий полный цикл RAG."""
        logger.info("Starting RAG pipeline...")

        # 1. Векторизация запроса
        query_vector = await self.__embedder.vectorize_query(user_query)
        
        # 2. Семантический поиск
        qdrant_results = await self.__searcher.semantic_search(query_vector)

        # 3. Извлечение контекста (управление транзакцией БД)
        async with self.__SessionLocal() as session:
            context, web_link, title, top_document_id, score = await self.__retriever.retrieve_full_context(qdrant_results, session)

        if not context.strip():
            logger.warning("Context is empty, returning a default message.")
            # Используем NOT_FOUND_PROMPT из менеджера промптов
            return self.__prompt_manager.get_not_found_message(), None, 0.0
        
        # ЛОГИКА ДИНАМИЧЕСКОГО ВЫБОРА ПРОМПТА
        final_system_instructions = self.__prompt_manager.get_instructions_by_document_id(top_document_id)

        # 4. Логика измерения и обрезки контекста
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(context)
        if len(tokens) > 400:
            context = tokenizer.decode(tokens[:400])
        
        size_bytes = len(context.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"💾 Контекст из базы (после обработки): {size_bytes} байт, {len(tokens)} токенов, что составляет {size_mb:.4f} МБ.")
        
        # 5. Генерация ответа
        final_answer = await self.__generator.generate_rag_response(
            context=context,
            user_query=user_query,
            system_instructions=final_system_instructions,
            low_precision=low_precision,
            title=title,
            web_link=web_link
        )
        logger.info("RAG pipeline finished successfully.")
        return final_answer, web_link, score
