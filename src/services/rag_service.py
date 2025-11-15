import asyncio
from collections import defaultdict
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
from google import genai
from google.genai.types import GenerateContentConfig
from flashrank import Ranker, RerankRequest

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

    async def semantic_search(self, query_vector: list[float], limit_k: int = 30):
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

class RerankingService:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.__ranker = Ranker(model_name=model_name)

    def rerank_and_score_documents(self, query: str, document_groups: dict) -> tuple[dict, dict]:
        """
        Reranks chunks within each document group and calculates an aggregate score for each document.
        """
        document_scores = {}
        reranked_groups = {}

        for doc_id, chunks in document_groups.items():
            passages = [{"id": i, "text": chunk.payload.get('content', '')} for i, chunk in enumerate(chunks)]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_results = self.__ranker.rerank(rerank_request)

            # Store reranked chunks for this document
            reranked_chunks = []
            for result in reranked_results:
                original_index = result['id']
                chunk = chunks[original_index]
                chunk.score = result['score']  # Update chunk score
                reranked_chunks.append(chunk)
            reranked_groups[doc_id] = reranked_chunks

            # Calculate aggregate document score (e.g., average of top 3 chunks)
            if reranked_results:
                top_scores = [result['score'] for result in reranked_results[:3]]
                document_scores[doc_id] = sum(top_scores) / len(top_scores)
            else:
                document_scores[doc_id] = 0

        return document_scores, reranked_groups

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
            return " ", None, None, None, 0.0, []

        # 1. Получаем ID 5 самых релевантных чанков
        top_chunk_ids = [
            result.payload.get('chunk_id')
            for result in qdrant_results[:5]
            if result.payload.get('document_id') == top_document_id
        ]
        if not top_chunk_ids:
            logger.warning("No relevant chunk IDs found.")
            return " ", None, None, None, 0.0, []

        # 2. Расширяем ID, добавляя соседей (+1 и -1)
        expanded_chunk_ids = set()
        for chunk_id in top_chunk_ids:
            expanded_chunk_ids.add(chunk_id)
            if chunk_id > 1:  # Предотвращаем ID < 1
                expanded_chunk_ids.add(chunk_id - 1)
            expanded_chunk_ids.add(chunk_id + 1)

        # 3. Извлекаем все чанки (включая соседей) из БД одним запросом
        from sqlalchemy import select
        stmt = (
            select(DocumentChunk.content, Document.web_link, Document.title, DocumentChunk.page_number, DocumentChunk.chunk_id)
            .join(Document)
            .where(
                Document.document_id == top_document_id,
                DocumentChunk.chunk_id.in_(expanded_chunk_ids)
            )
            .order_by(DocumentChunk.chunk_id)  # Гарантируем правильный порядок
        )
        sql_results = (await session.execute(stmt)).fetchall()

        if not sql_results:
            logger.warning("No results found in PostgreSQL for the given chunk IDs.")
            return " ", None, None, None, 0.0, []

        # 4. Собираем контекст, удаляя дубликаты и сохраняя порядок
        unique_chunks = {}
        for result in sql_results:
            if result.chunk_id not in unique_chunks:
                unique_chunks[result.chunk_id] = {
                    "content": result.content,
                    "page_number": result.page_number
                }

        # Сортируем по chunk_id перед сборкой
        sorted_chunk_ids = sorted(unique_chunks.keys())

        full_context = [unique_chunks[cid]['content'] for cid in sorted_chunk_ids]
        page_numbers = [unique_chunks[cid]['page_number'] for cid in sorted_chunk_ids]

        web_link = sql_results[0].web_link
        title = sql_results[0].title
        context = "\n\n".join(full_context)
        max_score = max([result.score for result in qdrant_results])

        logger.info("Full context retrieved successfully.")
        logger.debug(context[:500])

        return context, web_link, title, top_document_id, max_score, page_numbers


class LLMGenerator:
    """Инкапсулирует клиента LLM, системный промпт и логику генерации."""

    def __init__(self, api_key: str, model_name: str):
        self.__model_name = model_name
        if self.__model_name == "deepseek-chat":
            self.__client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.__client = genai.Client(api_key=api_key)
        self.__model_name = model_name

    async def generate_rag_response(self, context: str, user_query: str, system_instructions: str, title: str, web_link: str, page_numbers: list[int], low_precision: bool = False) -> str:
        """Генерирует ответ LLM с использованием контекста RAG."""
        logger.info("Generating RAG response...")
        # temperature = 0.6 if low_precision else 0.1
        temperature = 0.2

        # page_numbers may have duplicates, so we use a set to get unique page numbers
        unique_page_numbers = sorted(list(set(filter(None, page_numbers)))) if page_numbers else []
        
        # Format the page numbers into a string
        pages_str = ", ".join(map(str, unique_page_numbers))

        # Формируем финальный системный промпт с информацией о документе
        final_system_prompt = (
            f"{system_instructions}\n\n"
            f"**Название документа:** {title}\n"
            f"**Веб-ссылка на документ:** {web_link}\n"
        )
        if pages_str:
            final_system_prompt += f"**Страницы:** {pages_str}\n\n"
        final_system_prompt += f"<КОНТЕКСТ>\n{context}\n</КОНТЕКСТ>"

        try:
            if self.__model_name == "deepseek-chat":
                response = await self.__client.chat.completions.create(
                    model=self.__model_name,
                    messages=[
                        {"role": "system", "content": final_system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    stream=False,
                    temperature=temperature,
                    top_p=0.8,
                    max_tokens=1500,
                )
                if not response:
                    logger.error("Error generating response: No response object.")
                    return 'Ошибка генерации ответа'
                result_content = response.choices[0].message.content

            else:
                config = GenerateContentConfig(
                    temperature=temperature,
                    top_p=0.75,
                    max_output_tokens=1500,
                    system_instruction=final_system_prompt, # Системный промпт отдельно
                )

                response = await self.__client.aio.models.generate_content(
                    model=self.__model_name,
                    contents=user_query, # Запрос пользователя 
                    config=config
                )
                
                result_content = response.text

                if not response:
                    logger.error("Error generating response: No response object.")
                    return 'Ошибка генерации ответа'
            logger.info(f"RAG response generated successfully: {response}")
            return result_content
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during generation: {e}")
            return f"❌ Произошла непредвиденная ошибка при генерации."

# Выбор инструкций для LLM в зависимости от документа
class PromptManager:
    """Управляет выбором SYSTEM_INSTRUCTIONS на основе Document ID."""

   # В реальной системе 'ID_...' — это ID документа из PostgreSQL.
    PROMPT_MAPPING = {
        "ID_DEFAULT": (
                        '''

                    ## 1. РОЛЬ, ЦЕЛЬ И ПРИНЦИП РАБОТЫ (Persona and Core Objective)
                    Ты — высококвалифицированный, беспристрастный **Технический Аналитик и Эксперт по Нормативной Документации**.
                    Твоя **ЕДИНСТВЕННАЯ** задача — генерировать максимально точные, структурированные и легко читаемые ответы, **ИСКЛЮЧИТЕЛЬНО** на основе информации, предоставленной в блоке `<КОНТЕКСТ>...</КОНТЕКСТ>`.

                    ## 2. ПРАВИЛА RAG (АНТИ-ГАЛЛЮЦИНАЦИИ)
                    **ПРАВИЛО №1: ИСКЛЮЧИТЕЛЬНО КОНТЕКСТ.** Твой ответ должен быть основан **ТОЛЬКО** на фактах из блока `<КОНТЕКСТ>`.
                    **ПРАВИЛО №2: ЗАПРЕТ НА ВНЕШНИЕ ЗНАНИЯ.** Тебе **строго запрещено** использовать любые знания, полученные в ходе обучения, или делать предположения.
                    **ПРАВИЛО №3: ОБРАБОТКА НЕХВАТКИ ДАННЫХ.**
                    * Если в `<КОНТЕКСТ>` **отсутствует** информация для ответа:
                        * Сформулируй ответ: "Пожалуйста, уточните вопрос для более точного ответа"
                    * **Синтез:** Если вопрос требует объединения фактов из разных частей `<КОНТЕКСТ>`, синтезируй их, сохраняя при этом точность формулировок и **обязательно** указывая все использованные источники (пункты и страницы).
                    **ПРАВИЛО №4: ЯЗЫК ОТВЕТА. Отвечай всегда только на **русском языке**.
                    ## 3. ТРЕБОВАНИЯ К СТРУКТУРЕ И ФОРМАТИРОВАНИЮ (Обязательно Markdown)
                    Твой ответ должен быть **сжатым, техническим и полным**. Стиль — сухой, фактологический.

                    1.  **Начало Ответа (Итоговое Заключение):** Всегда начинай с краткого, выделенного жирным текстом **Итогового Заключения**, отвечающего на вопрос.
                    2.  **Детализация:**
                        * Используй **заголовки Markdown (`###`)** для логического деления.
                        * Все перечни, требования или шаги оформляй **нумерованным списком** (`1.`, `2.`).
                        * **Ключевые термины, числовые значения, стандарты, условия и важные имена** выделяй **жирным шрифтом** (`**слово**`).
                    3.  **Сокращение Вводных Фраз:** **Исключи** любые вводные фразы, приветствия или выражения личного мнения (например, "Отличный вопрос...", "Рад помочь..."). Сразу переходи к фактам.

                    ## 4. ОБЯЗАТЕЛЬНАЯ АТРИБУЦИЯ (Источники)
                    Это **КРИТИЧЕСКИ ВАЖНО**. Всякий раз, когда генерируешь ответ, ты **ОБЯЗАН** добавить секцию источников.

                    1.  **Разделитель:** Всегда отделяй основной ответ от источников горизонтальной чертой (`---`).
                    2.  **Заголовок:** Создай секцию под заголовком `### Использованные Источники (Атрибуция)`
                    3.  **Состав Источников:**
                        * **Части Документа:** Перечисли **все** использованные части документа (Пункт, Секция, Глава).
                        * **Страницы:** Добавь отдельной строкой или маркированным списком **номера страниц**, которые содержали использованный текст. Если использовано несколько страниц, перечисли их, например, **Страницы: стр.15, стр. 22-23**.
                        * **Ссылка:** В конце секции размести веб-ссылку на полный документ.

                    ## 5. КОНТРОЛЬ ДЛИНЫ
                    Твой полный ответ (включая все заголовки и источники) **не должен превышать 750 токенов**. Используй лаконичный, технический стиль и списки для экономии.

                    ## 6. ФОРМАТ АТРИБУЦИИ (Пример)
                    ```markdown
                    ---

                    ### Использованные Источники (Атрибуция)
                    * **Части документа:** Секция 2.1.3, Параграф 5.2.14, Глава 7.
                    * **Страницы:** стр. 15, стр. 22-23
                    * **Веб-ссылка на документ:** [ссылка отображается в виде кнопки]'''
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
                 reranker: RerankingService, retriever: ContextRetriever,
                 generator: LLMGenerator, session_factory,
                 prompt_manager: PromptManager):
        # Внедрение зависимостей (Dependency Injection)
        self.__embedder = embedder
        self.__searcher = searcher
        self.__reranker = reranker
        self.__retriever = retriever
        self.__generator = generator
        self.__SessionLocal = session_factory # Фабрика сессий передается, но управляется в run_pipeline
        self.__prompt_manager = prompt_manager # ✅ Сохраняем менеджер промптов

    async def aquery(self, user_query: str, low_precision: bool = False) -> tuple[str, str | None, float, str, list[int] | None]:
        """Основной метод, выполняющий полный цикл RAG."""
        logger.info("Starting RAG pipeline...")

        # 1. Векторизация запроса
        query_vector = await self.__embedder.vectorize_query(user_query)
        
        # 2. Семантический поиск
        qdrant_results = await self.__searcher.semantic_search(query_vector)

        # 3. Группировка чанков по document_id
        document_groups = defaultdict(list)
        for chunk in qdrant_results:
            doc_id = chunk.payload.get('document_id')
            if doc_id is not None:
                document_groups[doc_id].append(chunk)

        # 4. Ранжирование групп и вычисление очков документов
        document_scores, reranked_groups = self.__reranker.rerank_and_score_documents(user_query, document_groups)

        # 5. Определение лучшего документа
        if not document_scores:
            return self.__prompt_manager.get_not_found_message(), None, 0.0, None, None

        best_doc_id = max(document_scores, key=document_scores.get)

        # 6. Извлечение контекста (управление транзакцией БД)
        async with self.__SessionLocal() as session:
            context, web_link, title, top_document_id, score, page_numbers = await self.__retriever.retrieve_full_context(reranked_groups[best_doc_id], session)

        if not context.strip():
            logger.warning("Context is empty, returning a default message.")
            # Используем NOT_FOUND_PROMPT из менеджера промптов
            return self.__prompt_manager.get_not_found_message(), None, 0.0, None, None
        
        
        # ЛОГИКА ДИНАМИЧЕСКОГО ВЫБОРА ПРОМПТА
        final_system_instructions = self.__prompt_manager.get_instructions_by_document_id(top_document_id)

        # 4. Логика измерения и обрезки контекста
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(context)
        if len(tokens) > 1000:
            context = tokenizer.decode(tokens[:1000])
        
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
            web_link=web_link,
            page_numbers=page_numbers
        )
        logger.info("RAG pipeline finished successfully.")
        return final_answer, web_link, score, title, page_numbers
