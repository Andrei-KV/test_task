import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer, CrossEncoder
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
    

# Semantic search in Qdrant
class QueryQdrantClient:
    def __init__(self, host: str, collection_name: str):

        self.__client = AsyncQdrantClient(url=host)
        self.__collection_name = collection_name

    async def semantic_search(self, query_vector: list[float], limit_k: int = 30):
        """Выполняет семантический поиск в Qdrant.
        Извлекааает большое количество векторов для уточнения и фильтрации на уровне контекста."""
        logger.info("Performing semantic search in Qdrant...")
        search_result = await self.__client.query_points(
            collection_name=self.__collection_name,
            query=query_vector,
            limit=limit_k,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(
                exact=False,
                hnsw_ef=200
            ),
            score_threshold=0.7
        )
        candidates = search_result.points

        logger.info(f"Found {len(candidates)} candidates. Starting reranking...")

        if not candidates:
            logger.warning("No candidates found in semantic search.")
            return [], None, 0
        
        # 2: Giving document_id priority to the most relevant candidate
        first_candidate_payload = candidates[0].payload
        if not first_candidate_payload or 'document_id' not in first_candidate_payload:
            logger.error("The most relevant candidate is missing 'document_id' in its payload.")
            return [], None, 0
        
        max_score = candidates[0].score

        target_document_id = first_candidate_payload['document_id']
        logger.info(f"Most relevant document_id: {target_document_id}")

        # 3: Select the top-4 relevant chunks from the same document_id
        filtered_candidates = [
            p for p in candidates 
            if p.payload and p.payload.get('document_id') == target_document_id
        ]

        top_relevant_chunks = filtered_candidates[:4]
        
        if not top_relevant_chunks:
            logger.warning(f"No relevant chunks found for document_id: {target_document_id}")
            return [], None, 0
            
        logger.info(f"Selected {len(top_relevant_chunks)} top relevant chunks from target document.")

        # 4: Collect chunk_ids of the top relevant chunks
        target_chunk_ids = {p.payload['chunk_id'] for p in top_relevant_chunks if p.payload and 'chunk_id' in p.payload}
        
        if not target_chunk_ids:
             logger.error("Selected chunks are missing 'chunk_id'.")
             return [], None, 0

        # 5: Request neighboring chunks (+1 and -1)
        neighbor_chunk_ids = set()
        for chunk_id in target_chunk_ids:
            if chunk_id > 0:
                neighbor_chunk_ids.add(chunk_id - 1)
            neighbor_chunk_ids.add(chunk_id + 1)

        # Collect all required chunk IDs
        all_required_chunk_ids = target_chunk_ids.union(neighbor_chunk_ids)
        logger.info(f"Total unique chunk_ids to retrieve (including neighbors): {len(all_required_chunk_ids)}")
        
        final_sorted_chunk_ids = sorted(list(all_required_chunk_ids))
        
        logger.info(f"Final list of {len(final_sorted_chunk_ids)} unique chunk_ids is ready: {final_sorted_chunk_ids}")
        
        return final_sorted_chunk_ids, target_document_id, max_score


# Extact full context from PostgreSQL
class ContextRetriever:
    """Инкапсулирует логику извлечения полного контекста из PostgreSQL."""

    async def retrieve_full_context(self, qdrant_results, top_document_id, session: AsyncSession) -> tuple:
        """Извлекает полный текстовый контекст по результатам Qdrant."""
        logger.info("Retrieving full context from PostgreSQL...")
    
        if not qdrant_results or not top_document_id:
            logger.warning("No document ID found in Qdrant results.")
            return " ", None, None, None, []

              
        # Retrieve all chunks from database based on chunk IDs from Qdrant
        from sqlalchemy import select
        stmt = (
            select(DocumentChunk.content, Document.web_link, Document.title, DocumentChunk.page_number, DocumentChunk.chunk_id)
            .join(Document)
            .where(
                Document.document_id == top_document_id,
                DocumentChunk.chunk_id.in_(qdrant_results)
            )
            .order_by(DocumentChunk.chunk_id)  
        )
        sql_results = (await session.execute(stmt)).fetchall()

        if not sql_results:
            logger.warning("No results found in PostgreSQL for the given chunk IDs.")
            return " ", None, None, None, []

        # Collect unique chunks while preserving order
        unique_chunks = []
        for result in sql_results:
            unique_chunks.append({
                "content": result.content,
                "page_number": result.page_number
            })
        
        full_context = [unique_chunks[cid]['content'] for cid in qdrant_results]
        page_numbers = [unique_chunks[cid]['page_number'] for cid in qdrant_results]
        
        web_link = sql_results[0].web_link
        title = sql_results[0].title
        context = "\n".join(full_context)
        
        
        logger.info("Full context retrieved successfully.")
        logger.debug(f'Full context retrieved: {context}')
        
        return context, web_link, title, top_document_id, page_numbers


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
                    max_output_tokens=2000,
                    system_instruction=final_system_prompt, 
                )

                response = await self.__client.aio.models.generate_content(
                    model=self.__model_name,
                    contents=user_query,  
                    config=config
                )
                
                
                if not response:
                    logger.error("Error generating response: No response object.")
                    return 'Ошибка генерации ответа'
                
                result_content = response.text
            
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
                 retriever: ContextRetriever, generator: LLMGenerator, session_factory,
                 prompt_manager: PromptManager):
        # Внедрение зависимостей (Dependency Injection)
        self.__embedder = embedder
        self.__searcher = searcher
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
        qdrant_results, top_document_id, max_score = await self.__searcher.semantic_search(query_vector)

        # 3. Извлечение контекста (управление транзакцией БД)
        async with self.__SessionLocal() as session:
            context, web_link, title, top_document_id, page_numbers = await self.__retriever.retrieve_full_context(qdrant_results, top_document_id, session)
            score = max_score if max_score else 0.0
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
