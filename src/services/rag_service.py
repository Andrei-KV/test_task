import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import tiktoken
from src.database.models import Document, DocumentChunk
from src.config import LLM_MODEL, DEEPSEEK_API_KEY, COLLECTION_NAME, QDRANT_HOST
from src.app.logging_config import get_logger
from google.genai.types import GenerateContentConfig
from .vectorization import QueryEmbeddingService, QueryQdrantClient

logger = get_logger(__name__)

# Проверка переменных
if not all([LLM_MODEL, COLLECTION_NAME, DEEPSEEK_API_KEY, QDRANT_HOST]):
    raise ValueError("Одна или несколько переменных окружения не найдены. Проверьте .env.")

class ContextRetriever:
    async def retrieve_full_context(self, qdrant_results, session: AsyncSession) -> tuple:
        logger.info("Извлечение полного контекста из PostgreSQL...")
        try:
            top_document_id = qdrant_results[0].payload.get('document_id')
            if not top_document_id:
                logger.warning("Не найден document_id в результатах Qdrant.")
                return " ", None, None, None, 0.0, [], []
        except (IndexError, KeyError):
            logger.warning("Не удалось извлечь document_id из результатов Qdrant.")
            return " ", None, None, None, 0.0, [], []

        relevant_chunk_ids = [
            result.payload.get('chunk_id')
            for result in qdrant_results
            if result.payload.get('document_id') == top_document_id
        ]
        if not relevant_chunk_ids:
            logger.warning("Не найдены релевантные chunk_id.")
            return " ", None, None, None, 0.0, [], []

        stmt = (
            select(DocumentChunk.content, Document.web_link, Document.title, DocumentChunk.page_number, DocumentChunk.section)
           .join(Document, DocumentChunk.document_id == Document.document_id)
           .where(DocumentChunk.chunk_id.in_(relevant_chunk_ids))
           .order_by(DocumentChunk.chunk_id)
        )
        sql_results = (await session.execute(stmt)).fetchall()

        if not sql_results:
            logger.warning("В PostgreSQL не найдены результаты для данных chunk_id.")
            return " ", None, None, None, 0.0, [], []

        full_context = [result.content for result in sql_results]
        page_numbers = list(set(result.page_number for result in sql_results if result.page_number))
        sections = list(set(result.section for result in sql_results if result.section))
        web_link = sql_results[0].web_link
        title = sql_results[0].title
        context = "\n\n".join(full_context)
        max_score = max(result.score for result in qdrant_results)

        logger.info("Полный контекст успешно извлечен.")
        return context, web_link, title, top_document_id, max_score, page_numbers, sections

class LLMGenerator:
    def __init__(self, api_key: str, model_name: str):
        self._model_name = model_name
        # Инициализация клиента будет здесь, когда будет ясность с моделью
        
    async def generate_rag_response(self, context: str, user_query: str, system_instructions: str, title: str, web_link: str, page_numbers: list[int], sections: list[str], low_precision: bool = False) -> str:
        logger.info("Генерация RAG ответа...")
        temperature = 0.6 if low_precision else 0.1

        pages_str = ", ".join(map(str, sorted(page_numbers)))
        sections_str = ", ".join(sorted(sections))

        final_system_prompt = system_instructions.format(
            web_link=web_link,
            pages_str=pages_str,
            sections_str=sections_str,
            title=title,
            context=context
        )

        # Здесь будет логика вызова LLM
        # Заглушка для демонстрации
        await asyncio.sleep(1) # имитация асинхронной операции

        return f"Сгенерированный ответ на основе: {title}, страницы: {pages_str}, разделы: {sections_str}"


class PromptManager:
    PROMPT_MAPPING = {
        "ID_DEFAULT": (
            "ТЫ — **ЭКСПЕРТ ПО НОРМАТИВНОЙ ДОКУМЕНТАЦИИ**. Твой ответ должен быть основан **ТОЛЬКО** на информации из `<КОНТЕКСТ>`. "
            "Отвечай кратко, структурировано и по делу. В конце ответа добавь секцию `Использованные Источники`.\n\n"
            "### Требования:\n"
            "1.  **Краткое заключение (Summary)** в начале.\n"
            "2.  **Детали** в виде нумерованного или маркированного списка.\n"
            "3.  **Ключевые термины** — жирным шрифтом.\n"
            "4.  В конце ответа, после `---`, секция `### Использованные Источники`:\n"
            "    *   **Разделы:** {sections_str}\n"
            "    *   **Страницы:** {pages_str}\n"
            "    *   **Веб-ссылка:** [{title}]({web_link})\n\n"
            "<КОНТЕКСТ>\n{context}\n</КОНТЕКСТ>"
        ),
    }
    NOT_FOUND_PROMPT = "Извините, в предоставленных документах точный ответ не найден. Уточните вопрос."

    def get_instructions(self, document_id: str | None) -> str:
        return self.PROMPT_MAPPING.get(document_id, self.PROMPT_MAPPING["ID_DEFAULT"])

    def get_not_found_message(self):
        return self.NOT_FOUND_PROMPT

class RAGService:
    def __init__(self, embedder: QueryEmbeddingService, searcher: QueryQdrantClient, 
                 retriever: ContextRetriever, generator: LLMGenerator, session_factory,
                 prompt_manager: PromptManager):
        self._embedder = embedder
        self._searcher = searcher
        self._retriever = retriever
        self._generator = generator
        self._SessionLocal = session_factory
        self._prompt_manager = prompt_manager

    async def aquery(self, user_query: str, low_precision: bool = False) -> tuple:
        logger.info("Запуск RAG пайплайна...")
        query_vector = await self._embedder.vectorize_query(user_query)
        qdrant_results = await self._searcher.semantic_search(query_vector)

        async with self._SessionLocal() as session:
            context, web_link, title, doc_id, score, page_numbers, sections = await self._retriever.retrieve_full_context(qdrant_results, session)

        if not context.strip():
            logger.warning("Контекст пуст, возврат сообщения по умолчанию.")
            return self._prompt_manager.get_not_found_message(), None, 0.0, None, [], []
        
        system_instructions = self._prompt_manager.get_instructions(doc_id)
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(context)
        if len(tokens) > 1000:
            context = tokenizer.decode(tokens[:1000])
        
        final_answer = await self._generator.generate_rag_response(
            context=context, user_query=user_query, system_instructions=system_instructions,
            low_precision=low_precision, title=title, web_link=web_link,
            page_numbers=page_numbers, sections=sections
        )
        logger.info("RAG пайплайн успешно завершен.")
        return final_answer, web_link, score, title, page_numbers, sections
