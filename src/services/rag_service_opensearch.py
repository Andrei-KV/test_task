"""
RAG Service with OpenSearch support.
This is a new version that uses OpenSearch instead of Qdrant for hybrid search.
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import CrossEncoder
import tiktoken
import numpy as np
from openai import AsyncOpenAI
from ..database.database import AsyncSessionLocal
from ..database.models import Document, DocumentChunk
from ..config import LLM_MODEL, DEEPSEEK_API_KEY, EMBEDDING_MODEL_NAME, MAX_CONTEXT_TOKENS
from src.app.logging_config import get_logger
from google import genai
from google.genai.types import GenerateContentConfig
from .opensearch_client import QueryOpenSearchClient, opensearch_client
from ..config import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    OPENSEARCH_USE_SSL,
    OPENSEARCH_VERIFY_CERTS
)

logger = get_logger(__name__)


# Variables check
if (LLM_MODEL is None) or (DEEPSEEK_API_KEY is None) or (EMBEDDING_MODEL_NAME is None):
    raise ValueError("Переменные не найдены. Проверьте файл.env.")


# =====================================================================
from .retry_utils import retry_with_backoff

# =====================================================================
# Сервис векторизации запроса
class QueryEmbeddingService:
    def __init__(self, api_key: str, model_name: str):
        self.__client = genai.Client(api_key=api_key)
        self.__model_name = model_name

    @retry_with_backoff
    async def vectorize_query(self, query: str) -> list[float]:
        """Векторизует один текстовый запрос для поиска."""
        logger.info("Vectorizing user query with Gemini...")
        try:
            result = await self.__client.aio.models.embed_content(
                model=self.__model_name,
                contents=query,
                config=genai.types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=3072
                )
            )
            logger.info("User query vectorized successfully.")
            # result.embeddings is a list, we take the first one
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error vectorizing query: {e}")
            raise


from sqlalchemy import or_

# Extact full context from PostgreSQL
class ContextRetriever:
    """Инкапсулирует логику извлечения полного контекста из PostgreSQL."""

    async def retrieve_full_context(self, qdrant_results, top_document_id, session: AsyncSession) -> tuple:
        """Извлекает полный текстовый контекст по результатам поиска."""
        logger.info("Retrieving full context from PostgreSQL...")
    
        if not qdrant_results or not top_document_id:
            logger.warning("No document ID found in search results.")
            return " ", None, None, None, []

        from sqlalchemy import select

        # Build a set of conditions to fetch the required chunks and their neighbors
        conditions = []
        for chunk_id in qdrant_results:
            conditions.append(DocumentChunk.chunk_id == chunk_id)
            if chunk_id > 0:
                conditions.append(DocumentChunk.chunk_id == chunk_id - 1)
            conditions.append(DocumentChunk.chunk_id == chunk_id + 1)
            conditions.append(DocumentChunk.chunk_id == chunk_id + 2)
        
        # Retrieve all chunks from database based on the final list of chunk IDs
        stmt = (
            select(DocumentChunk.content, Document.web_link, Document.title, DocumentChunk.page_number, DocumentChunk.chunk_id)
            .join(Document)
            .where(
                Document.document_id == top_document_id,
                or_(*conditions)
            )
            .order_by(DocumentChunk.chunk_id)
            .distinct()
        )
        sql_results = (await session.execute(stmt)).fetchall()

        if not sql_results:
            logger.warning("No results found in PostgreSQL for the given chunk IDs.")
            return " ", None, None, None, []

        # Collect unique chunks while preserving order
        unique_chunks = []
        pages_str = ""
        for result in sql_results:
            unique_chunks.append({
                "content": result.content,
                "page_number": result.page_number
            })
        
        full_context = [f'[стр. {u_c['page_number']}] ' + u_c['content']  for u_c in unique_chunks]
        page_numbers = [u_c['page_number'] for u_c in unique_chunks]
        logger.debug(f'Page numbers collected: {page_numbers, type(page_numbers)}')
        
        def _format_page_ranges(pages: list[int]) -> str:
            if not pages:
                return ""
            pages = sorted(set(pages))
            ranges = []
            start = pages[0]
            
            for i in range(1, len(pages)):
                if pages[i] != pages[i-1] + 1:
                    end = pages[i-1]
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    start = pages[i]
            
            if start == pages[-1]:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{pages[-1]}")
                
            return ", ".join(ranges)
        # Format the page numbers into a string with ranges
        pages_str = _format_page_ranges(page_numbers)
        
        web_link = sql_results[0].web_link
        title = sql_results[0].title
        context = "\n".join(full_context)
        
        
        logger.info("Full context retrieved successfully.")
        logger.debug(f'Full context retrieved: {context}')
        
        return context, web_link, title, top_document_id, pages_str


class LLMGenerator:
    """Инкапсулирует клиента LLM, системный промпт и логику генерации."""

    def __init__(self, api_key: str, model_name: str):
        self.__model_name = model_name
        if self.__model_name == "deepseek-chat":
            self.__client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.__client = genai.Client(api_key=api_key)
        self.__model_name = model_name

    
    
    @retry_with_backoff
    async def generate_rag_response(self, context: str, user_query: str, system_instructions: str, title: str, web_link: str, page_numbers: str, low_precision: bool = False) -> str:
        """Генерирует ответ LLM с использованием контекста RAG."""
        logger.info("Generating RAG response...")
        temperature = 0.5 if low_precision else 0.2

        final_system_prompt = f"<КОНТЕКСТ>:\n{context}\n<КОНТЕКСТ>"
        final_system_prompt +=  f"Если в `<КОНТЕКСТ>` источник указан в формате **[стр.X]**, ты должен вставлять тег `[стр.X]` **непосредственно** после предложения или элемента списка, который ты использовал."
        final_system_prompt += f"Обязательно указывай номера пунктов или страниц, откуда взята информация внутри []."
        final_system_prompt += f"\nНа основе предоставленного выше <КОНТЕКСТ> ответь на запрост пользователя: {user_query}"

        try:
            if self.__model_name == "deepseek-chat":
                response = await self.__client.chat.completions.create(
                    model=self.__model_name,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": final_system_prompt},
                    ],
                    stream=False,
                    temperature=temperature,
                    top_p=0.75,
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
                    max_output_tokens=3000,
                    system_instruction=system_instructions, 
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
                usage = response.usage_metadata

                logger.info(f"Токены ввода (Ваш запрос + контекст): {usage.prompt_token_count}")
                logger.info(f"Токены вывода (Ответ модели): {usage.candidates_token_count}")
                logger.info(f"Всего токенов: {usage.total_token_count}")
            
            logger.info(f"RAG response generated successfully: {response}")
            return result_content
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during generation: {e}")
            # We re-raise here so the retry decorator can catch it. 
            # If we return an error string, retry won't trigger.
            raise

import os

def load_prompt(file_path: str) -> str:
    """Loads a prompt from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Выбор инструкций для LLM в зависимости от документа
class PromptManager:
    """Управляет выбором SYSTEM_INSTRUCTIONS на основе Document ID."""

    def __init__(self, prompts_dir: str = "src/prompts"):
        self.PROMPT_MAPPING = {
            "ID_DEFAULT": load_prompt(os.path.join(prompts_dir, "default.md")),
            "ID_GEMINI_2.5_FLASH_EXAMPLE": load_prompt(os.path.join(prompts_dir, "gemini.md")),
            "ID_DEEPSEEK": load_prompt(os.path.join(prompts_dir, "deepseek.md")),
        }
 
    # Promt for cases when no answer is found in documents
    NOT_FOUND_PROMPT = "Извините, в предоставленных документах точный ответ не найден. Уточните вопрос"


    def get_instructions_by_document_id(self, document_id: str) -> str:
        """Возвращает системные инструкции для заданного ID документа."""
        ID_DEFAULT = "ID_DEEPSEEK" if LLM_MODEL == "deepseek-chat" else "ID_GEMINI_2.5_FLASH_EXAMPLE"
        # Используем.get() для безопасного извлечения. Если ID не найден, 
        # возвращаем дефолтный промпт.
        return self.PROMPT_MAPPING.get(document_id,  self.PROMPT_MAPPING.get(ID_DEFAULT))

    def get_not_found_message(self):
         return self.NOT_FOUND_PROMPT
    
# =====================================================================
# ОРКЕСТРАТОР
# =====================================================================

class RAGService:
    """Центральный класс, управляющий последовательностью операций RAG с OpenSearch."""

    def __init__(self, embedder: QueryEmbeddingService, searcher: QueryOpenSearchClient, 
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
        """Основной метод, выполняющий полный цикл RAG с OpenSearch."""
        logger.info("Starting RAG pipeline with OpenSearch...")

        # 1. Vectorize user query
        query_vector = await self.__embedder.vectorize_query(user_query)
        
        # 2. Hybrid search in OpenSearch (knn + match)
        chunk_ids, top_document_id, max_score = await self.__searcher.semantic_search(query_vector, user_query)


        # 3. Retrieve full context from PostgreSQL
        async with self.__SessionLocal() as session:
            context, web_link, title, top_document_id, page_numbers = await self.__retriever.retrieve_full_context(chunk_ids, top_document_id, session)
            score = max_score if max_score else 0.0
        if not context.strip():
            logger.warning("Context is empty, returning a default message.")
            # Используем NOT_FOUND_PROMPT из менеджера промптов
            return self.__prompt_manager.get_not_found_message(), None, 0.0, None, None
        
        
        # Logic of selecting final system instructions based on document ID
        final_system_instructions = self.__prompt_manager.get_instructions_by_document_id(top_document_id)

        # Measuring context size
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(context)
        if len(tokens) > MAX_CONTEXT_TOKENS:
            context = tokenizer.decode(tokens[:MAX_CONTEXT_TOKENS])
            context += "\n[ОГРАНИЧЕНИЕ ПО ТОКЕНАМ]"
            tokens = tokenizer.encode(context)
        
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
