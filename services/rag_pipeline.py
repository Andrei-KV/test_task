from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from openai import OpenAI
from database.database import SessionLocal
from database.models import Document, DocumentChunk
from services.vectorization import get_embedding_model, get_qdrant_client
from config import COLLECTION_NAME, LLM_MODEL, DEEPSEEK_API_KEY
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Variables check
if (LLM_MODEL is None) or (COLLECTION_NAME is None) or (DEEPSEEK_API_KEY is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")


def vectorize_query(query: str, model: SentenceTransformer) -> list[float]:
    """Vectorizes a single text query."""
    logger.info("Vectorizing user query...")
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_tensor=False
    ).tolist()[0]
    logger.info("User query vectorized successfully.")
    return query_embedding

def semantic_search(query_vector: list[float], qdrant_client: QdrantClient, limit_k: int = 7):
    """Performs a semantic search in Qdrant."""
    logger.info("Performing semantic search in Qdrant...")
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
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

def retrieve_full_context(qdrant_results, session: Session) -> tuple:
    """Retrieves the full text context from PostgreSQL based on Qdrant results."""
    logger.info("Retrieving full context from PostgreSQL...")
    logger.info(f"qdrant_results: {qdrant_results}")
    try:
        top_document_id = qdrant_results[0].payload.get('document_id')
        logger.info(f"top_document_id: {top_document_id}")
    except (IndexError, KeyError):
        logger.warning("No document ID found in Qdrant results.")
        return " ", None

    relevant_chunk_ids = [
        result.payload.get('chunk_id')
        for result in qdrant_results
        if result.payload.get('document_id') == top_document_id
    ]
    logger.info(f"relevant_chunk_ids: {relevant_chunk_ids}")
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
    return context, web_link

def generate_rag_response(context: str, user_query: str) -> str:
    """Generates a response using the RAG model."""
    logger.info("Generating RAG response...")
    SYSTEM_INSTRUCTIONS = (
        "Вы — юридический ассистент. Ваша задача — синтезировать "
        "понятный и связный ответ на вопрос пользователя, используя "
        "предоставленный ниже контекст. Если контекст не содержит информации, "
        "достаточной для ответа, вы должны ответить: 'Извините, в предоставленных "
        "документах точный ответ не найден. Уточните вопрос' Сохраняйте профессиональный тон и "
        "отвечайте на русском языке."
    )
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": f"{SYSTEM_INSTRUCTIONS}--- КОНТЕКСТ ---{context}"},
                {"role": "user", "content": user_query},
            ],
            stream=False,
            temperature=0.8, # Увеличиваем случайность (свободу)
            top_p=0.9,       # Увеличиваем разнообразие выбора токенов
        )
        if not response:
            logger.error("Error generating response: No response object.")
            return 'Ошибка генерации ответа'
        logger.info("RAG response generated successfully.")
        return response.choices[0].message.content # type: ignore
    except Exception as e:
        logger.error(f"An unexpected error occurred during generation: {e}")
        return f"❌ Произошла непредвиденная ошибка при генерации: {e}"

def rag_pipeline(user_query: str) -> tuple[str, str | None]:
    import sys
    """The main RAG pipeline."""
    logger.info("Starting RAG pipeline...")
    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()

    query_vector = vectorize_query(user_query, embedding_model)
    qdrant_results = semantic_search(query_vector, qdrant_client)

    with SessionLocal() as session:
        context, web_link = retrieve_full_context(qdrant_results, session)

    if not context.strip():
        logger.warning("Context is empty, returning a default message.")
        return "Извините, в предоставленных документах точный ответ не найден.", None
    
    # -----------------------------------------------------
    # 🟢 ЛОГИКА ИЗМЕРЕНИЯ РАЗМЕРА КОНТЕКСТА В МБ 🟢
    # -----------------------------------------------------
    
    # 1. Измерение размера в байтах с кодировкой UTF-8
    size_bytes = len(context.encode('utf-8'))
    
    # 2. Конвертация байтов в мегабайты (1 МБ = 1024 * 1024 байт)
    size_mb = size_bytes / (1024 * 1024)
    
    # 3. Вывод в логи
    logger.info(f"💾 Контекст из базы: {size_bytes} байт, что составляет {size_mb:.4f} МБ.")
    
    # -----------------------------------------------------

    final_answer = generate_rag_response(context, user_query)
    logger.info("RAG pipeline finished successfully.")
    return final_answer, web_link
