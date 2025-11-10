from ...services.rag_service import (
    QueryEmbeddingService,
    QueryQdrantClient,
    ContextRetriever,
    LLMGenerator,
    PromptManager,
    RAGService,
)
from ...database.database import AsyncSessionLocal
from ...config import EMBEDDING_MODEL_NAME, QDRANT_HOST, COLLECTION_NAME, DEEPSEEK_API_KEY, LLM_MODEL

async def get_rag_service():
    """
    Dependency provider for the RAGService.
    Initializes all the necessary components and yields a RAGService instance.
    """
    embedder = QueryEmbeddingService(model_name=EMBEDDING_MODEL_NAME)
    searcher = QueryQdrantClient(host=QDRANT_HOST, collection_name=COLLECTION_NAME)
    retriever = ContextRetriever()
    generator = LLMGenerator(api_key=DEEPSEEK_API_KEY, model_name=LLM_MODEL)
    prompt_manager = PromptManager()

    rag_service = RAGService(
        embedder=embedder,
        searcher=searcher,
        retriever=retriever,
        generator=generator,
        session_factory=AsyncSessionLocal,
        prompt_manager=prompt_manager,
    )
    try:
        yield rag_service
    finally:
        # Add cleanup logic here if needed (e.g., closing connection pools)
        pass
