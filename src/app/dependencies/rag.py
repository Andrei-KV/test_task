from functools import lru_cache
from ...services.rag_service_opensearch import (
    QueryEmbeddingService,
    ContextRetriever,
    LLMGenerator,
    PromptManager,
    RAGService,
)
from ...services.opensearch_client import QueryOpenSearchClient
from ...database.database import AsyncSessionLocal
from ...config import (
    EMBEDDING_MODEL_NAME, 
    DEEPSEEK_API_KEY, 
    LLM_MODEL, 
    GEMINI_API_KEY,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    OPENSEARCH_USE_SSL,
    OPENSEARCH_VERIFY_CERTS
)

@lru_cache(maxsize=1)
def get_embedding_service():
    return QueryEmbeddingService(api_key=GEMINI_API_KEY, model_name=EMBEDDING_MODEL_NAME)

@lru_cache(maxsize=1)
def get_opensearch_client():
    return QueryOpenSearchClient(
        host=OPENSEARCH_HOST,
        port=OPENSEARCH_PORT,
        index_name=OPENSEARCH_INDEX,
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS
    )

@lru_cache(maxsize=1)
def get_llm_generator():
    if LLM_MODEL == "deepseek-chat":
        return LLMGenerator(api_key=DEEPSEEK_API_KEY, model_name=LLM_MODEL)
    else:
        return LLMGenerator(api_key=GEMINI_API_KEY, model_name=LLM_MODEL)

@lru_cache(maxsize=1)
def get_prompt_manager():
    return PromptManager()

async def get_rag_service():
    """
    Dependency provider for the RAGService.
    Initializes all the necessary components and yields a RAGService instance.
    """
    embedder = get_embedding_service()
    searcher = get_opensearch_client()
    # ContextRetriever is lightweight, can be instantiated every time or cached, 
    # but let's keep it simple as it has no state.
    retriever = ContextRetriever() 
    generator = get_llm_generator()
    prompt_manager = get_prompt_manager()

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
