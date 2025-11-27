import asyncio
import logging
import os
from dotenv import load_dotenv
from src.services.rag_service import QueryEmbeddingService
from src.config import EMBEDDING_MODEL_NAME, GEMINI_API_KEY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_embedding_service():
    logger.info("Testing QueryEmbeddingService...")
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found!")
        return

    service = QueryEmbeddingService(api_key=GEMINI_API_KEY, model_name=EMBEDDING_MODEL_NAME)
    
    query = "Как настроить RAG систему?"
    logger.info(f"Vectorizing query: '{query}'")
    
    try:
        vector = await service.vectorize_query(query)
        dim = len(vector)
        logger.info(f"✅ Vector generated successfully.")
        logger.info(f"Dimension: {dim}")
        logger.info(f"First 5 values: {vector[:5]}")
        
        if dim == 3072:
            logger.info("✅ Dimension check passed (3072).")
        else:
            logger.error(f"❌ Dimension mismatch! Expected 3072, got {dim}")
            
    except Exception as e:
        logger.error(f"❌ Error during vectorization: {e}")

if __name__ == "__main__":
    asyncio.run(test_embedding_service())
