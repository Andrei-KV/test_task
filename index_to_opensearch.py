"""
Document ingestion script for OpenSearch.
Indexes documents with embeddings into OpenSearch using bulk API.
"""
import asyncio
import logging
from opensearchpy import AsyncOpenSearch, helpers
from src.config import (
    OPENSEARCH_INDEX
)
from src.services.vectorization import indexing_pipe_line

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PayloadObject:
    """Helper class to convert dict to object for IndexingPipeline compatibility"""
    def __init__(self, data):
        self.content = data.get('content')
        self.document_id = data.get('document_id')
        self.document_title = data.get('document_title', 'Test Document') # Default title for test
        self.chunk_id = data.get('chunk_id')
        self.chunk_index = data.get('chunk_id') # Reuse chunk_id as index for test
        self.page_number = data.get('page_number')
        self.content_type = data.get('type')
        self.sheet_name = data.get('sheet_name')
        self.qdrant_id = data.get('qdrant_id')

async def index_documents_to_opensearch(chunks_data: list[dict]):
    """
    Index document chunks into OpenSearch using the shared IndexingPipeline.
    
    Args:
        chunks_data: List of chunk dictionaries.
    """
    logger.info(f"Starting indexing of {len(chunks_data)} chunks to OpenSearch...")
    
    # Convert dicts to objects expected by IndexingPipeline
    chunk_objects = [PayloadObject(chunk) for chunk in chunks_data]
    
    try:
        # Run pipeline (it is synchronous in current implementation)
        indexing_pipe_line.run(chunk_objects)
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise


async def test_indexing():
    """Test indexing with sample data."""
    logger.info("=" * 80)
    logger.info("TESTING DOCUMENT INDEXING TO OPENSEARCH")
    logger.info("=" * 80)
    
    # Sample test data
    test_chunks = [
        {
            'content': 'Это тестовый документ для проверки индексации в OpenSearch.',
            'document_id': 999,
            'chunk_id': 0,
            'page_number': 1,
            'type': 'text',
            'qdrant_id': 'test-chunk-001'
        },
        {
            'content': 'Второй тестовый чанк с информацией о гибридном поиске.',
            'document_id': 999,
            'chunk_id': 1,
            'page_number': 1,
            'type': 'text',
            'qdrant_id': 'test-chunk-002'
        }
    ]
    
    await index_documents_to_opensearch(test_chunks)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ INDEXING TEST COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_indexing())
