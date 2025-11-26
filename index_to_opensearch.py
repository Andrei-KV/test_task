"""
Document ingestion script for OpenSearch.
Indexes documents with embeddings into OpenSearch using bulk API.
"""
import asyncio
import logging
from google import genai
from opensearchpy import AsyncOpenSearch, helpers
from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL_NAME,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    OPENSEARCH_USE_SSL,
    OPENSEARCH_VERIFY_CERTS
)
from src.services.opensearch_client import opensearch_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def index_documents_to_opensearch(chunks_data: list[dict]):
    """
    Index document chunks into OpenSearch with embeddings.
    
    Args:
        chunks_data: List of chunk dictionaries with fields:
            - content: text content
            - document_id: document ID
            - chunk_id: chunk ID
            - page_number: page number
            - type: chunk type
            - sheet_name: (optional) sheet name for Excel
            - qdrant_id: unique ID
    """
    logger.info(f"Starting indexing of {len(chunks_data)} chunks to OpenSearch...")
    
    # Initialize Gemini client for embeddings
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Initialize OpenSearch client
    os_client = AsyncOpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=None,
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False
    )
    
    try:
        # Ensure index exists
        await opensearch_client.create_index()
        
        # Prepare bulk actions
        actions = []
        
        for i, chunk in enumerate(chunks_data):
            try:
                # Generate embedding with Gemini
                result = gemini_client.models.embed_content(
                    model=EMBEDDING_MODEL_NAME,
                    contents=chunk['content'],
                    config=genai.types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=3072
                    )
                )
                embedding = result.embeddings[0].values
                
                # Prepare document for indexing
                doc = {
                    '_index': OPENSEARCH_INDEX,
                    '_id': chunk['qdrant_id'],
                    '_source': {
                        'embedding': embedding,
                        'content': chunk['content'],
                        'document_id': chunk['document_id'],
                        'chunk_id': chunk['chunk_id'],
                        'page_number': chunk.get('page_number'),
                        'type': chunk.get('type'),
                        'sheet_name': chunk.get('sheet_name'),
                        'qdrant_id': chunk['qdrant_id']
                    }
                }
                actions.append(doc)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Prepared {i + 1}/{len(chunks_data)} documents for indexing")
                    
            except Exception as e:
                logger.error(f"Error preparing chunk {i}: {e}")
                continue
        
        # Bulk index
        if actions:
            logger.info(f"Bulk indexing {len(actions)} documents...")
            success, failed = await helpers.async_bulk(
                os_client,
                actions,
                chunk_size=100,
                raise_on_error=False
            )
            logger.info(f"✅ Indexed {success} documents successfully")
            if failed:
                logger.warning(f"⚠️ Failed to index {len(failed)} documents")
        
        # Refresh index to make documents searchable
        await os_client.indices.refresh(index=OPENSEARCH_INDEX)
        logger.info("✅ Index refreshed, documents are now searchable")
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise
    finally:
        await os_client.close()
        logger.info("OpenSearch client closed")


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
