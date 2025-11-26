"""
Test script for OpenSearch connection and index creation.
Run this after starting OpenSearch with docker-compose.
"""
import asyncio
import logging
from src.services.opensearch_client import opensearch_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_opensearch_connection():
    """Test OpenSearch connection and create index."""
    logger.info("=" * 80)
    logger.info("TESTING OPENSEARCH CONNECTION")
    logger.info("=" * 80)
    
    try:
        # Test connection by getting cluster info
        info = await opensearch_client._QueryOpenSearchClient__client.info()
        logger.info(f"✅ Connected to OpenSearch cluster:")
        logger.info(f"   Version: {info['version']['number']}")
        logger.info(f"   Cluster: {info['cluster_name']}")
        
        # Create index
        logger.info("\nCreating index...")
        await opensearch_client.create_index()
        logger.info("✅ Index creation completed")
        
        # Get index info
        index_info = await opensearch_client._QueryOpenSearchClient__client.indices.get(
            index=opensearch_client._QueryOpenSearchClient__index_name
        )
        logger.info(f"\n✅ Index '{opensearch_client._QueryOpenSearchClient__index_name}' details:")
        logger.info(f"   Mappings: {list(index_info[opensearch_client._QueryOpenSearchClient__index_name]['mappings']['properties'].keys())}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        raise
    finally:
        await opensearch_client.close()


if __name__ == "__main__":
    asyncio.run(test_opensearch_connection())
