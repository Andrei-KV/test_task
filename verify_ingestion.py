import asyncio
import os
from opensearchpy import AsyncOpenSearch
from src.config import (
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    OPENSEARCH_USE_SSL,
    OPENSEARCH_VERIFY_CERTS
)

# Force localhost for local verification script
OPENSEARCH_HOST = "localhost"

async def verify_ingestion():
    client = AsyncOpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=None,
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False
    )

    print(f"Connecting to OpenSearch at {OPENSEARCH_HOST}:{OPENSEARCH_PORT}...")

    try:
        # 1. Check total count
        if not await client.indices.exists(index=OPENSEARCH_INDEX):
            print(f"Index '{OPENSEARCH_INDEX}' does not exist!")
            return

        count_response = await client.count(index=OPENSEARCH_INDEX)
        total_chunks = count_response['count']
        print(f"Total chunks indexed: {total_chunks}")

        # 2. Get unique document IDs (using terms aggregation)
        # We want to find 3 distinct documents.
        aggs_query = {
            "size": 0,
            "aggs": {
                "unique_docs": {
                    "terms": {
                        "field": "document_id",
                        "size": 10  # Get a few to pick from
                    }
                }
            }
        }
        
        response = await client.search(index=OPENSEARCH_INDEX, body=aggs_query)
        buckets = response['aggregations']['unique_docs']['buckets']
        
        if not buckets:
            print("No documents found in the index.")
            return

        print(f"Found {len(buckets)} unique documents (showing top 10).")
        
        docs_to_export = buckets[:3]
        output_file = "chunks_verification.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Total chunks in index: {total_chunks}\n")
            f.write("="*50 + "\n\n")

            for doc_bucket in docs_to_export:
                doc_id = doc_bucket['key']
                doc_count = doc_bucket['doc_count']
                
                f.write(f"Document ID: {doc_id} (Chunks: {doc_count})\n")
                f.write("-" * 30 + "\n")
                
                # Fetch chunks for this document, sorted by chunk_id or page_number
                # Note: Assuming 'chunk_id' is sortable or we use 'page_number'
                # Let's try to sort by 'chunk_id' if available, else 'page_number'
                search_query = {
                    "size": 100, # Limit chunks per doc for safety
                    "query": {
                        "term": {
                            "document_id": doc_id
                        }
                    },
                    "sort": [
                        {"chunk_id": {"order": "asc"}}
                    ]
                }
                
                chunks_resp = await client.search(index=OPENSEARCH_INDEX, body=search_query)
                hits = chunks_resp['hits']['hits']
                
                # Try to get title from the first hit if available
                if hits:
                    source = hits[0]['_source']
                    # Assuming title might be in metadata or we just rely on content
                    # Based on previous code, title is stored in postgres, but maybe we put it in OS too?
                    # Let's check _source keys
                    pass

                for hit in hits:
                    source = hit['_source']
                    content = source.get('content', '[No Content]')
                    page = source.get('page_number', '?')
                    chunk_id = source.get('chunk_id', '?')
                    
                    f.write(f"[Chunk {chunk_id} | Page {page}]\n")
                    f.write(f"{content}\n")
                    f.write("\n")
                
                f.write("="*50 + "\n\n")
        
        print(f"Verification data saved to {output_file}")

    except Exception as e:
        print(f"Error during verification: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(verify_ingestion())
