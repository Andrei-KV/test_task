import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from opensearchpy import OpenSearch
from dotenv import load_dotenv

load_dotenv()

async def check_postgres():
    print("--- PostgreSQL Chunks ---")
    db_uri = os.getenv('DB_URI')
    if not db_uri:
        print("DB_URI not found in environment")
        return
        
    try:
        engine = create_async_engine(db_uri)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT document_title, count(*) FROM document_chunks GROUP BY document_title;"))
            rows = result.fetchall()
            if not rows:
                print("No records found in document_chunks table.")
            for row in rows:
                print(f"Document: {row[0]}, Chunks: {row[1]}")
    except Exception as e:
        print(f"PostgreSQL Error: {e}")
    finally:
        await engine.dispose()

def check_opensearch():
    print("\n--- OpenSearch Indices ---")
    try:
        host = os.getenv('OPENSEARCH_HOST', 'opensearch')
        port = int(os.getenv('OPENSEARCH_PORT', 9200))
        auth = ('admin', os.getenv('OPENSEARCH_PASSWORD', 'admin'))
        client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=False
        )
        indices = client.cat.indices(format='json')
        if not indices:
            print("No indices found.")
        for idx in indices:
            print(f"Index: {idx['index']}, Docs count: {idx['docs.count']}")
            
        # Also check the specific index if possible
        index_name = os.getenv('OPENSEARCH_INDEX', 'rag_chunks')
        if client.indices.exists(index_name):
            count = client.count(index=index_name)['count']
            print(f"Specific index '{index_name}' count: {count}")
    except Exception as e:
        print(f"OpenSearch Error: {e}")

async def main():
    await check_postgres()
    check_opensearch()

if __name__ == "__main__":
    asyncio.run(main())
