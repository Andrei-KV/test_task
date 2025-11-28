import asyncio
from sqlalchemy import select
from src.database.database import AsyncSessionLocal
from src.database.models import DocumentChunk, Document

async def show_chunks():
    async with AsyncSessionLocal() as session:
        # Join with Document to get titles
        stmt = select(DocumentChunk, Document.title).join(Document).limit(10)
        result = await session.execute(stmt)
        chunks = result.all()
        
        print(f"\n{'='*80}")
        print(f"TOP 10 CHUNKS FROM POSTGRESQL")
        print(f"{'='*80}\n")
        
        for i, (chunk, doc_title) in enumerate(chunks, 1):
            print(f"Chunk #{i}")
            print(f"Document: {doc_title}")
            print(f"Page: {chunk.page_number}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Content Preview: {chunk.content[:200]}...")
            print(f"{'-'*80}\n")

if __name__ == "__main__":
    asyncio.run(show_chunks())
