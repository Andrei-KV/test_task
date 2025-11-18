
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import TextIndexParams, TokenizerType
from src.config import QDRANT_HOST, COLLECTION_NAME

async def create_full_text_index():
    """Creates a full-text index in the Qdrant collection."""
    client = AsyncQdrantClient(url=QDRANT_HOST)
    
    try:
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="content",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                lowercase=True
            )
        )
        print("Full-text index on 'content' field created successfully.")
    except Exception as e:
        print(f"Error creating full-text index: {e}")

if __name__ == "__main__":
    asyncio.run(create_full_text_index())
