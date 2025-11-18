import asyncio
from qdrant_client import QdrantClient
from src.config import QDRANT_HOST, COLLECTION_NAME

def diagnose_qdrant():
    """
    Connects to Qdrant and retrieves diagnostic information about the collection,
    index configuration, and sample data points.
    """
    if not QDRANT_HOST or not COLLECTION_NAME:
        print("❌ Error: QDRANT_HOST or COLLECTION_NAME is not set in the .env file.")
        return

    print(f"--- Qdrant Diagnostics ---")
    print(f"Connecting to Qdrant at: {QDRANT_HOST}")
    print(f"Target collection: {COLLECTION_NAME}")
    print("-" * 26)

    try:
        # Using a synchronous client for a simple script is fine
        client = QdrantClient(url=QDRANT_HOST)

        # 1. Check collection existence and configuration
        print("\n[1/2] Fetching Collection Info...")
        try:
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            
            print(f"✅ Collection '{COLLECTION_NAME}' found.")
            
            # Check for the payload index on the 'content' field
            payload_schema = collection_info.payload_schema
            if "content" in payload_schema:
                content_index_params = payload_schema["content"].params
                print("\n--- Payload Index on 'content' field ---")
                print(f"  Index Type: {content_index_params.type}")
                if hasattr(content_index_params, 'tokenizer'):
                    print(f"  Tokenizer: {content_index_params.tokenizer.value}")
                if hasattr(content_index_params, 'lowercase'):
                    print(f"  Lowercase: {content_index_params.lowercase}")
                print("-----------------------------------------")
                if hasattr(content_index_params, 'tokenizer') and content_index_params.tokenizer.value.upper() == 'WORD':
                    print("✅ Configuration seems correct.")
                else:
                    print("❌ WARNING: Tokenizer is NOT 'word'. This is likely the cause of the problem.")

            else:
                print("❌ ERROR: No payload index found for the 'content' field.")
                print("   This is the root cause. The index is not being created.")

        except Exception as e:
            print(f"❌ ERROR: Could not get collection info for '{COLLECTION_NAME}'.")
            print(f"   Reason: {e}")
            print("   This likely means the collection does not exist or Qdrant is down.")
            return

        # 2. Fetch a few sample points to inspect their payload
        print("\n[2/2] Fetching Sample Data Points...")
        try:
            records, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=3,
                with_payload=True,
                with_vectors=False
            )

            if not records:
                print("❌ WARNING: The collection is empty. No data has been indexed.")
                print("   Please run your indexing script (`process_new_documents_test.py`).")
                return
            
            print(f"✅ Found {len(records)} sample points. Inspecting payload:")
            for i, record in enumerate(records):
                print(f"\n--- Sample Point {i+1} (ID: {record.id}) ---")
                payload = record.payload
                if payload:
                    print(f"  document_id: {payload.get('document_id')}")
                    print(f"  chunk_id: {payload.get('chunk_id')}")
                    content = payload.get('content', '!!! NOT FOUND !!!')
                    print(f"  'content' field exists: {'content' in payload}")
                    print(f"  'content' preview: {content[:150]}...")
                else:
                    print("  ❌ ERROR: Payload is empty for this point.")
            print("-" * 26)

        except Exception as e:
            print(f"❌ ERROR: Could not scroll data from '{COLLECTION_NAME}'.")
            print(f"   Reason: {e}")

    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to Qdrant at {QDRANT_HOST}.")
        print(f"   Reason: {e}")
        print("   Please ensure Docker and the Qdrant container are running correctly.")


if __name__ == "__main__":
    diagnose_qdrant()
