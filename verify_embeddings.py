"""
Script to verify Google Gemini Embeddings API access.
"""
import os
import asyncio
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

async def verify_embeddings():
    print("=" * 60)
    print("VERIFYING GEMINI EMBEDDINGS API ACCESS")
    print("=" * 60)

    if not API_KEY:
        print("❌ ERROR: GEMINI_API_KEY not found in environment variables.")
        return

    print(f"API Key found: {API_KEY[:5]}...{API_KEY[-5:]}")

    try:
        client = genai.Client(api_key=API_KEY)
        
        print("\nListing available models...")
        try:
            async for model in await client.aio.models.list():
                # print(f"DEBUG: Model object keys: {dir(model)}")
                print(f" - {model.name}")
        except Exception as list_err:
            print(f"⚠️ Warning: Could not list models: {list_err}")

        # Try gemini-embedding-001 as requested by user
        test_model = "models/gemini-embedding-001" 
        print(f"\nTrying requested model: {test_model}...")
        
        text_to_embed = "This is a test sentence."
        result = await client.aio.models.embed_content(
            model=test_model,
            contents=text_to_embed
        )

        embedding = result.embeddings[0].values
        dim = len(embedding)
        
        print(f"\n✅ SUCCESS with {test_model}!")
        print(f"Dimension: {dim}")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to list models or generate embedding.")
        print(f"Details: {e}")

if __name__ == "__main__":
    asyncio.run(verify_embeddings())
