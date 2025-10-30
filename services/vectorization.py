from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from config import QDRANT_HOST, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from database.models import DocumentChunk
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import uuid4

# Variables check
if (QDRANT_HOST is None) or (COLLECTION_NAME is None) or (EMBEDDING_MODEL_NAME is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")

def get_embedding_model():
    """Returns the embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_qdrant_client():
    """Returns the Qdrant client."""
    return QdrantClient(url=QDRANT_HOST)

def create_qdrant_collection(qdrant_client: QdrantClient, vector_dimension: int):
    """Creates a new collection in Qdrant if it doesn't already exist."""
    try:
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
            )
            print(f"✅ Collection '{COLLECTION_NAME}' created.")
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"❌ Qdrant error: {e}. Make sure Qdrant is running.")

def vectorize_chunks(embedding_model: SentenceTransformer, chunks: list[str]) -> list[list[float]]:
    """Vectorizes a list of text chunks."""
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=False
    ).tolist()
    return embeddings

def create_qdrant_points(chunk_objects: list[DocumentChunk], embeddings: list[list[float]]) -> list[PointStruct]:
    """Creates a list of PointStruct objects for upserting into Qdrant."""
    points = []
    for idx, vector in enumerate(embeddings):
        chunk_obj = chunk_objects[idx]
        qdrant_id_int = int(chunk_obj.qdrant_id.replace('-', '')[:15], 16)
        payload = {
            "chunk_id": chunk_obj.chunk_id,
            "document_id": chunk_obj.document_id,
            "content_preview": chunk_obj.content[:100] + "..."
        }
        point = PointStruct(
            id=qdrant_id_int,
            vector=vector,
            payload=payload
        )
        points.append(point)
    return points

def upsert_points_to_qdrant(qdrant_client: QdrantClient, points: list[PointStruct]):
    """Upserts a list of points into a Qdrant collection."""
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        print(f"✅ {len(points)} vectors successfully uploaded to Qdrant (Collection: {COLLECTION_NAME}).")
    except Exception as e:
        print(f"❌ Error uploading to Qdrant: {e}")
