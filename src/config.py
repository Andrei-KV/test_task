import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_URI = os.getenv('DB_URI')

# Google Drive configuration
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
TARGET_FOLDER_ID = os.getenv('TARGET_FOLDER_ID')
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "rag_chunks")
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"


# Embedding model configuration
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 3072
# RERANKER_MODEL_NAME = os.getenv('RERANKER_MODEL_NAME')

# LLM configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
LLM_MODEL = "deepseek-chat"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# LLM_MODEL = "gemini-2.5-flash"
RESERVE_LLM_MODEL = "deepseek-chat"

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

# Telegram bot configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
