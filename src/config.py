import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_URI = os.getenv('DB_URI')

# Google Drive configuration
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
TARGET_FOLDER_ID = os.getenv('TARGET_FOLDER_ID')
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']



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

# =====================================================================
# LLM Generation Configuration
# =====================================================================
# Temperature settings for different modes
LLM_TEMPERATURE_PRECISE = 0.2  # For normal queries (high accuracy)
LLM_TEMPERATURE_CREATIVE = 0.5 # For low_precision queries

# Sampling parameters
LLM_TOP_P = 0.75

# Token limits
# Token limits
LLM_MAX_OUTPUT_TOKENS = 2000  # Default limit (Deepseek)
LLM_MAX_OUTPUT_TOKENS_EXTENDED = 3000 # For Gemini or verbose models
LLM_MAX_INPUT_TOKENS = 4000   # Max context window for LLM input

# =====================================================================
# RAG Retrieval & OpenSearch Configuration
# =====================================================================
# Hybrid Search Parameters
SEARCH_LIMIT_FINAL_K = 15      # Final number of chunks to return to LLM context (Optimized for speed)
SEARCH_RERANK_LIMIT = 10      # Number of candidates to rerank

SEARCH_KNN_SIZE = 25          # Number of vector search candidates to fetch
SEARCH_BM25_SIZE = 25         # Number of keyword search candidates to fetch
SEARCH_RRF_K = 60             # Constant 'k' for Reciprocal Rank Fusion (Standard is 60)

# =====================================================================
# Document Processing & Chunking Configuration
# =====================================================================
CHUNKING_SIZE = 1000          # Target tokens per chunk
CHUNKING_OVERLAP = 150        # Overlap tokens between chunks
CHUNKING_OVERLAP_LIMIT_RATIO = 1.3 # Max overlap factor (1.3 * 150) before strict cut-off
CHUNKING_SENTENCE_HARD_LIMIT = 500 # Force split sentence if > this tokens

# Parser Timeouts
PARSER_PAGE_TIMEOUT = 30      # Seconds per page processing
PARSER_MAX_CONSECUTIVE_TIMEOUTS = 3 # Max timeouts before aborting doc
