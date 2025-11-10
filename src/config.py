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
QDRANT_HOST = os.getenv('QDRANT_HOST')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')

# LLM configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
LLM_MODEL = "deepseek-chat"

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

# Telegram bot configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
