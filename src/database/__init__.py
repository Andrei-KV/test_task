from .database import init_db, SessionLocal
from .models import Document, DocumentChunk

# The init_db() function is commented out to prevent it from running on import
# init_db()