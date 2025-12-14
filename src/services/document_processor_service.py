import logging
from typing import List, Dict, Any
from src.services.document_parser import document_parser
from src.services.chunking_service import chunking_service
from src.app.logging_config import get_logger

logger = get_logger(__name__)

class DocumentProcessorService:
    """
    Orchestrates the document processing pipeline:
    1. Parse document (DocumentParser)
    2. Chunk content (ChunkingService)
    """

    def process_document(
        self, 
        file_content: bytes, 
        file_name: str, 
        mime_type: str,
        document_id: int,
        document_title: str,
        max_pages: int = None
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline: Downloaded bytes -> Parsed Pages -> Chunks with Metadata.
        """
        logger.info(f"Starting processing pipeline for: {file_name}")

        # 1. Parse
        parsed_pages = document_parser.parse_file(file_content, file_name, mime_type, max_pages=max_pages)
        
        if not parsed_pages:
            logger.warning(f"No content extracted from {file_name}")
            return []

        logger.info(f"Extracted {len(parsed_pages)} pages/sections from {file_name}")

        # 2. Chunk
        chunks_data = chunking_service.create_chunks_with_metadata(
            parsed_pages=parsed_pages,
            document_id=document_id,
            document_title=document_title,
            chunk_size=1000, # Configurable
            overlap=150
        )

        logger.info(f"Created {len(chunks_data)} chunks for {file_name}")
        return chunks_data

document_processor_service = DocumentProcessorService()
