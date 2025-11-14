import asyncio
import logging
import time
from uuid import uuid4

import schedule
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from src.config import SERVICE_ACCOUNT_FILE, TARGET_FOLDER_ID
from src.database.database import AsyncSessionLocal, async_engine, init_db
from src.database.models import Document, DocumentChunk
from src.services.document_processor import (
    clean_text,
    parse_doc,
    parse_docx,
    parse_md,
    parse_excel,
    parse_image,
    parse_pdf,
    parse_rtf,
    parse_txt,
    split_text_into_chunks,
)
from src.services.google_drive import (
    download_drive_file_content,
    get_drive_web_link,
    init_drive_service,
    list_files_in_folder,
)
from src.services.vectorization import indexing_pipe_line

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Variables check
if (SERVICE_ACCOUNT_FILE is None) or (TARGET_FOLDER_ID is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")


async def process_new_documents():
    """Checks for new documents in Google Drive, processes them, and adds them to the database."""
    logger.info("Checking for new documents...")

    async with async_engine.connect() as conn:

        def has_table(sync_conn):
            inspector = inspect(sync_conn)
            return inspector.has_table("documents")

        table_exists = await conn.run_sync(has_table)

    if not table_exists:
        logger.info("Database not found, initializing...")
        await init_db()
        logger.info("Database initialized.")

    drive_service = init_drive_service(SERVICE_ACCOUNT_FILE)
    if not drive_service:
        logger.error("Failed to initialize Google Drive service.")
        return

    files_in_folder = list_files_in_folder(drive_service, TARGET_FOLDER_ID)
    if not files_in_folder:
        logger.info("No files found in the target folder.")
        return

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document.drive_file_id))
        existing_ids = result.scalars().all()
        # To Set для быстрой проверки
        existing_ids_set = set(existing_ids)

        for file_info in files_in_folder:
            file_id = file_info["ID Файла"]
            file_name = file_info["Имя Файла"]
            file_mime_type = file_info["MIME Тип"]
            # Check doc in database
            if file_id in existing_ids_set:
                # Документ уже проиндексирован, пропускаем
                logger.info(f"Document: {file_name} had indexing befor")
                continue

            logger.info(f"Processing new document: {file_name}")

            web_link = get_drive_web_link(drive_service, file_id)
            if not web_link:
                logger.warning(f"Could not get web link for file: {file_name}")

                continue

            new_document = Document(
                title=file_name,
                drive_file_id=file_id,
                web_link=web_link,
            )
            session.add(new_document)
            await session.commit()
            await session.refresh(new_document)
            document_id = new_document.document_id
            logger.info(f"Added new document to the database: {file_name}")

            raw_content_bytes = download_drive_file_content(
                drive_service, file_id, file_name
            )
            if raw_content_bytes is None:
                continue

            pages = []
            if (
                file_mime_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                pages = parse_docx(raw_content_bytes)
            elif file_mime_type == "application/pdf":
                pages = parse_pdf(raw_content_bytes)
            elif file_mime_type == "application/msword":
                pages = parse_doc(raw_content_bytes)
            elif file_mime_type == "application/rtf":
                pages = parse_rtf(raw_content_bytes)
            elif file_mime_type == "text/markdown":
                pages = parse_md(raw_content_bytes.decode("utf-8"))
            elif file_mime_type == "text/plain":
                pages = parse_txt(raw_content_bytes.decode("utf-8"))
            elif (
                file_mime_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                or file_mime_type == "application/vnd.ms-excel"
            ):
                pages = parse_excel(raw_content_bytes)
            elif file_mime_type.startswith("image/"):
                pages = parse_image(raw_content_bytes)
            else:
                logger.warning(f"Unsupported file format: {file_mime_type}")
                continue
            
            previous_overlap_sentences = []
            chunk_objects_to_process = []
            for page_content, page_num in pages:
                # cleaned_content = clean_text(page_content)
                chunks, current_tail = split_text_into_chunks(
                    page_content, 
                    chunk_size=300, 
                    overlap=50,
                    previous_overlap_sentences=previous_overlap_sentences # tail prev page
                )
                previous_overlap_sentences = current_tail
                for chunk in chunks:
                    qdrant_uuid = str(uuid4())
                    new_document_chunk = DocumentChunk(
                        document_id=document_id,
                        content=chunk,
                        page_number=page_num,
                        qdrant_id=qdrant_uuid,
                    )
                    session.add(new_document_chunk)
                    chunk_objects_to_process.append(new_document_chunk)
            await session.commit()
            logger.info("Committed new chunks to the database.")
            indexing_pipe_line.run(chunk_objects_to_process)

    logger.info("Finished checking for new documents.")


if __name__ == "__main__":
    asyncio.run(process_new_documents())
