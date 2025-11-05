import time
import schedule
from sqlalchemy.orm import Session
from sqlalchemy import select, inspect
from database.database import SessionLocal, engine, init_db
from database.models import Document
from services import init_drive_service, list_files_in_folder, download_drive_file_content, get_drive_web_link
from services import (
    parse_docx, parse_doc, parse_rtf, parse_md, parse_txt,
    clean_text, split_text_into_chunks
)
from services import indexing_pipe_line
from config import SERVICE_ACCOUNT_FILE, TARGET_FOLDER_ID
from uuid import uuid4
from database.models import DocumentChunk
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Variables check
if (SERVICE_ACCOUNT_FILE is None) or (TARGET_FOLDER_ID is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")

def process_new_documents():
    """Checks for new documents in Google Drive, processes them, and adds them to the database."""
    logger.info("Checking for new documents...")
    
    inspector = inspect(engine)
    if not inspector.has_table("documents"):
        logger.info("Database not found, initializing...")
        init_db()
        logger.info("Database initialized.")

    drive_service = init_drive_service(SERVICE_ACCOUNT_FILE)
    if not drive_service:
        logger.error("Failed to initialize Google Drive service.")
        return

    files_in_folder = list_files_in_folder(drive_service, TARGET_FOLDER_ID)
    if not files_in_folder:
        logger.info("No files found in the target folder.")
        return

    with SessionLocal() as session:
        existing_ids = session.execute(
            select(Document.drive_file_id)
        ).scalars().all()
        # To Set для быстрой проверки 
        existing_ids_set = set(existing_ids)

        for file_info in files_in_folder:
            file_id = file_info["ID Файла"]
            file_name = file_info["Имя Файла"]
            file_mime_type = file_info["MIME Тип"]
            # Check doc in database
            if file_id in existing_ids_set:
                # Документ уже проиндексирован, пропускаем
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
            session.commit()
            session.refresh(new_document)
            document_id = new_document.document_id
            logger.info(f"Added new document to the database: {file_name}")

            raw_content_bytes = download_drive_file_content(drive_service, file_id, file_name)
            if raw_content_bytes is None:
                continue

            if file_mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text_content = parse_docx(raw_content_bytes)
            elif file_mime_type == "application/msword":
                text_content = parse_doc(raw_content_bytes)
            elif file_mime_type == "application/rtf":
                text_content = parse_rtf(raw_content_bytes)
            elif file_mime_type == "text/markdown":
                text_content = parse_md(raw_content_bytes.decode('utf-8'))
            elif file_mime_type == "text/plain":
                text_content = parse_txt(raw_content_bytes.decode('utf-8'))
            else:
                logger.warning(f"Unsupported file format: {file_mime_type}")
                
                continue

            logger.info(f"Parsed content from file: {file_name}")
            cleaned_content = clean_text(text_content)
            chunks = split_text_into_chunks(cleaned_content)
            logger.info(f"Split content into {len(chunks)} chunks.")
            
            chunk_objects_to_process = []
            for chunk in chunks:
                qdrant_uuid = str(uuid4())
                new_document_chunk = DocumentChunk(
                    document_id=document_id,
                    content=chunk,
                    qdrant_id=qdrant_uuid,
                )
                session.add(new_document_chunk)
                chunk_objects_to_process.append(new_document_chunk)
            session.commit()
            logger.info("Committed new chunks to the database.")


            #TEST embedding_model = get_embedding_model()
            # qdrant_client = get_qdrant_client()
            # create_qdrant_collection(qdrant_client, 768)

            #TEST texts_to_embed = [c.content for c in chunk_objects_to_process]
            #TEST embeddings = vectorize_chunks(embedding_model, texts_to_embed)
            # points = create_qdrant_points(chunk_objects_to_process, embeddings)
            # upsert_points_to_qdrant(qdrant_client, points)
            indexing_pipe_line.run(chunk_objects_to_process)
            
    logger.info("Finished checking for new documents.")


if __name__ == "__main__":
    # For monitoring of new files
    # schedule.every(1).minutes.do(process_new_documents)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
    process_new_documents()