"""
Тестовый скрипт для анализа проблемы с garbage detection.
Скачивает только файл 0412-1-2023-ЭОМ с Google Drive, парсит его и сохраняет чанки.
"""
import asyncio
import logging
from pathlib import Path

from src.config import SERVICE_ACCOUNT_FILE, TARGET_FOLDER_ID
from src.services.google_drive import (
    download_drive_file_content,
    init_drive_service,
    list_files_in_folder,
)
from src.services.document_processor_service import document_processor_service

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Variables check
if (SERVICE_ACCOUNT_FILE is None) or (TARGET_FOLDER_ID is None):
    raise ValueError("Переменные не найдены. Проверьте файл .env.")


async def test_single_document():
    """Тестирует обработку одного документа 0412-1-2023-ЭОМ."""
    logger.info("="*60)
    logger.info("ТЕСТ: Обработка документа 0412-1-2023-ЭОМ")
    logger.info("="*60)
    
    # Инициализация Google Drive (пропускаем для локального теста)
    # drive_service = init_drive_service(SERVICE_ACCOUNT_FILE)
    # if not drive_service:
    #     logger.error("Failed to initialize Google Drive service.")
    #     return
    
    # Используем локальный файл
    file_name = "0412-1-2023-ЭОМ -Электроснабжение,электроосвещение и силовое электрооборудование.pdf"
    local_path = Path(file_name)
    
    if not local_path.exists():
        logger.error(f"❌ Файл '{file_name}' не найден локально!")
        return
        
    file_id = "local_test_id"
    file_mime_type = "application/pdf"
    
    logger.info(f"\n✅ Используем локальный файл: {file_name}")
    
    # Читаем файл
    logger.info(f"\n{'='*60}")
    logger.info("ЧТЕНИЕ ФАЙЛА")
    logger.info(f"{'='*60}")
    
    try:
        with open(local_path, "rb") as f:
            raw_content_bytes = f.read()
    except Exception as e:
        logger.error(f"❌ Ошибка чтения файла: {e}")
        return
    
    logger.info(f"✅ Файл прочитан: {len(raw_content_bytes)} байт")
    
    # Обрабатываем документ через pipeline
    logger.info(f"\n{'='*60}")
    logger.info("ОБРАБОТКА ДОКУМЕНТА (парсинг + чанкинг)")
    logger.info(f"{'='*60}")
    
    chunks_data = document_processor_service.process_document(
        file_content=raw_content_bytes,
        file_name=file_name,
        mime_type=file_mime_type,
        document_id=999,  # Фиктивный ID для теста
        document_title=file_name,
        max_pages=1
    )
    
    if not chunks_data:
        logger.warning(f"❌ Could not create chunks for document: {file_name}")
        return
    
    logger.info(f"\n✅ Создано чанков: {len(chunks_data)}")
    
    # Сохраняем результаты в файл
    output_file = "chunks_verification_new.txt"
    logger.info(f"\n{'='*60}")
    logger.info(f"СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В {output_file}")
    logger.info(f"{'='*60}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Анализ документа: {file_name}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Общая информация:\n")
        f.write(f"  - Размер файла: {len(raw_content_bytes)} байт\n")
        f.write(f"  - Создано чанков: {len(chunks_data)}\n")
        f.write(f"  - MIME тип: {file_mime_type}\n")
        f.write(f"\n{'='*80}\n\n")
        
        # Записываем все чанки
        for i, chunk_data in enumerate(chunks_data, 1):
            f.write(f"[Chunk {i} | Page {chunk_data.get('page_number', '?')}]\n")
            f.write(f"Документ: {file_name}. Стр: {chunk_data.get('page_number', '?')}.\n")
            f.write(f"{chunk_data['content']}\n\n")
            f.write(f"{'-'*80}\n\n")
    
    logger.info(f"✅ Результаты сохранены в {output_file}")
    
    # Анализ первого чанка
    logger.info(f"\n{'='*60}")
    logger.info("АНАЛИЗ ПЕРВОГО ЧАНКА")
    logger.info(f"{'='*60}")
    
    if chunks_data:
        first_chunk = chunks_data[0]['content']
        logger.info(f"\nДлина: {len(first_chunk)} символов")
        logger.info(f"Первые 500 символов:")
        logger.info(f"{'-'*60}")
        logger.info(first_chunk[:500])
        logger.info(f"{'-'*60}")
        
        # Проверка на мусор
        import re
        allowed_pattern = re.compile(r'[а-яА-ЯёЁa-zA-Z0-9\s.,;:!?()\[\]{}"\'\-_+=/\\|%@#№$€£*<>&]')
        garbage_chars = [c for c in first_chunk[:1000] if not allowed_pattern.match(c)]
        
        logger.info(f"\n📊 Анализ на мусор (первые 1000 символов):")
        logger.info(f"   Мусорных символов: {len(garbage_chars)}")
        logger.info(f"   Процент мусора: {len(garbage_chars) / min(1000, len(first_chunk)) * 100:.2f}%")
        
        if garbage_chars:
            logger.info(f"   Примеры мусорных символов (первые 50):")
            logger.info(f"   {garbage_chars[:50]}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ ТЕСТ ЗАВЕРШЕН!")
    logger.info(f"{'='*60}")
    logger.info(f"\n📝 Проверьте файл {output_file} для детального анализа")


if __name__ == "__main__":
    asyncio.run(test_single_document())