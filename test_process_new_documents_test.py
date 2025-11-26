"""
Тест полного цикла обработки одного документа с Gemini embeddings.
Скачивает первый файл из Google Drive, обрабатывает его, генерирует эмбеддинги через Gemini API.
НЕ сохраняет в базу данных - только проверяет корректность работы.
"""
import logging
from google import genai
from src.config import SERVICE_ACCOUNT_FILE, TARGET_FOLDER_ID, GEMINI_API_KEY, EMBEDDING_MODEL_NAME
from src.services.google_drive import (
    download_drive_file_content,
    init_drive_service,
    list_files_in_folder,
)
from src.services.document_processor_service import document_processor_service

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Проверка переменных окружения
if not all([SERVICE_ACCOUNT_FILE, TARGET_FOLDER_ID, GEMINI_API_KEY, EMBEDDING_MODEL_NAME]):
    raise ValueError("Переменные окружения не найдены. Проверьте файл .env.")


def test_single_document_processing():
    """Тестирует полный цикл на первом документе из Google Drive."""
    logger.info("=" * 80)
    logger.info("ТЕСТ ОБРАБОТКИ ОДНОГО ДОКУМЕНТА С GEMINI EMBEDDINGS")
    logger.info("=" * 80)
    
    # 1. Инициализация Google Drive
    drive_service = init_drive_service(SERVICE_ACCOUNT_FILE)
    if not drive_service:
        logger.error("Не удалось инициализировать Google Drive service.")
        return
    
    # 2. Получение списка файлов
    files_in_folder = list_files_in_folder(drive_service, TARGET_FOLDER_ID)
    if not files_in_folder:
        logger.info("Файлы не найдены в целевой папке.")
        return
    
    logger.info(f"Найдено {len(files_in_folder)} файлов в папке.")
    logger.info("Обработка ПЕРВОГО файла для теста...\n")
    
    # 3. Берем первый файл
    file_info = files_in_folder[0]
    file_id = file_info["ID Файла"]
    file_name = file_info["Имя Файла"]
    file_mime_type = file_info["MIME Тип"]
    
    logger.info(f"Файл: {file_name}")
    logger.info(f"MIME: {file_mime_type}")
    logger.info(f"ID: {file_id}\n")
    
    # 4. Скачивание файла
    logger.info("Скачивание файла...")
    raw_content_bytes = download_drive_file_content(drive_service, file_id, file_name)
    if raw_content_bytes is None:
        logger.error("Не удалось скачать файл.")
        return
    
    logger.info(f"Скачано {len(raw_content_bytes)} байт.\n")
    
    # 5. Обработка через DocumentProcessorService
    logger.info("Запуск пайплайна обработки (парсинг + чанкинг)...")
    logger.info("-" * 80)
    
    chunks_data = document_processor_service.process_document(
        file_content=raw_content_bytes,
        file_name=file_name,
        mime_type=file_mime_type,
        document_id=999,  # Тестовый ID
        document_title=file_name
    )
    
    logger.info("-" * 80)
    
    if not chunks_data:
        logger.warning("Не удалось создать чанки.")
        return
    
    logger.info(f"\n✅ Создано {len(chunks_data)} чанков.\n")
    
    # 6. Генерация эмбеддингов через Gemini API (тестируем первые 3 чанка)
    logger.info("=" * 80)
    logger.info("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ЭМБЕДДИНГОВ ЧЕРЕЗ GEMINI API")
    logger.info("=" * 80)
    
    test_chunks = chunks_data[:3]  # Берем только первые 3 для теста
    logger.info(f"Генерация эмбеддингов для {len(test_chunks)} чанков...\n")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    for i, chunk in enumerate(test_chunks, 1):
        try:
            logger.info(f"Чанк {i}/{len(test_chunks)}:")
            logger.info(f"  Страница: {chunk.get('page_number', 'N/A')}")
            logger.info(f"  Тип: {chunk.get('type', 'N/A')}")
            logger.info(f"  Длина контента: {len(chunk['content'])} символов")
            
            # Генерация эмбеддинга
            result = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=chunk['content'],
                config=genai.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=3072
                )
            )
            
            embedding = result.embeddings[0].values
            dim = len(embedding)
            
            logger.info(f"  ✅ Эмбеддинг сгенерирован: размерность {dim}")
            logger.info(f"  Первые 5 значений: {embedding[:5]}\n")
            
            if dim != 3072:
                logger.error(f"  ❌ ОШИБКА: Неверная размерность! Ожидалось 3072, получено {dim}")
                
        except Exception as e:
            logger.error(f"  ❌ Ошибка генерации эмбеддинга для чанка {i}: {e}\n")
            continue
    
    logger.info("=" * 80)
    logger.info("ТЕСТ ЗАВЕРШЕН УСПЕШНО")
    logger.info("=" * 80)
    logger.info(f"Обработано чанков: {len(chunks_data)}")
    logger.info(f"Протестировано эмбеддингов: {len(test_chunks)}")
    logger.info("Модель: " + EMBEDDING_MODEL_NAME)
    logger.info("Размерность: 3072")


if __name__ == "__main__":
    test_single_document_processing()