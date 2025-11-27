"""
Тестовый скрипт для проверки чанкинга первых двух документов.
Скачивает документы из Google Drive, парсит их и создает чанки,
сохраняя результаты в текстовый файл для верификации.
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.services.google_drive import GoogleDriveService
from src.services.document_parser import document_parser
from src.services.chunking_service import chunking_service
from src.config import GOOGLE_DRIVE_FOLDER_ID, GOOGLE_APPLICATION_CREDENTIALS

def main():
    print("=== Тест чанкинга первых двух документов ===\n")
    
    # 1. Инициализация Google Drive сервиса
    print("1. Подключение к Google Drive...")
    drive_service = GoogleDriveService(
        credentials_path=GOOGLE_APPLICATION_CREDENTIALS,
        folder_id=GOOGLE_DRIVE_FOLDER_ID
    )
    
    # 2. Получение списка файлов
    print("2. Получение списка файлов...")
    files = drive_service.list_files()
    
    if not files:
        print("Файлы не найдены!")
        return
    
    # Берем только первые 2 файла
    files_to_process = files[:2]
    print(f"Найдено файлов: {len(files)}, обрабатываем первые 2:\n")
    
    for idx, file_info in enumerate(files_to_process, 1):
        print(f"  {idx}. {file_info['name']} ({file_info['mimeType']})")
    
    print()
    
    # 3. Обработка каждого файла
    all_results = []
    
    for file_info in files_to_process:
        file_name = file_info['name']
        file_id = file_info['id']
        mime_type = file_info['mimeType']
        
        print(f"\n{'='*80}")
        print(f"Обработка: {file_name}")
        print(f"{'='*80}\n")
        
        # Скачивание
        print("  - Скачивание...")
        content = drive_service.download_file(file_id)
        
        if not content:
            print(f"  ⚠️  Не удалось скачать файл")
            continue
        
        # Парсинг
        print("  - Парсинг...")
        parsed_pages = document_parser.parse_file(content, file_name, mime_type)
        
        if not parsed_pages:
            print(f"  ⚠️  Не удалось распарсить файл")
            continue
        
        print(f"  ✓ Извлечено страниц: {len(parsed_pages)}")
        
        # Чанкинг
        print("  - Создание чанков...")
        chunks = chunking_service.create_chunks_with_metadata(
            parsed_pages=parsed_pages,
            document_id=file_info.get('id', 0),  # Используем file_id как document_id
            document_title=file_name,
            chunk_size=1000,
            overlap=150
        )
        
        print(f"  ✓ Создано чанков: {len(chunks)}")
        
        # Сохранение результатов
        all_results.append({
            'file_name': file_name,
            'file_id': file_id,
            'pages': len(parsed_pages),
            'chunks': chunks
        })
    
    # 4. Сохранение в файл
    output_file = "test_chunks_verification.txt"
    print(f"\n{'='*80}")
    print(f"Сохранение результатов в {output_file}...")
    print(f"{'='*80}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ТЕСТОВАЯ ВЕРИФИКАЦИЯ ЧАНКИНГА (ПЕРВЫЕ 2 ДОКУМЕНТА)\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"Документ: {result['file_name']}\n")
            f.write(f"File ID: {result['file_id']}\n")
            f.write(f"Страниц: {result['pages']}\n")
            f.write(f"Чанков: {len(result['chunks'])}\n")
            f.write(f"{'='*80}\n\n")
            
            for idx, chunk in enumerate(result['chunks'], 1):
                f.write(f"[Chunk {idx} | Page {chunk.get('page_number', 'N/A')}]\n")
                f.write(f"{chunk['content']}\n\n")
                f.write("-" * 80 + "\n\n")
    
    print(f"✓ Результаты сохранены в {output_file}")
    print(f"\nВсего обработано документов: {len(all_results)}")
    print(f"Всего создано чанков: {sum(len(r['chunks']) for r in all_results)}")

if __name__ == "__main__":
    main()
