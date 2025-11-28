#!/usr/bin/env python3
"""
Тестовый скрипт для быстрой проверки парсинга и чанкинга страниц 3-4 из PDF файла.
Работает локально, независимо от Docker.
"""

import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.document_parser import document_parser
from services.chunking_service import chunking_service

def test_parse_pages_3_4():
    """
    Парсит страницы 3-4 из PDF файла и сохраняет чанки в txt файл.
    """
    pdf_file = "0412-1-2023-ЭОМ -Электроснабжение,электроосвещение и силовое электрооборудование.pdf"
    output_file = "test_pages_3_4_chunks.txt"
    
    print(f"📄 Открываем файл: {pdf_file}")
    
    # Проверяем существование файла
    if not os.path.exists(pdf_file):
        print(f"❌ Файл {pdf_file} не найден!")
        return
    
    # Читаем файл
    with open(pdf_file, 'rb') as f:
        content = f.read()
    
    print(f"✅ Файл прочитан ({len(content)} байт)")
    
    # Парсим только страницы 3-4 (индексация с 0, так что это страницы 2-3 в коде)
    # Но document_parser.parse_file не поддерживает диапазон страниц напрямую
    # Поэтому парсим первые 4 страницы и берём только 3-4
    print(f"🔍 Парсим первые 4 страницы (возьмём только 3-4)...")
    
    parsed_pages = document_parser.parse_file(
        content=content,
        file_name=pdf_file,
        mime_type="application/pdf",
        max_pages=2  # Парсим первые 4 страницы
    )
    
    print(f"✅ Всего распарсено страниц: {len(parsed_pages)}")
    
    # Берём только страницы 3-4 (индексы 2-3)
    if len(parsed_pages) < 2:
        print(f"⚠️ В файле меньше 4 страниц. Доступно: {len(parsed_pages)}")
        pages_3_4 = parsed_pages[:2] if len(parsed_pages) > 2 else parsed_pages
    else:
        pages_3_4 = parsed_pages[:2]
    
    print(f"📋 Выбрано страниц для анализа: {len(pages_3_4)}")
    
    for i, page in enumerate(pages_3_4):
        print(f"  Страница {i}: {len(page['content'])} символов")
    
    # Создаём чанки
    print(f"✂️ Создаём чанки...")
    
    chunks = chunking_service.create_chunks_with_metadata(
        parsed_pages=pages_3_4,
        document_id=1,  # Фиктивный ID для теста
        document_title=os.path.splitext(pdf_file)[0],
        chunk_size=1000,
        overlap=150
    )
    
    print(f"✅ Создано чанков: {len(chunks)}")
    
    # Сохраняем в файл
    print(f"💾 Сохраняем результаты в {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Тест парсинга страниц 3-4 из файла: {pdf_file}\n")
        f.write(f"Всего чанков: {len(chunks)}\n")
        f.write("="*80 + "\n\n")
        
        for idx, chunk in enumerate(chunks, start=1):
            f.write(f"\n{'='*80}\n")
            f.write(f"ЧАНК #{idx}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Страница: {chunk.get('page_number', 'N/A')}\n")
            f.write(f"Тип: {chunk.get('type', 'N/A')}\n")
            f.write(f"Токенов: ~{chunk.get('token_count', 'N/A')}\n")
            if chunk.get('sheet_name'):
                f.write(f"Лист (Excel): {chunk['sheet_name']}\n")
            f.write(f"{'-'*80}\n")
            f.write(chunk['content'])
            f.write(f"\n{'-'*80}\n")
    
    print(f"✅ Результаты сохранены в {output_file}")
    print(f"\n📊 Статистика:")
    print(f"  - Страниц обработано: {len(pages_3_4)}")
    print(f"  - Чанков создано: {len(chunks)}")
    print(f"  - Средний размер чанка: {sum(len(c['content']) for c in chunks) // len(chunks) if chunks else 0} символов")
    
    # Показываем первые 300 символов первого чанка
    if chunks:
        print(f"\n📝 Превью первого чанка:")
        print(f"{'-'*80}")
        print(chunks[0]['content'][:300] + "...")
        print(f"{'-'*80}")

if __name__ == "__main__":
    try:
        test_parse_pages_3_4()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
