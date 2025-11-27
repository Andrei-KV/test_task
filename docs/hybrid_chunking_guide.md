# Гибридный Chunking Service - Руководство

## Обзор

`HybridChunkingService` объединяет три подхода к чанкингу:
1. **Recursive splitting** - иерархическое разбиение (по умолчанию)
2. **Semantic chunking** - разбиение по смене темы (опционально)
3. **Сохранение overlap между страницами** - критично для навигации

## Ключевые особенности

### 1. Сохранение `current_tail` (overlap между страницами)

```python
# Страница 1
chunks_page1 = ["...конец страницы 1"]
tail = ["последние предложения страницы 1"]

# Страница 2 НАЧИНАЕТСЯ с tail из страницы 1
chunks_page2 = [
    "последние предложения страницы 1 + начало страницы 2",
    "...продолжение страницы 2"
]
```

**Зачем это нужно?**
- Контекст не теряется на границе страниц
- LLM видит связь между страницами
- Улучшает качество ответов на вопросы, охватывающие несколько страниц

### 2. Recursive Splitting

**Иерархия разделителей:**
```
1. "\n\n\n" - разделы документа
2. "\n\n"   - параграфы
3. "\n"     - строки
4. ". "     - предложения (nltk)
```

**Пример:**
```
Входной текст:
"Раздел 1\n\n\nПараграф 1.\n\nПараграф 2."

Шаг 1: Разбить по \n\n\n → ["Раздел 1", "Параграф 1.\n\nПараграф 2."]
Шаг 2: Разбить по \n\n → ["Раздел 1", "Параграф 1.", "Параграф 2."]
Шаг 3: Если часть слишком большая, разбить по предложениям
```

### 3. Semantic Chunking (опционально)

**Как работает:**
1. Разбивает текст на предложения
2. Получает embeddings для каждого предложения
3. Вычисляет cosine similarity между соседними предложениями
4. Если similarity < 0.5 → граница чанка (смена темы)

**Когда использовать:**
- ✅ Критичные документы (медицинские, юридические)
- ✅ Документы с частой сменой тем
- ❌ НЕ для массовой обработки (медленно)

## Использование

### Базовый пример (Recursive splitting)

```python
from src.services.chunking_service_hybrid import chunking_service_syntactic

# Обработка документа
chunks = chunking_service_syntactic.create_chunks_with_metadata(
    parsed_pages=[
        {'content': 'Текст страницы 1', 'page_number': 1, 'type': 'text'},
        {'content': 'Текст страницы 2', 'page_number': 2, 'type': 'text'},
    ],
    document_id=123,
    document_title="Инструкция по ТБ",
    chunk_size=500,
    overlap=100
)

# Результат:
# [
#   {'content': 'Документ: Инструкция по ТБ. Стр: 1.\nТекст...', 'page_number': 1, ...},
#   {'content': 'Документ: Инструкция по ТБ. Стр: 2.\n[overlap]Текст...', 'page_number': 2, ...},
# ]
```

### Semantic chunking для критичных страниц

```python
from src.services.chunking_service_hybrid import chunking_service_semantic

# Semantic chunking только для страниц 5, 10, 15
chunks = chunking_service_semantic.create_chunks_with_metadata(
    parsed_pages=parsed_pages,
    document_id=123,
    document_title="Медицинский протокол",
    chunk_size=500,
    overlap=100,
    use_semantic_for_pages=[5, 10, 15]  # Только эти страницы
)
```

### Глобальный semantic chunking

```python
from src.services.chunking_service_hybrid import HybridChunkingService

# Создаем сервис с semantic chunking для ВСЕХ текстовых страниц
semantic_service = HybridChunkingService(use_semantic=True)

chunks = semantic_service.create_chunks_with_metadata(
    parsed_pages=parsed_pages,
    document_id=123,
    document_title="Учебник",
    chunk_size=500,
    overlap=100
)
```

## Интеграция в DocumentProcessorService

```python
# src/services/document_processor_service.py

from src.services.chunking_service_hybrid import (
    chunking_service_syntactic,
    chunking_service_semantic
)

class DocumentProcessorService:
    def process_document(
        self, 
        file_content: bytes, 
        file_name: str, 
        mime_type: str,
        document_id: int,
        document_title: str,
        use_semantic: bool = False  # Новый параметр
    ):
        # Парсинг
        parsed_pages = document_parser.parse_file(...)
        
        # Выбор сервиса чанкинга
        chunking_service = chunking_service_semantic if use_semantic else chunking_service_syntactic
        
        # Чанкинг
        chunks_data = chunking_service.create_chunks_with_metadata(
            parsed_pages=parsed_pages,
            document_id=document_id,
            document_title=document_title,
            chunk_size=500,
            overlap=100
        )
        
        return chunks_data
```

## Сравнение производительности

| Метод | Скорость | Качество | Когда использовать |
|-------|----------|----------|-------------------|
| **Recursive** | ⚡⚡⚡ Быстро | ⭐⭐⭐ Хорошо | По умолчанию для всех документов |
| **Semantic** | 🐌 Медленно (в 10-20 раз) | ⭐⭐⭐⭐⭐ Отлично | Критичные документы, точность важнее скорости |

## Тестирование

Создайте тестовый скрипт:

```python
# test_hybrid_chunking.py

from src.services.chunking_service_hybrid import (
    chunking_service_syntactic,
    chunking_service_semantic
)

# Тестовые данные
test_pages = [
    {
        'content': 'Пожарная безопасность очень важна. Необходимо соблюдать правила. Теперь об электробезопасности. Электричество опасно.',
        'page_number': 1,
        'type': 'text'
    },
    {
        'content': 'Продолжение темы электробезопасности. Важно знать правила.',
        'page_number': 2,
        'type': 'text'
    }
]

# Тест 1: Recursive
print("=== RECURSIVE CHUNKING ===")
chunks_recursive = chunking_service_syntactic.create_chunks_with_metadata(
    parsed_pages=test_pages,
    document_id=1,
    document_title="Тест",
    chunk_size=100,
    overlap=30
)

for i, chunk in enumerate(chunks_recursive):
    print(f"\nChunk {i+1} (Page {chunk['page_number']}):")
    print(chunk['content'][:200])

# Тест 2: Semantic
print("\n\n=== SEMANTIC CHUNKING ===")
chunks_semantic = chunking_service_semantic.create_chunks_with_metadata(
    parsed_pages=test_pages,
    document_id=1,
    document_title="Тест",
    chunk_size=100,
    overlap=30
)

for i, chunk in enumerate(chunks_semantic):
    print(f"\nChunk {i+1} (Page {chunk['page_number']}):")
    print(chunk['content'][:200])
```

## Миграция с текущего ChunkingService

### Вариант 1: Постепенная миграция

```python
# Оставьте старый сервис для совместимости
from src.services.chunking_service import chunking_service as old_chunking_service

# Используйте новый для новых документов
from src.services.chunking_service_hybrid import chunking_service_syntactic as new_chunking_service

# В коде выбирайте нужный
if use_new_chunking:
    chunks = new_chunking_service.create_chunks_with_metadata(...)
else:
    chunks = old_chunking_service.create_chunks_with_metadata(...)
```

### Вариант 2: Полная замена

```python
# src/services/chunking_service.py

# Импортируем новый сервис под старым именем
from src.services.chunking_service_hybrid import chunking_service_syntactic as chunking_service

# Весь остальной код работает без изменений!
```

## FAQ

**Q: Нужно ли переобрабатывать старые документы?**
A: Нет, если вас устраивает качество. Новый сервис можно использовать только для новых документов.

**Q: Semantic chunking требует GPU?**
A: Нет, работает на CPU, но медленнее. Для ускорения можно использовать GPU.

**Q: Можно ли комбинировать recursive и semantic для одного документа?**
A: Да! Используйте параметр `use_semantic_for_pages=[5, 10]` для выборочного применения.

**Q: Сохраняется ли overlap между страницами в semantic режиме?**
A: Да! Это ключевая фича, которая работает в обоих режимах.
