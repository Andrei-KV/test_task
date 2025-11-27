# Гибридный Chunking - Краткая Шпаргалка

## Быстрый старт

### 1. Базовое использование (Recursive - рекомендуется)

```python
from src.services.chunking_service_hybrid import chunking_service_syntactic

chunks = chunking_service_syntactic.create_chunks_with_metadata(
    parsed_pages=parsed_pages,
    document_id=123,
    document_title="Название документа",
    chunk_size=500,
    overlap=100
)
```

### 2. Semantic chunking (для критичных документов)

```python
from src.services.chunking_service_hybrid import chunking_service_semantic

chunks = chunking_service_semantic.create_chunks_with_metadata(
    parsed_pages=parsed_pages,
    document_id=123,
    document_title="Медицинский протокол",
    chunk_size=500,
    overlap=100,
    use_semantic_for_pages=[5, 10, 15]  # Только для этих страниц
)
```

## Ключевые отличия от старого ChunkingService

| Фича | Старый | Новый (Hybrid) |
|------|--------|----------------|
| Overlap между страницами | ✅ Есть | ✅ Есть (сохранено!) |
| Контекстные заголовки | ✅ Есть | ✅ Есть |
| Обработка таблиц | ✅ Есть | ✅ Есть |
| Recursive splitting | ❌ Нет | ✅ Есть |
| Semantic chunking | ❌ Нет | ✅ Опционально |

## Когда использовать что?

### Recursive (по умолчанию)
- ✅ Все обычные документы
- ✅ Массовая обработка
- ✅ Скорость важна
- ⚡ Быстро

### Semantic (выборочно)
- ✅ Критичные документы (медицина, право)
- ✅ Документы с частой сменой тем
- ✅ Качество важнее скорости
- 🐌 Медленно (в 10-20 раз)

## Миграция

### Вариант 1: Замена в chunking_service.py

```python
# src/services/chunking_service.py
from src.services.chunking_service_hybrid import chunking_service_syntactic as chunking_service
```

Весь остальной код работает без изменений!

### Вариант 2: Обновление document_processor_service.py

```python
# src/services/document_processor_service.py
from src.services.chunking_service_hybrid import chunking_service_syntactic

class DocumentProcessorService:
    def process_document(self, ...):
        chunks_data = chunking_service_syntactic.create_chunks_with_metadata(...)
```

## Тестирование

```bash
# Запустить тест
poetry run python test_hybrid_chunking.py

# Проверить на реальном документе
poetry run python test_pipeline_quick.py
```

## Зависимости

### Базовые (уже есть)
- tiktoken
- nltk

### Для semantic chunking (опционально)
```bash
poetry add sentence-transformers
```

Если не установлено, semantic chunking автоматически откатится на recursive.

## Производительность

| Документ | Recursive | Semantic |
|----------|-----------|----------|
| 10 страниц | ~2 сек | ~20 сек |
| 100 страниц | ~20 сек | ~3-5 мин |

## Рекомендации

1. **По умолчанию:** Используйте `chunking_service_syntactic` (recursive)
2. **Для важных документов:** Используйте `use_semantic_for_pages=[...]` выборочно
3. **Не используйте semantic для всех документов** - слишком медленно
4. **Overlap между страницами работает в обоих режимах** - это важно!
