# Анализ зависимостей проекта

## Используемые зависимости (НЕОБХОДИМЫЕ)

### Core Framework
- ✅ `fastapi` - основной веб-фреймворк
- ✅ `uvicorn` - ASGI сервер
- ✅ `python-dotenv` - загрузка переменных окружения

### Database & Storage
- ✅ `sqlalchemy` - ORM для PostgreSQL
- ✅ `asyncpg` - async драйвер для PostgreSQL
- ✅ `redis` - кэширование и сессии
- ✅ `opensearch-py` - векторный поиск

### Document Processing
- ✅ `PyMuPDF` (fitz) - парсинг PDF
- ✅ `pytesseract` - OCR для изображений
- ✅ `Pillow` - обработка изображений
- ✅ `pypandoc` - конвертация документов
- ✅ `pandoc` - зависимость для pypandoc
- ✅ `python-docx` - парсинг DOCX
- ✅ `striprtf` - парсинг RTF
- ✅ `pandas` - обработка Excel таблиц
- ✅ `openpyxl` - чтение XLSX
- ✅ `xlrd` - чтение XLS
- ✅ `markdown` - парсинг Markdown
- ✅ `beautifulsoup4` - парсинг HTML
- ✅ `pdf2image` - конвертация PDF в изображения для OCR

### Text Processing
- ✅ `tiktoken` - подсчет токенов для OpenAI моделей
- ✅ `nltk` - сегментация предложений

### AI/ML
- ✅ `google-genai` - Gemini API для embeddings и LLM
- ✅ `google-api-python-client` - Google Drive API

### Utilities
- ✅ `tenacity` - retry логика для API
- ✅ `aiohttp` - async HTTP клиент
- ✅ `grpcio` - gRPC для Google APIs

## ЛИШНИЕ зависимости (МОЖНО УДАЛИТЬ)

### 1. Qdrant (УДАЛИТЬ - мигрировали на OpenSearch)
- ❌ `qdrant-client` - **НЕ ИСПОЛЬЗУЕТСЯ**, заменен на OpenSearch
  - Размер: ~50MB
  - Используется в: НИГДЕ (был в старом rag_service.py)

### 2. Sentence Transformers (УДАЛИТЬ - используем Gemini)
- ❌ `sentence-transformers` - **НЕ ИСПОЛЬЗУЕТСЯ**, embeddings через Gemini API
  - Размер: ~500MB (включая модели)
  - Зависимости: torch, transformers, scipy, numpy
  - Используется в: НИГДЕ

### 3. PyTorch (УДАЛИТЬ - не нужен без sentence-transformers)
- ❌ `torch` - **НЕ ИСПОЛЬЗУЕТСЯ**, был нужен для sentence-transformers
  - Размер: ~800MB (CPU версия)
  - Используется в: НИГДЕ

### 4. Transformers (УДАЛИТЬ - не нужен без sentence-transformers)
- ❌ `transformers` - **НЕ ИСПОЛЬЗУЕТСЯ**, был нужен для sentence-transformers
  - Размер: ~400MB
  - Используется в: НИГДЕ

### 5. SciPy (УДАЛИТЬ - не нужен без sentence-transformers)
- ❌ `scipy` - **НЕ ИСПОЛЬЗУЕТСЯ**, был нужен для sentence-transformers
  - Размер: ~100MB
  - Используется в: НИГДЕ

### 6. NumPy (ОСТОРОЖНО - проверить)
- ⚠️ `numpy` - **ВОЗМОЖНО НЕ ИСПОЛЬЗУЕТСЯ**
  - Размер: ~50MB
  - Был нужен для sentence-transformers и torch
  - Pandas зависит от numpy, но подтянет свою версию автоматически
  - **РЕКОМЕНДАЦИЯ**: Удалить из явных зависимостей, pandas подтянет сам

### 7. OpenAI (УДАЛИТЬ - используем Gemini)
- ❌ `openai` - **НЕ ИСПОЛЬЗУЕТСЯ**, используем только Gemini
  - Размер: ~10MB
  - Используется в: НИГДЕ (был в старых версиях)

### 8. Langchain Text Splitters (УДАЛИТЬ - используем свой chunking)
- ❌ `langchain-text-splitters` - **НЕ ИСПОЛЬЗУЕТСЯ**, есть свой ChunkingService
  - Размер: ~20MB
  - Используется в: НИГДЕ

### 9. Docx2Python (УДАЛИТЬ - используем python-docx)
- ❌ `docx2python` - **НЕ ИСПОЛЬЗУЕТСЯ**, используем python-docx
  - Размер: ~5MB
  - Используется в: НИГДЕ

### 10. PDFPlumber (УДАЛИТЬ - используем PyMuPDF)
- ❌ `pdfplumber` - **НЕ ИСПОЛЬЗУЕТСЯ**, используем PyMuPDF (fitz)
  - Размер: ~10MB
  - Используется в: НИГДЕ

## Итоговая экономия памяти

### Критические (ОБЯЗАТЕЛЬНО удалить):
1. `torch` - ~800MB
2. `sentence-transformers` - ~500MB
3. `transformers` - ~400MB
4. `scipy` - ~100MB
5. `qdrant-client` - ~50MB

**Итого: ~1.85 GB экономии**

### Дополнительные (РЕКОМЕНДУЕТСЯ удалить):
6. `numpy` - ~50MB (удалить из явных, pandas подтянет)
7. `langchain-text-splitters` - ~20MB
8. `openai` - ~10MB
9. `pdfplumber` - ~10MB
10. `docx2python` - ~5MB

**Итого дополнительно: ~95 MB**

## ОБЩАЯ ЭКОНОМИЯ: ~1.95 GB

## Рекомендуемый pyproject.toml (очищенный)

```toml
[tool.poetry.dependencies]
python = ">=3.12,<3.15"

# Core Framework
"fastapi" = ">=0.121.2,<0.122.0"
"uvicorn" = ">=0.38.0,<0.39.0"
"python-dotenv" = ">=1.2.1,<2.0.0"

# Database & Storage
"sqlalchemy" = ">=2.0.44,<3.0.0"
"asyncpg" = ">=0.29.0,<1.0.0"
"redis" = ">=7.0.1,<8.0.0"
"opensearch-py" = "^3.1.0"

# Document Processing
"PyMuPDF" = ">=1.24.0,<2.0.0"
"pytesseract" = ">=0.3.10,<0.4.0"
"Pillow" = ">=10.0.0,<11.0.0"
"pypandoc" = ">=1.15,<2.0"
"pandoc" = ">=2.4,<3.0"
"python-docx" = "^1.2.0"
"striprtf" = "^0.0.29"
"pandas" = ">=2.3.3,<3.0.0"
"openpyxl" = ">=3.1.5,<4.0.0"
"xlrd" = ">=2.0.2,<3.0.0"
"markdown" = ">=3.9,<4.0"
"beautifulsoup4" = ">=4.14.2,<5.0.0"
"pdf2image" = "^1.17.0"

# Text Processing
"tiktoken" = ">=0.12.0,<0.13.0"
"nltk" = ">=3.8.1,<4.0.0"

# AI/ML
"google-genai" = ">=1.49.0,<2.0.0"
"google-api-python-client" = ">=2.187.0,<3.0.0"

# Utilities
"tenacity" = "^8.2.3"
"aiohttp" = "^3.13.2"
"grpcio" = ">=1.76.0,<2.0.0"
```

## Команды для очистки

```bash
# 1. Обновить pyproject.toml (удалить лишние зависимости)
# 2. Обновить lock файл
poetry lock

# 3. Переустановить зависимости
poetry install

# 4. Пересобрать Docker образ
docker-compose build --no-cache app
```
