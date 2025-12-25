# Этап 1: Сборка зависимостей
FROM python:3.12-slim as builder

# Установка Poetry
RUN pip install poetry

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Установка зависимостей с помощью Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction

# Pre-download NLTK data to avoid runtime network issues
RUN python -m nltk.downloader punkt punkt_tab stopwords -d /usr/share/nltk_data

# Pre-download tiktoken encoding
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Этап 2: Создание конечного образа
FROM python:3.12-slim

# Установка системных зависимостей, необходимых для работы приложения
RUN apt-get update && apt-get install -y \
    tesseract-ocr-rus \
    pandoc \
    poppler-utils \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка переменных окружения
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Копирование установленных зависимостей из этапа сборки
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data
COPY --from=builder /app/tiktoken_cache /app/tiktoken_cache

# Копирование исходного кода приложения
COPY src ./src
COPY main.py .
COPY rag-test-task-743ae7e0d95d.json .
COPY process_new_documents_test.py .
COPY delete_document.py .
COPY clear_databases.py .

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7150", "--workers", "4"]
